"""Validate bundled GLM-5.2 NextN weights without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

_FLOAT_DTYPES = {"BF16", "F16", "F32"}
_EXPERT_SCHEMES = {"W8A8_DYNAMIC", "W4A8_DYNAMIC"}
_QUANT_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_second",
    ".weight_offset",
    ".weight_offset_second",
    ".weight_scale_inv",
    ".scale_bias",
    ".input_scale",
    ".input_scale_inv",
)


def _read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_header(path):
    try:
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError("missing safetensors header length")
            size = struct.unpack("<Q", size_bytes)[0]
            if not 0 < size <= min(100_000_000, path.stat().st_size - 8):
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(size))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}")
    return {name: info for name, info in header.items() if name != "__metadata__"}


class _Checkpoint:
    def __init__(self, root):
        self.root, self.headers = root, {}
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            self.weight_map = _read_json(index_path).get("weight_map")
            if not isinstance(self.weight_map, dict) or not all(
                isinstance(name, str) and isinstance(shard, str)
                for name, shard in self.weight_map.items()
            ):
                raise ValueError(f"Invalid weight_map in {index_path}")
        else:
            path = root / "model.safetensors"
            if not path.is_file():
                raise ValueError(
                    "Bundled MTP validation needs model.safetensors.index.json "
                    "or model.safetensors"
                )
            self.headers[path.name] = _read_header(path)
            self.weight_map = {name: path.name for name in self.headers[path.name]}

    def tensor(self, name):
        shard = self.weight_map[name]
        relative = Path(shard)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Checkpoint shard lies outside model directory: {shard}")
        # HF snapshots deliberately symlink their shards into the blobs cache.
        # Validate the index path, while allowing normal symlink resolution.
        path = self.root / relative
        if shard not in self.headers:
            self.headers[shard] = _read_header(path)
        info = self.headers[shard].get(name)
        if not isinstance(info, dict) or not isinstance(info.get("dtype"), str):
            raise ValueError(f"Missing tensor header for {name} in {shard}")
        return info


def validate_mtp_checkpoint(model_path, draft_quantization="unquant"):
    """Return a summary or raise ValueError for an unsupported bundled draft.

    Only MTP shard headers are opened for indexed checkpoints. The default
    accepts actual floating tensors, including checkpoints without a ModelSlim
    description; absence of a description alone never implies floating weights.
    """
    if draft_quantization not in ("unquant", "modelslim"):
        raise ValueError("MTP quantization must be unquant or modelslim")
    root = Path(model_path).expanduser().resolve()
    config = _read_json(root / "config.json")
    if config.get("model_type") != "glm_moe_dsa" or "GlmMoeDsaForCausalLM" not in (
        config.get("architectures") or ()
    ):
        raise ValueError("Bundled MTP requires GLM-5.2 / GlmMoeDsaForCausalLM")
    if (
        type(config.get("num_nextn_predict_layers")) is not int
        or config["num_nextn_predict_layers"] != 1
    ):
        raise ValueError("GLM bundled MTP requires num_nextn_predict_layers=1")
    layer_id, experts = config.get("num_hidden_layers"), config.get("n_routed_experts")
    if any(type(value) is not int or value <= 0 for value in (layer_id, experts)):
        raise ValueError("GLM MTP requires positive layer and routed-expert counts")
    prefix = f"model.layers.{layer_id}."
    checkpoint = _Checkpoint(root)
    names = {
        name
        for name in checkpoint.weight_map
        if name.startswith(prefix)
        and not name.startswith(
            (prefix + "shared_head.head.", prefix + "embed_tokens.")
        )
    }
    if not names:
        raise ValueError(f"No bundled MTP tensors found under {prefix}")
    # Native loaders accept weight_packed as an alias for ModelSlim's weight.
    aliases = {name.replace(".weight_packed", ".weight"): name for name in names}
    if len(aliases) != len(names):
        raise ValueError(
            "MTP checkpoint contains duplicate weight/weight_packed tensors"
        )

    special = ("eh_proj", "enorm", "hnorm", "shared_head.norm")
    attention = ["kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o_proj"]
    attention += (
        ["q_a_proj", "q_a_layernorm", "q_b_proj"]
        if config.get("q_lora_rank") is not None
        else ["q_proj"]
    )
    required = {prefix + name + ".weight" for name in special}
    required.update(prefix + "self_attn." + name + ".weight" for name in attention)
    required.update(
        prefix + name + ".weight"
        for name in ("input_layernorm", "post_attention_layernorm", "mlp.gate")
    )
    # NextN still constructs its DSA indexer even when it can reuse target top-k.
    indexer = prefix + "self_attn.indexer."
    required.update(
        indexer + name + ".weight" for name in ("wq_b", "wk", "weights_proj", "k_norm")
    )
    kpool = config.get("index_kpool", 1)
    if type(kpool) is not int or kpool <= 0:
        raise ValueError("GLM MTP index_kpool must be a positive integer")
    if kpool > 1 or config.get("index_k_norm_type", "layer") != "rms":
        required.add(indexer + "k_norm.bias")
    if kpool > 1:
        required.update(
            indexer + name
            for name in ("index_kpool_compress_ape", "index_kpool_compress_gate")
        )
    if config.get("topk_method") == "noaux_tc":
        required.add(prefix + "mlp.gate.e_score_correction_bias")
    expert_weights = {
        f"{prefix}mlp.experts.{expert}.{projection}.weight"
        for expert in range(experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    required.update(expert_weights)
    if config.get("n_shared_experts", 0):
        required.update(
            f"{prefix}mlp.shared_experts.{projection}.weight"
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    missing = sorted(required - aliases.keys())
    if missing:
        raise ValueError(f"Incomplete bundled MTP checkpoint; missing {missing[0]}")
    description_path = root / "quant_model_description.json"
    description = _read_json(description_path) if description_path.is_file() else {}

    def dtype(name):
        return checkpoint.tensor(aliases[name])["dtype"]

    def scheme(name):
        return description.get(name, description.get(aliases[name]))

    if draft_quantization == "unquant":
        for name in sorted(aliases):
            if name.endswith(_QUANT_SUFFIXES) or scheme(name) not in (None, "FLOAT"):
                raise ValueError(
                    f"MTP tensor {name} is quantized; unquant cannot decode it. "
                    "Use MTP_DRAFT_QUANTIZATION=modelslim for a supported quantized draft"
                )
            if dtype(name) not in _FLOAT_DTYPES:
                raise ValueError(
                    f"Unquant MTP requires floating tensors; {name} is {dtype(name)}"
                )
        expert_scheme = "FLOAT"
    else:
        expert_schemes = {scheme(name) for name in expert_weights}
        if len(expert_schemes) != 1 or not expert_schemes <= _EXPERT_SCHEMES:
            raise ValueError(
                "ModelSlim MTP requires every routed-expert projection to use the same "
                "W8A8_DYNAMIC or W4A8_DYNAMIC scheme; FLOAT experts need unquant"
            )
        expert_scheme = next(iter(expert_schemes))
        group_size = description.get("group_size", 0)
        if expert_scheme == "W4A8_DYNAMIC":
            if isinstance(group_size, str):
                try:
                    group_size = int(group_size)
                except ValueError as exc:
                    raise ValueError("Invalid ModelSlim W4A8 group_size") from exc
            if type(group_size) is not int or group_size < 0:
                raise ValueError("Invalid ModelSlim W4A8 group_size")
        for name in sorted(aliases):
            if not name.endswith(".weight"):
                if dtype(name) not in _FLOAT_DTYPES:
                    raise ValueError(f"MTP auxiliary tensor must be floating: {name}")
                continue
            quant_type = scheme(name)
            if name in expert_weights:
                allowed = {expert_scheme}
            elif name in {prefix + part + ".weight" for part in special} or any(
                marker in name
                for marker in (
                    "layernorm",
                    ".mlp.gate.",
                    ".indexer.weights_proj.",
                    ".indexer.k_norm.",
                )
            ):
                allowed = {None, "FLOAT"}
            else:
                allowed = {None, "FLOAT", "W8A8_DYNAMIC"}
            if quant_type not in allowed:
                raise ValueError(
                    f"Unsupported ModelSlim MTP scheme for {name}: {quant_type}"
                )
            expected = _FLOAT_DTYPES if quant_type in (None, "FLOAT") else {"I8"}
            if dtype(name) not in expected:
                raise ValueError(
                    f"MTP dtype disagrees with {quant_type or 'unquant'} metadata: {name}"
                )
            if quant_type in _EXPERT_SCHEMES:
                suffixes = ["weight_scale", "weight_offset"]
                if quant_type == "W4A8_DYNAMIC":
                    suffixes.append("scale_bias")
                    if group_size:
                        suffixes.extend(("weight_scale_second", "weight_offset_second"))
                for suffix in suffixes:
                    component = name.removesuffix("weight") + suffix
                    if (
                        component not in aliases
                        or dtype(component) not in _FLOAT_DTYPES
                    ):
                        raise ValueError(
                            f"Missing floating MTP quantization component: {component}"
                        )
    return {
        "mtp_layer_id": layer_id,
        "draft_quantization": draft_quantization,
        "expert_quantization": expert_scheme,
        "mtp_tensor_count": len(names),
        "checked_shards": sorted(checkpoint.headers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument(
        "--draft-quantization", choices=("unquant", "modelslim"), default="unquant"
    )
    args = parser.parse_args()
    try:
        result = validate_mtp_checkpoint(args.model_path, args.draft_quantization)
    except ValueError as exc:
        parser.exit(2, f"GLM MTP checkpoint validation failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
