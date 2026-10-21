"""Framework entry point shared by module and expert offloaders."""

from abc import ABC
from contextlib import nullcontext


class BaseOffloader(ABC):
    def wrap_modules(
        self,
        all_modules_generator,
        submodule_accessor=None,
        whitelist_param_names_creator=None,
    ):
        return list(all_modules_generator)

    def post_init(self):
        pass

    def bind_model_runner(self, model_runner):
        """Optional model/graph integration after post-load finalization."""

    def forward_scope(self, forward_batch):
        """Host lifecycle around eager execution or graph replay."""
        return nullcontext()

    def eager_step_scope(self, forward_batch):
        """Device step hooks around an actual eager invocation, never replay."""
        return nullcontext()

    @property
    def forbid_copy_engine_usage(self):
        return False
