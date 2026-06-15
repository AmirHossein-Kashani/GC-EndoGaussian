import importlib.util
import os

_PARAM_GROUPS = ("ModelParams", "OptimizationParams", "ModelHiddenParams", "PipelineParams")


def load_config(path):
    """Load a per-scene Python config file (e.g. arguments/endonerf/pulling.py).

    Replaces ``mmcv.Config.fromfile`` for this project. The config files are plain
    Python modules that define module-level dicts named after the four
    ``ParamGroup`` classes; this loader executes the module and returns a
    ``dict[str, dict]`` keyed by those group names. ``utils.params_utils.merge_hparams``
    consumes it directly.
    """
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("scene_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: getattr(module, name) for name in _PARAM_GROUPS if hasattr(module, name)}
