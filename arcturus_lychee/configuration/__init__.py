"""Save a configuration to a TOML file, and load it again.

The run directory keeps the configuration in record/configuration.toml.
Thus a later run can load the configuration, change one value, and start
again. This is the base of an ablation study.
"""

import os
import tomllib
import tomli_w

from datetime    import datetime, date, time
from dataclasses import fields
from typing      import Any

from arcturus_lychee.configuration.basic_template import TrainingConfiguration


# The value types that TOML can hold. TOML has no value for None.
_TOML_SCALARS = (bool, int, float, str, datetime, date, time)


def _is_toml_serializable(value : Any) -> bool:
    """Return True if TOML can hold the value."""
    if isinstance(value, _TOML_SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_toml_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_toml_serializable(v) for k, v in value.items())
    return False


def toml_compatible(values : dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Divide a dict into the values that TOML can hold and the names of the other values.

    The function returns (compatible_values, skipped_names).
    """
    compatible = {}
    skipped    = []
    for key, value in values.items():
        if _is_toml_serializable(value):
            compatible[key] = value
        else:
            skipped.append(key)
    return compatible, skipped


def save_config[T](config_obj : T, filepath : str) -> list[str]:
    """Write the configuration and its added attributes to a TOML file.

    The function writes only values that TOML can hold. It returns the names
    of the values that it did not write, for example values that are None.
    The caller can then show these names in the log.

    A tuple goes into the file as a list. load_config() gives it back as a list.
    """
    # vars() also gives the attributes that the entry script added.
    # dataclasses.asdict() gives only the fields.
    config_dict, skipped = toml_compatible(vars(config_obj))

    with open(filepath, "wb") as f:
        tomli_w.dump(config_dict, f)

    return skipped


def load_config[T](dataclass_cls : type[T], filepath : str) -> T:
    """Make an object of dataclass_cls from a TOML file, and return it.

    Keys that are fields of the dataclass go to the constructor. The other
    keys become attributes of the object. A field that is not in the file
    gets its default value.
    """
    with open(filepath, "rb") as f:
        data = tomllib.load(f)

    known_fields     = {f.name for f in fields(dataclass_cls)}
    dataclass_kwargs = {k: v for k, v in data.items() if k in known_fields}
    extra_kwargs     = {k: v for k, v in data.items() if k not in known_fields}

    instance = dataclass_cls(**dataclass_kwargs)

    for key, value in extra_kwargs.items():
        setattr(instance, key, value)

    return instance


def load_run_configuration[T](
        run_directory : str,
        dataclass_cls : type[T] = TrainingConfiguration,
    ) -> T:
    """Load the configuration of a previous run directory, and return it.

    The function reads record/configuration.toml. If that file does not exist,
    it reads configuration.toml in the run directory (old run directories).

    Example for an ablation:

        configuration = load_run_configuration("results/2026_09_26_10_00-baseline")
        configuration.learning_rate   = 1e-4
        configuration.experiment_name = "baseline_lr_1e-4"
    """
    candidates = [
        os.path.join(run_directory, "record", "configuration.toml"),
        os.path.join(run_directory, "configuration.toml"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return load_config(dataclass_cls, path)

    raise FileNotFoundError(
        f"The run directory '{run_directory}' has no configuration file. "
        f"Make sure that one of these files exists: {', '.join(candidates)}."
    )
