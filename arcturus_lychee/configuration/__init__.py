import tomllib
import tomli_w
from datetime import datetime, date, time
from dataclasses import fields
 
from arcturus_lychee.configuration.basic_template import TrainingConfiguration
 

# TOML cannot represent arbitrary Python objects (e.g. torch device/dtype) and
# has no null. We persist only TOML-serializable values.
_TOML_SCALARS = (bool, int, float, str, datetime, date, time)
 
 
def _is_toml_serializable(value) -> bool:
    if isinstance(value, _TOML_SCALARS):
        return True
    if isinstance(value, list):
        return all(_is_toml_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_toml_serializable(v) for k, v in value.items())
    return False
 
 
def save_config[T](config_obj: T, filepath: str) -> None:
    """Saves a dataclass instance and any dynamically added keys to a TOML file.
 
    Only TOML-serializable values are written; anything else is skipped rather
    than raising. In particular the environment-derived `device` / `dtype`
    fields are dropped on purpose - they are not TOML types, and they should
    follow the *loading* machine (recomputed by their default_factory), not be
    pinned to whatever machine happened to save the config.
    """
    # Using __dict__ instead of dataclasses.asdict() ensures we capture
    # any ad-hoc variables you attached after initialization.
    config_dict = {
        key: value
        for key, value in vars(config_obj).items()
        if _is_toml_serializable(value)
    }
 
    with open(filepath, "wb") as f:
        tomli_w.dump(config_dict, f)
 
 
def load_config[T](dataclass_cls: type[T], filepath: str) -> T:
    """Accepts any class 'T' and promises to return an instance of 'T'."""
    with open(filepath, "rb") as f:
        data = tomllib.load(f)
    
    known_fields = {f.name for f in fields(dataclass_cls)}
    dataclass_kwargs = {k: v for k, v in data.items() if k in known_fields}
    extra_kwargs = {k: v for k, v in data.items() if k not in known_fields}
    
    instance = dataclass_cls(**dataclass_kwargs)
    
    for key, value in extra_kwargs.items():
        setattr(instance, key, value)
        
    return instance

