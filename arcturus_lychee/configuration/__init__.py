import tomllib
import tomli_w
from dataclasses import fields

from arcturus_lychee.configuration.basic_template import TrainingConfiguration

def save_config[T](config_obj: type[T], filepath: str):
    """Saves a dataclass instance and any dynamically added keys to a TOML file."""
    # Using __dict__ instead of dataclasses.asdict() ensures we capture 
    # any ad-hoc variables you attached after initialization.
    config_dict = dict(config_obj.__dict__)
    
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