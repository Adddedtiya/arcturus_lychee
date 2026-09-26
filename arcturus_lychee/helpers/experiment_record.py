"""The record/ directory of a run. It keeps all data that describes the run.

With this data, you can find what a run did, and start it again with one
change. This is the base of an ablation study.

The record does not know the task or the libraries of a project. Project
data, for example augmentations or class names, uses save_json() or
save_text() in the project code.
"""

import os
import sys
import json
import zipfile
import tomli_w

import torch
import torch.nn as nn

from typing import Any, Optional, Union

import arcturus_lychee
from arcturus_lychee.configuration import toml_compatible


class ExperimentRecord:
    """Write files into the record/ directory of one run.

    The logger makes this object and gives it as logger.record.
    Only rank 0 has a real record. The other ranks get a NullRecord.
    """

    def __init__(self, record_dir : str) -> None:
        self.record_dir = record_dir
        os.makedirs(self.record_dir, exist_ok = True)

    def path(self, file_name : str) -> str:
        """Return the path of a file in the record/ directory."""
        return os.path.join(self.record_dir, file_name)

    def save_text(self, file_name : str, lines : Union[str, list[str]]) -> None:
        """Write a text, or one line for each item of a list."""
        text = lines if isinstance(lines, str) else "\n".join(str(line) for line in lines)
        with open(self.path(file_name), "w", encoding = "utf-8") as file:
            file.write(text)
            file.write("\n")

    def save_toml(self, file_name : str, values : dict[str, Any]) -> list[str]:
        """Write a dict to a TOML file. A dict inside the dict becomes a TOML table.

        TOML cannot hold all values, for example None. The function does not
        write these values. It returns their names, for example "optimizer.foreach".
        """
        skipped : list[str] = []
        compatible          = self._toml_clean(values, "", skipped)
        with open(self.path(file_name), "wb") as file:
            tomli_w.dump(compatible, file)
        return skipped

    @staticmethod
    def _toml_clean(values : dict[str, Any], prefix : str, skipped : list[str]) -> dict[str, Any]:
        """Return a copy of the dict with only the values that TOML can hold.

        The function adds the full name of each removed value to skipped.
        """
        result = {}
        for key, value in values.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                result[key] = ExperimentRecord._toml_clean(value, f"{name}.", skipped)
                continue
            compatible, missing = toml_compatible({key: value})
            result.update(compatible)
            skipped.extend(f"{prefix}{m}" for m in missing)
        return result

    def save_json(self, file_name : str, values : Any) -> None:
        """Write a value to a JSON file. A value that JSON cannot hold becomes a string."""
        with open(self.path(file_name), "w", encoding = "utf-8") as file:
            json.dump(values, file, indent = 2, default = str)

    def save_code(self, extra_paths : Optional[list[str]] = None) -> None:
        """Write code.zip with the entry script and the full arcturus_lychee package.

        The zip file records the code of the run, also the values that are
        hard-coded. Git is not necessary. extra_paths can add more files or
        directories, for example a project package outside arcturus_lychee.
        """
        package_dir = os.path.dirname(os.path.abspath(arcturus_lychee.__file__))
        paths       = [package_dir] + list(extra_paths or [])

        # sys.argv[0] is the entry script. mp.spawn gives the same value to each process.
        entry_script = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        if os.path.isfile(entry_script):
            paths.append(entry_script)

        with zipfile.ZipFile(self.path("code.zip"), "w", compression = zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                self._add_to_zip(archive, os.path.abspath(path))

    @staticmethod
    def _add_to_zip(archive : zipfile.ZipFile, path : str) -> None:
        """Add a file, or a directory without __pycache__, to the zip file."""
        if os.path.isfile(path):
            archive.write(path, arcname = os.path.basename(path))
            return

        parent = os.path.dirname(path)
        for root, directories, files in os.walk(path):
            directories[:] = [d for d in directories if d != "__pycache__"]
            for name in files:
                if name.endswith(".pyc"):
                    continue
                full_path = os.path.join(root, name)
                archive.write(full_path, arcname = os.path.relpath(full_path, parent))

    def save_model(self, model : nn.Module) -> None:
        """Write model.txt with the structure of the model and the number of parameters."""
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.save_text("model.txt", [
            f"Class                : {type(model).__module__}.{type(model).__qualname__}",
            f"Parameters           : {total:,}",
            f"Trainable parameters : {trainable:,}",
            "",
            repr(model),
        ])

    def save_optimizer(
            self,
            optimizer : torch.optim.Optimizer,
            scheduler : Optional[Any] = None,
        ) -> list[str]:
        """Write optimizer.toml with the classes and the hyperparameters.

        The file has one table for each parameter group of the optimizer.
        The file does not contain the parameters themselves. The function
        returns the names of values that TOML cannot hold.
        """
        # In a parameter group, None means "use the default". The file does not show these values.
        values : dict[str, Any] = {"optimizer": {"class": type(optimizer).__qualname__}}
        for index, group in enumerate(optimizer.param_groups):
            values["optimizer"][f"group_{index}"] = {
                k: v for k, v in group.items() if k != "params" and v is not None
            }

        if scheduler is not None:
            state = {k: v for k, v in scheduler.state_dict().items() if not k.startswith("_")}
            values["scheduler"] = {"class": type(scheduler).__qualname__, **state}

        return self.save_toml("optimizer.toml", values)


class NullRecord:
    """A record that does nothing. The ranks other than rank 0 use it."""

    def path(self, file_name : str) -> str:                          return file_name
    def save_text(self, *args, **kwargs) -> None:                    pass
    def save_toml(self, *args, **kwargs) -> list[str]:               return []
    def save_json(self, *args, **kwargs) -> None:                    pass
    def save_code(self, *args, **kwargs) -> None:                    pass
    def save_model(self, *args, **kwargs) -> None:                   pass
    def save_optimizer(self, *args, **kwargs) -> list[str]:          return []
