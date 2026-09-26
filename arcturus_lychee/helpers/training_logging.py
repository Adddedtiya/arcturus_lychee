import os
import copy
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from typing   import Union

from arcturus_lychee.configuration import TrainingConfiguration, save_config


class NullLogger:
    """A logger that does nothing. The ranks other than rank 0 use it.

    With this class, the trainer does not need an is_main_process() test
    before each log call. The methods are the same as in
    DirectoryTrainingLogger, but they do nothing. The trainer also saves
    checkpoints only on rank 0. Thus get_weights_path() returns only the file name.
    """
    run_name = ""

    def log(self, message : Union[str, list[str]]) -> None:  pass
    def append(self, *args, **kwargs) -> None:              pass
    def load_from_csv(self) -> None:                        pass
    def is_best(self) -> bool:                              return False
    def get_weights_path(self, file_name : str) -> str:     return file_name


class DirectoryTrainingLogger:
    """Write the logs, the metrics, the plots, and the checkpoints of one run.

    Each run gets a run directory: <working_directory>/<run_name>/
    If prefix_date is True, run_name is "<date>-<experiment_name>".
    If prefix_date is False, run_name is experiment_name, and the logger uses
    the directory again if it exists. This is necessary to continue a run.

    The logger does not change the configuration. It keeps the run name in
    self.run_name, and writes it into configuration.toml.
    """

    def __init__(self, configuration : TrainingConfiguration) -> None:

        # With the date prefix, the run name uses "-" between the date and the name.
        # Thus the name itself uses "_". Without the prefix, the name does not change,
        # so the name of a previous run finds the same directory again.
        if configuration.prefix_date:
            experiment_name = configuration.experiment_name.replace('-', '_')
            current_time    = datetime.now().strftime("%Y_%m_%d_%H_%M")
            run_name        = f"{current_time}-{experiment_name}"
            run_name        = self._unused_name(configuration.working_directory, run_name)
        else:
            run_name = configuration.experiment_name
        self.run_name = run_name

        # run directory
        self.root_dir = os.path.abspath(os.path.join(configuration.working_directory, run_name))
        os.makedirs(self.root_dir, exist_ok = True)

        # metric files
        self.log_dir    = self._create_subdir('log')
        self.train_path = os.path.join(self.log_dir, "train.csv")
        self.eval_path  = os.path.join(self.log_dir, "eval.csv")

        # best metric
        self.best_metric_key  = configuration.metric_to_track
        self.higher_is_better = configuration.higher_is_better
        self.best_value       = float('-inf') if self.higher_is_better else float('inf')
        self.best_epoch       = None
        self._last_was_best   = False

        # metric history
        self.train_df = pd.DataFrame()
        self.eval_df  = pd.DataFrame()

        # text log
        self.log_file = os.path.join(self.root_dir, 'log_messages.txt')
        self._append_log_file("| TIMESTAMP              | MESSAGE")

        # other directories
        self.samples_dir = self._create_subdir('samples')
        self.weights_dir = self._create_subdir("weights")
        self.plots_dir   = self._create_subdir("plots")

        self.log(f"Run directory: {self.root_dir}")

        # The saved configuration also contains the run name.
        saved_configuration          = copy.copy(configuration)
        saved_configuration.run_name = run_name
        skipped = save_config(saved_configuration, os.path.join(self.root_dir, 'configuration.toml'))
        if skipped:
            self.log(
                "NOTE: configuration.toml does not contain these values, "
                f"because TOML cannot hold them: {', '.join(skipped)}."
            )

    @staticmethod
    def _unused_name(working_directory : str, run_name : str) -> str:
        """Return run_name, or run_name with a number if that directory already exists.

        Two runs that start in the same minute then get different directories.
        """
        candidate = run_name
        number    = 2
        while os.path.exists(os.path.join(working_directory, candidate)):
            candidate = f"{run_name}_{number}"
            number   += 1
        return candidate

    def _create_subdir(self, subdir_name : str) -> str:
        path = os.path.join(self.root_dir, subdir_name)
        os.makedirs(path, exist_ok = True)
        return path

    def _append_log_file(self, text : str) -> None:
        with open(self.log_file, 'a+', encoding = "utf-8") as file:
            file.write(text)
            file.write("\n")

    def log(self, message : Union[str, list[str]]) -> None:
        """Write one message, or a list of messages, to the console and to log_messages.txt."""
        messages = [message] if isinstance(message, str) else message
        for text in messages:
            current_time      = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            formatted_message = f"| {current_time} | {text}"
            self._append_log_file(formatted_message)
            print(formatted_message, flush = True)

    def append(
            self,
            train_metrics : dict[str, float],
            eval_metrics  : Union[dict[str, float], None],
            epoch         : int,
        ) -> None:
        """Add the metrics of one epoch to the CSV files and to the plots.

        The epoch is the number that the files show. The trainer gives
        epoch + 1, so the first epoch is 1. If eval_metrics is not None,
        the logger also updates the best value. is_best() then gives the result.
        """
        self._last_was_best = False
        timestamp           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        train_row     = pd.DataFrame([{"timestamp": timestamp, "epoch": epoch, **train_metrics}])
        self.train_df = pd.concat([self.train_df, train_row], ignore_index = True)
        self.train_df.to_csv(self.train_path, index = False)

        if eval_metrics is not None:
            eval_row     = pd.DataFrame([{"timestamp": timestamp, "epoch": epoch, **eval_metrics}])
            self.eval_df = pd.concat([self.eval_df, eval_row], ignore_index = True)
            self.eval_df.to_csv(self.eval_path, index = False)

            if self.best_metric_key not in eval_metrics:
                self.log(f"NOTE: The evaluation metrics do not contain '{self.best_metric_key}'.")
            else:
                value = eval_metrics[self.best_metric_key]
                if self.higher_is_better:
                    improved = value > self.best_value
                else:
                    improved = value < self.best_value
                if improved:
                    self.best_value     = value
                    self.best_epoch     = epoch
                    self._last_was_best = True

        self._plot_dataframe(self.train_df, title = "Train Variables",      file_name = "train.png")
        self._plot_dataframe(self.eval_df,  title = "Evaluation Variables", file_name = "eval.png")

    def is_best(self) -> bool:
        """Return True if the last call of append() gave a new best value."""
        return self._last_was_best

    def load_from_csv(self) -> None:
        """Load the metric history from the CSV files, and restore the best value.

        Use this function to continue a run in the same run directory.
        """
        if os.path.exists(self.train_path):
            self.train_df = pd.read_csv(self.train_path)

        if os.path.exists(self.eval_path):
            self.eval_df = pd.read_csv(self.eval_path)

            if not self.eval_df.empty and self.best_metric_key in self.eval_df.columns:
                column = self.eval_df[self.best_metric_key]
                row    = column.idxmax() if self.higher_is_better else column.idxmin()
                self.best_value = float(column[row])
                self.best_epoch = int(self.eval_df['epoch'][row])

            self.log(f"The metric history is loaded. The best {self.best_metric_key} is {self.best_value}.")

    def _plot_dataframe(self, df : pd.DataFrame, title : str, file_name : str) -> None:
        """Make one plot for each metric column. Without data, the function does nothing."""
        if df.empty:
            return

        keys_to_plot = [c for c in df.columns if c not in ('epoch', 'timestamp')]
        if not keys_to_plot:
            return

        fig, axes = plt.subplots(len(keys_to_plot), 1, figsize = (10, 4 * len(keys_to_plot)), squeeze = False)
        fig.suptitle(title, fontsize = 16, y = 1.02)

        for i, key in enumerate(keys_to_plot):
            ax = axes[i, 0]
            ax.plot(df['epoch'], df[key], label = key, marker = 'o', linestyle = '-')
            ax.set_title(key)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle = '--', alpha = 0.6)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, file_name))
        plt.close(fig)

    def get_weights_path(self, file_name : str) -> str:
        """Return the path of a file in the weights directory."""
        return os.path.join(self.weights_dir, file_name)


if __name__ == "__main__":
    print("Logger demo. The demo writes a run directory in 'playbox_playground'.")

    import time
    import random

    config = TrainingConfiguration()
    config.working_directory = "playbox_playground"
    config.experiment_name   = "logger_smoke_test"
    config.metric_to_track   = "accuracy"
    config.higher_is_better  = True

    logger = DirectoryTrainingLogger(config)

    for epoch in range(1, 10):
        logger.log(f"Epoch {epoch}")
        train_stats = {"loss": 0.9 / epoch, "accuracy": 0.5 + (0.01 * epoch)}

        # An evaluation after each second epoch
        eval_stats = {"loss": 1.0 / epoch, "accuracy": 0.45 + (0.01 * epoch)} if epoch % 2 == 0 else None

        logger.append(train_stats, eval_stats, epoch = epoch)
        if logger.is_best():
            logger.log(f"Epoch {epoch} has the best accuracy.")

        time.sleep(random.random() * 0.2)

    print("The demo is complete.")
