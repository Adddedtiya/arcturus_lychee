import os
import pandas as pd

from datetime import datetime
from typing   import Union

import matplotlib.pyplot as plt

class DirectoryTrainingLogger:
    def __init__(
            self, 
            working_directory : str, 
            best_metric       : str, 
            higher_is_better  : bool             = True,
            experiment_name   : Union[str, None] = None
        ):

        # check if experiment_name is valid or not
        experiment_name = experiment_name if experiment_name else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # root directory
        working_directory = os.path.join(working_directory, experiment_name)
        self.root_dir = os.path.abspath(working_directory)
        os.makedirs(self.root_dir, exist_ok = True)
        
        # create directory specific to logs
        self.log_dir = self.__create_subdir('log')        
        self.train_path = os.path.join(self.log_dir, "train.csv")
        self.eval_path  = os.path.join(self.log_dir, "eval.csv")
        
        # Internal Metrics keys
        self.best_metric_key  = best_metric
        self.higher_is_better = higher_is_better
        
        # Internal flag to track the result of the most recent log call
        self._current_epoch = 0
        self._last_was_best = False

        self.train_df = pd.DataFrame()
        self.eval_df  = pd.DataFrame()
        self.best_value = float('-inf') if self.higher_is_better else float('inf')
        
        # application logs 
        self.log_file = os.path.join(self.root_dir, 'log_messages.txt')
        self.__append_log_file("| TIMESTAMP              | MESSAGE")

        # create directory specific to samples
        self.samples_dir = self.__create_subdir('samples')
        self.weights_dir = self.__create_subdir("weights")
        self.plots_dir   = self.__create_subdir("plots")

        # Start Your Training
        self.log("Training Start !")

    def __create_subdir(self, subdir_name : str) -> str:
        x = os.path.join(self.root_dir, subdir_name)
        os.makedirs(x, exist_ok = True)
        return x

    def __append_log_file(self, string : str) -> None:
        with open(self.log_file, 'a+') as file:
            file.write(string)
            file.write("\n")

    def log(self, message : str) -> None:
        
        # get the current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        formmated_message = f"| {current_time} | {message}"

        self.__append_log_file(formmated_message)
        print(formmated_message, flush = True)

    def print(self, message : str) -> None:
        self.log(message)
    
    def print_lines(self, messages : list[str]) -> None:
        for msg in messages:
            self.log(msg)

    def append(
            self, 
            train_metrics : dict[str, float], 
            eval_metrics  : Union[dict[str, float], None] = None,
            epoch         : Union[int, None]              = None
        ) -> None:

        # Reset the Flag
        self._last_was_best = False

        # get the epoch
        current_epoch = epoch if epoch else self._current_epoch + 1
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # add the values to logger
        train_row = pd.DataFrame([{
            "timestamp" : timestamp_str, 
            "epoch"     : current_epoch, 
            **train_metrics
        }])
        self.train_df = pd.concat([self.train_df, train_row], ignore_index = True)
        self.train_df.to_csv(self.train_path, index = False)

        # add eval values to metrics
        if eval_metrics is not None:

            # 2. Update and append the test values
            eval_row = pd.DataFrame([{
                "timestamp": timestamp_str, 
                "epoch"    : current_epoch, 
                **eval_metrics
            }])
            self.eval_df = pd.concat([self.eval_df, eval_row], ignore_index = True)
            self.eval_df.to_csv(self.eval_path, index = False)


            # 3. Update // Check if current epoch is the best value
            current_val = eval_metrics.get(
                self.best_metric_key, 
                float('-inf') if self.higher_is_better else float('inf')
            )
            if self.higher_is_better:
                if current_val > self.best_value:
                    self.best_value     = current_val
                    self._last_was_best = True
            else:
                if current_val < self.best_value:
                    self.best_value     = current_val
                    self._last_was_best = True

        self.__plot_dataframe(self.train_df, title = "Train Variables",      file_name = "train.png")
        self.__plot_dataframe(self.eval_df,  title = "Evaluation Variables", file_name = "eval.png")

        # the end update the current_epoch variable
        self._current_epoch = current_epoch
    
    def is_best(self) -> bool:
        return self._last_was_best

    def load_from_csv(self):
        """Loads data from CSVs and restores best_value accurately."""
        if os.path.exists(self.train_path):
            self.train_df = pd.read_csv(self.train_path)

        if os.path.exists(self.eval_path):
            
            # load the csv file
            self.eval_df = pd.read_csv(self.eval_path)

            # check if it even empty or not
            if not self.eval_df.empty and self.best_metric_key in self.eval_df.columns:
                
                # Get the extreme value from the existing history
                if self.higher_is_better:
                    self.best_value = self.eval_df[self.best_metric_key].max()
                else:
                    self.best_value = self.eval_df[self.best_metric_key].min()
            
            self.log(f"Resumed. Best {self.best_metric_key} so far: {self.best_value}")

    def __plot_dataframe(
            self, 
            df    : pd.DataFrame, 
            keys      : Union[str, list[str], None] = None,
            title     : Union[str, None]            = None,
            file_name : Union[str, None]            = None
        ) -> None:

        if file_name is None:
            self.log("[WARNING] Your Trying to Create a Plot without Output File !")
            return

        if df.empty:
            self.log(f"[WARNING] No data available to plot for {title} !")
            return

        # Determine which keys to plot
        meta_cols = {'epoch', 'timestamp'}
        available_keys = [c for c in df.columns if c not in meta_cols]
        
        if keys is None:
            keys_to_plot = available_keys
        else:
            if isinstance(keys, str): keys = [keys]
            keys_to_plot = [k for k in keys if k in available_keys]

        if not keys_to_plot:
            self.log(f"[WARNING] None of the requested keys {keys} were found in the logs.")
            return

        # Create subplots
        num_plots = len(keys_to_plot)
        fig, axes = plt.subplots(num_plots, 1, figsize = (10, 4 * num_plots), squeeze = False)

        fig.suptitle(title, fontsize = 16, y = 1.02)

        for i, key in enumerate(keys_to_plot):
            ax = axes[i, 0]
            ax.plot(df['epoch'], df[key], label = key, marker = 'o', linestyle = '-')
            ax.set_title(f"{key}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, file_name))
        plt.close()

    def get_weights_path(self, file_name : str) -> str:

        # return a file in model path
        fpath = os.path.join(self.weights_dir, file_name)
        return fpath


if __name__ == "__main__":
    print("Training Logging Testing Sample - Iterative Tests")
    
    # for debug
    import time
    import random
    
    # Initialize for a 'loss' metric (where lower is better)
    logger = DirectoryTrainingLogger(
        working_directory = "C:\\Users\\aditya\\Documents\\Projects\\TracedLight\\playbox_playground",
        best_metric       = 'accuracy',
        higher_is_better  = True,
        experiment_name   = None
    )

    # Optional: Load existing data
    # logger.load_from_csv()

    # Example training step
    for epoch in range(1, 40):

        logger.log(f"Running Epoch : {epoch}")
        train_stats = {"loss": 0.9 / epoch, "accuracy": 0.5 + (0.01 * epoch)}
        
        # Run evaluation every 2nd epoch
        eval_stats = {"loss": 1.0 / epoch, "accuracy": 0.45 + (0.01 * epoch)} if epoch % 2 == 0 else None
        
        logger.append(train_stats, eval_stats)

        if logger.is_best():
            logger.log(f"--- Epoch {epoch}: Best model detected! ---")

        rdelay = random.random()
        time.sleep(rdelay * 1.5)

    print("Testing Finished")