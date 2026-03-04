import numpy as np

import torch
import torch.nn             as nn
import torch.nn.functional  as F

from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.grad_scaler import GradScaler
from arcturus_lychee.helpers import DirectoryTrainingLogger, SpeedTimer, generate_report, generate_confusion_matrix
from torch.utils.data   import DataLoader
from collections import defaultdict

from typing import Union

class WrapperForClassification:
    def __init__(
            self,
            model        : nn.Module,
            logger       : DirectoryTrainingLogger,
            total_epochs : int,
            device       : torch.device = 'cpu',
        ):

        # create the logger for the training session
        self.log = logger

        # default device and ensure everything in the device
        self.device = torch.device(device)

        # ensure model is device and the training code
        self.model = model.to(self.device)

        # create the optimizer
        self.optimizer = AdamW(self.model.parameters(), lr = 1e-5)

        # set the scheduler too
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max = total_epochs)

        # Cross Entropy Loss for Classification
        self.criterion = nn.CrossEntropyLoss()

        # AMP for bfloat16 training (if cuda)
        self.use_amp = str(device).startswith('cuda')
        self.gradient_scaler = GradScaler(self.device, enabled = self.use_amp)

        # also save the amount epoch we want to train
        self.total_epochs = total_epochs

    def save_state(self, fpath : str) -> None:

        # save the parameter states
        state = {
            'model_state'     : self.model.state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'scaler_state'    : self.gradient_scaler.state_dict()
        }
        torch.save(state, fpath)
    
    def load_state(self, fpath : str) -> None:

        # load the parameters
        state = torch.load(fpath, map_location = self.device, weights_only = True)

        # set the parameters
        self.model.load_state_dict(state['model_state'], strict = True)
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.gradient_scaler.load_state_dict(state['scaler_state'])
    
    def __compute_top_n(self, x_real : torch.Tensor, y_pred : torch.Tensor, top_k : tuple[int, ...] = (1, 5)) -> dict[str, float]:

        # sanity check dont worry
        y_pred = y_pred.detach()

        # result buffer
        results = {}

        # sanity buffer
        with torch.no_grad():
            max_k = max(top_k)
            batch_size = x_real.size(0)

            # 1. Get the indices of the top K predictions
            # _, pred shape: [batch_size, maxk]
            _, pred = y_pred.topk(max_k, 1, True, True)
            pred = pred.t() # Transpose to [maxk, batch_size]
            
            # 2. Compare with targets
            # targets.view(1, -1) shape: [1, batch_size]
            # correct shape: [maxk, batch_size] (boolean)
            correct = pred.eq(x_real.view(1, -1).expand_as(pred))

            # check for each K
            for k in top_k:
                # Sum up the correct hits for the top K
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res = float(correct_k.item() / batch_size)
                results[f"top-{k}"] = res 
        
        # return the results
        return results

    def __compute_metrics(self, x_real : torch.Tensor, y_pred : torch.Tensor) -> dict[str, float]:

        # in classification we mesure the top-1 and top-5 accuracy
        report = self.__compute_top_n(
            x_real, y_pred,
            top_k = (1, 5)
        )

        return report
    
    def __train_single_batch(self, input_tensor : torch.Tensor, target_tensor : torch.Tensor) -> dict[str, float]:

        # ensure the model is in training mode
        self.model.train()  
        
        # ensure we are using autocast AMP
        with torch.autocast(device_type = str(self.device), dtype = torch.bfloat16, enabled = self.use_amp):

            # move the data to device
            input_tensor  = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)
            
            # forward pass to the model
            prediction : torch.Tensor = self.model(input_tensor)
            loss : torch.Tensor = self.criterion(prediction, target_tensor)
        
        # compute the loss and scale it back
        loss = self.gradient_scaler.scale(loss)
        loss.backward()

        # step the optmizer too
        self.gradient_scaler.step(self.optimizer)

        # update the scaler
        self.gradient_scaler.update()

        # reset the optmizer state
        self.optimizer.zero_grad(set_to_none = True)

        # compute the metrics too while we are there
        metrics = self.__compute_metrics(target_tensor, prediction)
        metrics['loss'] = float(loss.item())

        # return the metrics from this batch
        return metrics
    
    def __test_single_batch(self, input_tensor : torch.Tensor, target_tensor : torch.Tensor) -> dict[str, float]:

        # ensure the model is in eval mode
        self.model.eval()

        # ensure we are in amp mode and no gradients
        with torch.autocast(device_type = str(self.device), dtype = torch.bfloat16, enabled = self.use_amp), torch.no_grad():

            # move the data to device
            input_tensor  = input_tensor.to(self.device)
            target_tensor = target_tensor.to(self.device)

            # forward pass through the model
            prediction : torch.Tensor = self.model(input_tensor)

            # compute just the metrics
            metrics = self.__compute_metrics(target_tensor, prediction)
        
        # return the metrics only
        return metrics
    
    def __calculate_metric_averages(self, metrics : list[dict[str, float]]) -> dict[str, float]:
        
        # create default dictonary
        averages = defaultdict(list)
        for item in metrics:
            for key, value in item.items():
                averages[key].append(value)
        
        # compute the value for each item in dictonary
        for key in averages:
            averages[key] = float(sum(averages[key]) / len(averages[key]))

        # convert to normal dict before going back
        return dict(averages)

    def train_single_epoch(self, dataloader : DataLoader, enable_tqdm : bool = True) -> dict[str, float]:
        
        # setup state and variables
        active_metrics = []
        self.model.train()

        # training loop
        for index, (input_tensor, target_tensor) in enumerate(tqdm(dataloader, disable = not enable_tqdm)):

            # forward pass
            batch_metrics = self.__train_single_batch(input_tensor, target_tensor)
            
            # append
            active_metrics.append(batch_metrics)
        
        # compute the average for all
        batch_averages = self.__calculate_metric_averages(active_metrics)
        return batch_averages


    def test_single_epoch(self, dataloader : DataLoader, enable_tqdm : bool = True) -> dict[str, float]:
        
        # setup states and variables
        active_metrics = []
        self.model.eval()

        # testing loop
        for index, (input_tensor, target_tensor) in enumerate(tqdm(dataloader, disable = not enable_tqdm)):

            # forward pass
            batch_metrics = self.__test_single_batch(input_tensor, target_tensor)
            
            # append
            active_metrics.append(batch_metrics)
        
        # compute the average for all
        batch_averages = self.__calculate_metric_averages(active_metrics)
        return batch_averages
    
    def __get_current_learning_rate(self) -> float:

        # convert to float
        current_rates : list[torch.Tensor] = self.scheduler.get_last_lr()
        current_rates = [float(x) for x in current_rates]
        
        # get the amount and total
        total_rates  = len(current_rates)
        summed_rated = sum(current_rates)

        # compute the average
        average_rate = float(summed_rated / total_rates)
        return average_rate

    
    def run_single_epoch(
            self, 
            train_dataloader : DataLoader, 
            test_dataloader  : DataLoader, 
            current_epoch    : int , 
            enable_tqdm      : bool = True,
            test_every       : int  = 1 
        ) -> tuple[dict[str, float], Union[dict[str, float], None], SpeedTimer]:

        # and start the clock
        total_runtime = SpeedTimer()
        
        # train the model first
        train_stats = self.train_single_epoch(train_dataloader, enable_tqdm)

        # check if we need to test too (if N epoch, or first one)
        test_stats = None
        if (current_epoch % test_every) == 0 or (current_epoch == 0):
            test_stats = self.test_single_epoch(test_dataloader, enable_tqdm)

        # get the current learning rate
        train_stats['lr'] = self.__get_current_learning_rate()

        # update the lr too
        self.scheduler.step()

        # stop the clock
        total_runtime.stop()
        
        # return the values
        return train_stats, test_stats, total_runtime


    def run_everything(
            self,
            train_dataloader : DataLoader, 
            test_dataloader  : DataLoader, 
            enable_tqdm      : bool = True,
            test_every       : int  = 1,
            start_epoch      : int  = 0,
            save_every       : int  = 1,
        ) -> None:
        
        self.log.print("Starting Training !")

        # we will run the whole training pipeline
        for current_epoch in range(start_epoch, self.total_epochs):

            # write on the log
            self.log.print(f"Current Epoch : {current_epoch + 1}")

            # run the training and test for the epoch
            train_stats, eval_stats, total_runtime = self.run_single_epoch(
                train_dataloader = train_dataloader,
                test_dataloader  = test_dataloader,
                current_epoch    = current_epoch,
                enable_tqdm      = enable_tqdm,
                test_every       = test_every
            )

            # add it to the log
            self.log.append(train_stats, eval_stats)

            # check if the current epoch is the best
            if self.log.is_best():
                self.log.print(f"--- Best model detected! ---")

                # write the model to file
                best_model_path = self.log.get_weights_path('best.pt')
                self.save_state(best_model_path)
            
            # check if we need to save in the current epoch
            if (current_epoch % save_every) == 0 or (current_epoch == 0):

                latest_model_path = self.log.get_weights_path("latest.pt")
                self.save_state(latest_model_path)
            
            # estimate the total training time
            leftover_epochs = self.total_epochs - current_epoch
            eta_message = SpeedTimer.estimate_time(total_runtime, leftover_epochs)
            self.log.print(eta_message)

        # save the model final state too
        final_model_path = self.log.get_weights_path("final.pt")
        self.save_state(final_model_path)

        # yay
        self.log.print("Training is Complete !")

    def test_model(
            self,
            test_dataloader  : DataLoader, 
            report_prefix    : str                    = "",
            enable_tqdm      : bool                   = True,
            class_names      : Union[list[str], None] = None
        ) -> None:

        # test the model on the a dataset
        predicted_results   : list[int] = []
        ground_truth_values : list[int] = []

        # internal accuracy report
        accuracy_reports : list[dict[str, float]] = []

        # write into log
        self.log.print("") # spacing...
        self.log.print(f"Testing For : {report_prefix}")

        # testing loop
        for index, (input_tensor, target_tensor) in enumerate(tqdm(test_dataloader, disable = not enable_tqdm)):

            # ensure we are in amp mode and no gradients
            with torch.autocast(device_type = str(self.device), dtype = torch.bfloat16, enabled = self.use_amp), torch.no_grad():

                # move the data to device
                input_tensor  : torch.Tensor = input_tensor.to(self.device)
                target_tensor : torch.Tensor = target_tensor.to(self.device)

                # forward pass through the model
                prediction : torch.Tensor = self.model(input_tensor)

                # mesure the top-k
                top_k_values = self.__compute_top_n(target_tensor, prediction, top_k = (1, 3, 5))
                accuracy_reports.append(top_k_values)

            # break down the prediction
            # target     : [N,]
            # prediction : [N, C]

            # add the targets
            ground_truth_values += target_tensor.detach().cpu().tolist() 

            # add predictions
            preds = torch.argmax(prediction, dim = 1)
            predicted_results += preds.detach().cpu().tolist()
        
        # compute the top-k
        average_top_k = self.__calculate_metric_averages(accuracy_reports)
        for top_k in average_top_k.keys():
            self.log.print(f"Accuracy in {top_k} : {average_top_k[top_k]:.2f}")
        self.log.print("")

        # compute the metrics now !
        self.log.print("Report :")
        report_lines = generate_report(ground_truth_values, predicted_results, class_names)
        self.log.print_lines(report_lines)
        self.log.print("") # padding lines

        # write the confusion matrix
        self.log.print("Confusion Matrix :")
        confusion_lines = generate_confusion_matrix(ground_truth_values, predicted_results, class_names)
        self.log.print_lines(confusion_lines)
        self.log.print("") # padding lines

