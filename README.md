# arcturus_lychee

A PyTorch training template for one machine with one or more GPUs.

The template does the repetitive parts of a training: processes, DDP, precision, seeds, loaders, logs, checkpoints, and a full record of each run. You write the parts that are specific to your project: the dataset, the model, and the training step.

The template has no command-line options and no configuration files to write. All values are Python variables in the entry script. Each run saves these values, the code, and the environment. Thus you can load a run again, change one value, and start an ablation.

## Contents

1. [Principles](#1-principles)
2. [Installation](#2-installation)
3. [Start a training](#3-start-a-training)
4. [Folder layout](#4-folder-layout)
5. [Configuration](#5-configuration)
6. [The run directory](#6-the-run-directory)
7. [Ablation studies](#7-ablation-studies)
8. [Continue a stopped run](#8-continue-a-stopped-run)
9. [Read the timing columns](#9-read-the-timing-columns)
10. [Adapt the template to a new task](#10-adapt-the-template-to-a-new-task)
11. [Rules for multi-GPU code](#11-rules-for-multi-gpu-code)
12. [Singularity and SLURM](#12-singularity-and-slurm)
13. [Writing rules for this repository](#13-writing-rules-for-this-repository)

## 1. Principles

- **Plain Python start.** You start each training with `python main.py`. The template does not use torchrun or `python -m`. The same command works in SLURM, Singularity, Docker, and a shell.
- **One code path.** With one GPU, the training operates in one process without DDP. With two or more GPUs, the template starts one process for each GPU. The trainer code is the same in both cases.
- **Configuration in Python.** A dataclass holds the values. `build_config()` in the entry script changes them.
- **Everything is recorded.** Each run directory has a `record/` directory. It holds the configuration, a zip file of the code, the environment, the model, the optimizer, and a summary.
- **Readable classes.** A trainer is one explicit class. All state is on `self`. The template has no hidden framework and no hooks.
- **Shared code does not know the task.** The shared tools work with images, volumes, audio, and other data.

## 2. Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13. `pyproject.toml` pins PyTorch for CUDA 12.6.

1. Install uv.
2. Open a shell in the repository directory.
3. Install the dependencies:

```bash
uv sync
```

4. If your cluster needs a different CUDA version, change the `pytorch` index URL and the `torch` and `torchvision` versions in `pyproject.toml`. Then do `uv sync` again.

## 3. Start a training

1. Open `main.py`.
2. In `build_config()`, set the dataset directories, the number of classes, and the other values.
3. Start the training:

```bash
python main.py
```

The number of processes depends on the visible GPUs:

| Visible GPUs | Result |
|---|---|
| 0 (CPU) or 1 | One process. No DDP. |
| 2 or more | One process for each GPU. DDP with NCCL. |

To use fewer GPUs, set `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py
```

`batch_size` is the batch for each GPU. The global batch is `batch_size` x the number of GPUs.

Do not start the script with torchrun, or with `srun` and more than one task. The script then stops with an error. Without this error, each external process starts one process for each GPU, and the GPUs get too many processes.

To debug with breakpoints, use one GPU. The training then operates in one process:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

## 4. Folder layout

```
main.py                          Entry script: build_config() and worker(). Copy it for each experiment.
arcturus_lychee/
  configuration/                 SHARED: TrainingConfiguration, save_config, load_config, load_run_configuration
  helpers/                       SHARED: the tools of the template. Do not change them for a project.
    distributed.py               launch(), ranks, get_device(), reductions across ranks
    data_loading.py              build_dataloader(), move_to_device(), TimedLoader
    metrics.py                   MetricAccumulator
    model_setup.py               wrap_model(), unwrap_model(), Precision
    training_logging.py          DirectoryTrainingLogger, NullLogger
    experiment_record.py         ExperimentRecord (the record/ directory)
    environment_report.py        describe_environment()
    reproducibility.py           set_seed(), seed_worker()
    speedster_tracker.py         SpeedTimer
    image_directory.py           scan_directory_for_images()
    classification_metrics_display.py   text reports for classification
  datasets/                      PROJECT: example datasets and augmentations
  models/                        PROJECT: example models and transformer blocks
  trainers/                      PROJECT: example trainer (ClassificationTrainer)
```

The **shared** directories are the same for all projects. The **project** directories contain examples. Change them, or add new files next to them.

`.gitignore` ignores `main_*.py`. Thus your own entry scripts, for example `main_segmentation.py`, stay local. Only `main.py` is part of the repository.

## 5. Configuration

`TrainingConfiguration` in `arcturus_lychee/configuration/basic_template.py` holds the common values:

| Field | Default | Meaning |
|---|---|---|
| `working_directory` | `"results"` | The directory for all run directories. |
| `experiment_name` | `"generic_training"` | The name of the run, without the date. |
| `prefix_date` | `True` | If True, the run name starts with the date and the time. |
| `metric_to_track` | `"top-1"` | The evaluation metric for `best.pt`. |
| `higher_is_better` | `True` | Set False for a metric like a loss. |
| `total_epochs` | `128` | The number of epochs. |
| `batch_size` | `8` | The batch for each GPU. |
| `total_workers` | `4` | Loader workers for each process. |
| `test_every_n` | `4` | Evaluate after each n epochs, and after the last epoch. |
| `save_every_n` | `8` | Save `latest.pt` after each n epochs, and after the last epoch. |
| `learning_rate` | `1e-5` | The learning rate of the optimizer. |
| `precision` | `"auto"` | `"auto"`, `"bf16"`, `"fp16"`, or `"fp32"`. `"auto"` selects bf16 if the GPU supports it, else fp32. |
| `ddp_backend` | `"nccl"` | Use `"gloo"` for a debug on the CPU. |
| `ddp_timeout_seconds` | `1800` | This time must be longer than the evaluation on rank 0. |
| `scale_lr_by_world_size` | `False` | If True, the trainer multiplies the learning rate by the number of GPUs. |
| `use_sync_batchnorm` | `False` | This is useful if the batch on each GPU is small. |
| `find_unused_parameters` | `False` | Set True only if a part of the model gets no gradient. |
| `time_with_cuda_sync` | `True` | If True, the step times are accurate. The cost is small. |
| `seed` | `42` | The base seed. |
| `deterministic` | `False` | If True, cuDNN uses deterministic algorithms. The training is slower. |

To add a project value, set a new attribute in `build_config()`:

```python
configuration.dataset_root_train = "/data/train"
configuration.model_output_class = 40
configuration.max_grad_norm      = 1.0
```

The template saves these attributes with the fields. Use only plain values: numbers, strings, booleans, lists, and dicts. TOML must hold the values, and pickle must send them to each process. If a value is not plain, `launch()` stops with an error that names the attribute.

Do not use CUDA in `build_config()`. Do not make models or tensors there. The parent process calls `build_config()` before it starts the other processes. A CUDA call there puts a CUDA context on GPU 0.

## 6. The run directory

Each run makes one run directory:

```
results/<run_name>/
  log_messages.txt              All log lines
  log/
    train.csv                   One row for each epoch: metrics, learning rate, times
    eval.csv                    One row for each evaluation
  plots/
    train.png, eval.png         One plot for each column
  weights/
    best.pt                     The best evaluation (metric_to_track)
    latest.pt                   After each save_every_n epochs, and after the last epoch
    final.pt                    After the last epoch
  record/
    configuration.toml          All configuration values, with run_name
    code.zip                    The entry script and the full arcturus_lychee package
    environment.txt             Hardware, software, CPU cores, loader workers, /dev/shm
    model.txt                   The model structure and the number of parameters
    optimizer.toml              The optimizer, the scheduler, and their hyperparameters
    augmentation_train.json     The training augmentations (albumentations format)
    augmentation_eval.json      The evaluation augmentations
    class_names.txt             The class order
    report_best.txt             The report on the test set
    summary.toml                Best value, best epoch, last metrics, total time
```

The run name is `<date>-<experiment_name>`, for example `2026_09_26_10_00-baseline`. If two runs start in the same minute, the second run gets `_2` at the end.

Each checkpoint contains `epoch`, `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`, and `configuration`. The model has no DDP wrapper. Thus a checkpoint loads with one GPU and with more GPUs.

## 7. Ablation studies

An ablation starts from a previous run and changes one value.

1. Find the run directory of the base run.
2. At the start of `build_config()`, load its configuration:

```python
def build_config() -> TrainingConfiguration:
    configuration = load_run_configuration("results/2026_09_26_10_00-baseline")
    configuration.learning_rate   = 1e-4
    configuration.experiment_name = "baseline_lr_1e-4"
    return configuration
```

3. Start the training with `python main.py`.
4. Compare the `record/summary.toml` files of the runs.

A value that changes between ablation runs must be a configuration field or attribute. `code.zip` records the hard-coded values, but a hard-coded value is not easy to change.

## 8. Continue a stopped run

The run continues in the same run directory. It starts after the epoch of `latest.pt`.

1. Load the configuration of the stopped run.
2. Set `prefix_date` to False, and set `experiment_name` to the run name. The logger then uses the same directory.
3. Load `latest.pt` and the metric history.
4. Give `start_epoch` to `fit()`.

```python
configuration = load_run_configuration("results/2026_09_26_10_00-baseline")
configuration.prefix_date     = False
configuration.experiment_name = configuration.run_name

# in worker(). All ranks load latest.pt, so the path comes from the configuration.
latest     = os.path.join(configuration.working_directory, configuration.run_name, "weights", "latest.pt")
logger     = DirectoryTrainingLogger(configuration) if is_main_process() else NullLogger()
trainer    = ClassificationTrainer(model, configuration, logger)
last_epoch = trainer.load_state(latest)
logger.load_from_csv()
trainer.fit(train_loader, eval_loader, start_epoch = last_epoch + 1)
```

## 9. Read the timing columns

`train.csv` has these columns for each epoch:

| Column | Meaning |
|---|---|
| `time_data_wait_s` | The time in `next()` of the loader. The GPU has no work in this time. |
| `time_step_s` | The time of the training steps. |
| `time_data_wait_pct` | The data wait as a percent of the total. |
| `time_train_s` | The time of the training part of the epoch. |
| `samples_per_s` | The samples of all GPUs in one second. |
| `time_eval_s` | The time of the evaluation on rank 0. The other GPUs wait in this time. |
| `time_epoch_s` | The time of the full epoch. |

With DDP, the times are the maximum of all ranks. The slowest rank sets the speed.

How to read the values:

- If `time_data_wait_pct` is more than approximately 30%, the loader is too slow. More GPUs then do not make the training faster.
- The first epoch includes the start of the loader workers. Read the data wait from the second epoch.
- The first evaluation includes the start of the loader workers of the evaluation loader.
- If `samples_per_s` does not increase with more GPUs, look at `record/environment.txt`. Compare the loader workers with the usable CPU cores.
- If `time_eval_s` is a large part of `time_epoch_s`, increase `test_every_n`.

## 10. Adapt the template to a new task

### 10.1 General procedure

1. Copy `main.py` to `main_<task>.py`.
2. Write a dataset in `arcturus_lychee/datasets/`. A sample can be a tuple, a dict, or a tensor.
3. Write or select a model in `arcturus_lychee/models/`.
4. Copy `arcturus_lychee/trainers/basic_classification.py` to a new file, for example `segmentation.py`.
5. Rename the class, for example to `SegmentationTrainer`.
6. Change `train_step()`, `eval_step()`, and `report()`. Keep the other methods.
7. In `__init__()`, change the loss, the optimizer, or the scheduler if necessary.
8. In `build_config()`, set `metric_to_track` and `higher_is_better` for your metric.
9. In `worker()`, save project data with `logger.record`, for example the augmentations.

For a small change, a subclass is also possible:

```python
class ClippedClassificationTrainer(ClassificationTrainer):
    def train_step(self, batch):
        ...
```

### 10.2 The contract of the step methods

`train_step(batch)` and `eval_step(batch)` must return two values:

1. A dict of metric names and scalar tensors, for example `{"loss": loss.detach(), "dice": dice}`.
2. The number of samples in the batch.

`train_epoch()` and `evaluate()` then calculate the weighted mean of each metric. The metric names become the columns of `train.csv` and `eval.csv`. `metric_to_track` must be one of the names that `eval_step()` returns.

A training step with the `Precision` object:

```python
def train_step(self, batch):
    inputs, targets = move_to_device(batch, self.device)

    with self.precision.autocast():
        prediction = self.model(inputs)
        loss       = self.criterion(prediction, targets)

    self.precision.backward(loss)
    self.precision.step(self.optimizer)
    self.optimizer.zero_grad(set_to_none = True)

    return {"loss": loss.detach()}, inputs.size(0)
```

Use `self.model` in `train_step()`. It has the DDP wrapper. Use `self.raw_model` in `eval_step()` and `report()`. Only rank 0 does the evaluation, and the DDP wrapper needs all ranks.

### 10.3 2D segmentation (nnU-Net style)

The dataset returns an image and a mask. albumentations changes the image and the mask together:

```python
def __getitem__(self, index):
    image = np.array(Image.open(self.images[index]).convert("RGB"))
    mask  = np.array(Image.open(self.masks[index]))              # class index for each pixel
    result = self.augmentation(image = image, mask = mask)
    return result["image"], result["mask"].long()                # [3, H, W], [H, W]
```

`nn.CrossEntropyLoss` accepts logits `[batch, classes, H, W]` and masks `[batch, H, W]`. Thus `train_step()` is the same as in section 10.2, with `"loss"` as the only metric.

`eval_step()` returns the Dice score:

```python
def dice_score(prediction : torch.Tensor, target : torch.Tensor, total_classes : int) -> torch.Tensor:
    """Mean Dice over the foreground classes. Class 0 is the background."""
    scores = []
    for c in range(1, total_classes):
        p, t        = prediction == c, target == c
        denominator = p.sum() + t.sum()
        if denominator > 0:
            scores.append(2 * (p & t).sum() / denominator)
    return torch.stack(scores).mean() if scores else torch.tensor(1.0, device = prediction.device)


@torch.no_grad()
def eval_step(self, batch):
    images, masks = move_to_device(batch, self.device)
    with self.precision.autocast():
        logits = self.raw_model(images)
    dice = dice_score(logits.argmax(dim = 1), masks, self.configuration.model_output_class)
    return {"dice": dice}, images.size(0)
```

In `build_config()`:

```python
configuration.metric_to_track  = "dice"
configuration.higher_is_better = True
```

Deep supervision: the model returns a list of outputs. Calculate the loss of each output in `train_step()`, and add the losses with weights. For a sliding-window evaluation of large images, do the windows in `report()`.

### 10.4 3D volumes

- The batch shape is `[batch, channels, depth, height, width]`. `move_to_device()` works with this shape.
- albumentations is for 2D images. Use your own transforms, or a library for 3D (for example MONAI or TorchIO), in the dataset.
- The batch on each GPU is usually small. Set `use_sync_batchnorm = True`, or use `InstanceNorm3d` or `GroupNorm` in the model.
- Set `precision = "bf16"` if the GPU supports it. The memory use then decreases.
- Do the sliding-window evaluation in `report()`. Only rank 0 does it, so it can use a large amount of memory.
- `DirectoryClassification` is for 2D images. Write a new dataset for volume files.

### 10.5 1D audio

Audio samples often have different lengths. PyTorch cannot stack them without a change. Give a `collate_fn` to `build_dataloader()`. The function pads the waveforms to the longest waveform:

```python
def pad_audio_batch(samples : list[tuple[torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    """Pad the waveforms of a batch to the longest waveform."""
    waveforms, labels = zip(*samples)
    lengths = torch.tensor([w.shape[-1] for w in waveforms])
    padded  = torch.nn.utils.rnn.pad_sequence(
        [w.transpose(0, 1) for w in waveforms], batch_first = True,
    ).transpose(1, 2)                                            # [batch, channels, time]
    return {"waveform": padded, "length": lengths, "label": torch.tensor(labels)}


train_loader = build_dataloader(train_set, configuration.batch_size, configuration.total_workers,
                                shuffle = True, distributed = world_size > 1, seed = configuration.seed,
                                collate_fn = pad_audio_batch)
```

Define the `collate_fn` at file level. Pickle must send it to the loader workers.

The batch is a dict. `move_to_device()` moves each tensor in the dict:

```python
def train_step(self, batch):
    batch = move_to_device(batch, self.device)
    with self.precision.autocast():
        logits = self.model(batch["waveform"], batch["length"])
        loss   = self.criterion(logits, batch["label"])
    ...
    return {"loss": loss.detach()}, batch["label"].size(0)
```

The shared code does not import OpenCV. An audio project does not need OpenCV or albumentations. `seed_worker()` sets the seeds of NumPy and `random` in each loader worker. Thus random crops and random gains are repeatable.

### 10.6 Diffusion models

A diffusion trainer has three changes: the noise schedule, an EMA model, and a loss metric.

In `__init__()`, after the code of the template:

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

self.total_steps    = self.configuration.diffusion_steps               # for example 1000
betas               = torch.linspace(1e-4, 0.02, self.total_steps, device = self.device)
self.alphas_cumprod = torch.cumprod(1.0 - betas, dim = 0)
self.ema            = AveragedModel(self.raw_model, multi_avg_fn = get_ema_multi_avg_fn(0.999))
self.criterion      = nn.MSELoss()
```

The names `self.alphas_cumprod` and `self.total_steps` are different from `self.scheduler`. `self.scheduler` is the learning-rate scheduler.

`train_step()` adds noise, predicts the noise, and updates the EMA model:

```python
def train_step(self, batch):
    images, _ = move_to_device(batch, self.device)
    noise     = torch.randn_like(images)
    t         = torch.randint(0, self.total_steps, (images.size(0),), device = self.device)
    alpha     = self.alphas_cumprod[t].view(-1, 1, 1, 1)
    noisy     = alpha.sqrt() * images + (1 - alpha).sqrt() * noise

    with self.precision.autocast():
        loss = self.criterion(self.model(noisy, t), noise)

    self.precision.backward(loss)
    self.precision.step(self.optimizer)
    self.optimizer.zero_grad(set_to_none = True)
    self.ema.update_parameters(self.raw_model)

    return {"loss": loss.detach()}, images.size(0)
```

- In `eval_step()`, use `self.ema.module` instead of `self.raw_model`.
- The noise and `t` are random. For a stable evaluation loss, make them with a `torch.Generator` that has a fixed seed.
- In `build_config()`, set `metric_to_track = "loss"` and `higher_is_better = False`.
- In `save_state()`, add `'ema_state': self.ema.state_dict()`. In `load_state()`, load it with `self.ema.load_state_dict()`.
- Make sample images in `report()`, and save them in `logger.samples_dir`.

### 10.7 Two optimizers (GAN)

- Make two models and wrap each one with `wrap_model()`.
- Make two optimizers in `__init__()`, for example `self.optimizer_g` and `self.optimizer_d`.
- With fp16, call `self.precision.scaler.step()` for each optimizer. Then call `self.precision.scaler.update()` one time for each step.
- Add both models and both optimizers to `save_state()` and `load_state()`.
- If a model does not get a gradient in each step, set `find_unused_parameters = True`.

### 10.8 Common additions to train_step

Gradient clipping. Unscale the gradients before the clip:

```python
self.precision.backward(loss)
self.precision.scaler.unscale_(self.optimizer)
torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configuration.max_grad_norm)
self.precision.step(self.optimizer)
self.optimizer.zero_grad(set_to_none = True)
```

Gradient accumulation. Keep a step counter in `__init__()`, for example `self.micro_step = 0`. With DDP, use `no_sync()` for the steps without an optimizer step:

```python
import contextlib

self.micro_step += 1
is_update = self.micro_step % self.configuration.accumulation_steps == 0
context   = contextlib.nullcontext() if is_update or self.world_size == 1 else self.model.no_sync()

with context:
    with self.precision.autocast():
        loss = self.criterion(self.model(inputs), targets) / self.configuration.accumulation_steps
    self.precision.backward(loss)

if is_update:
    self.precision.step(self.optimizer)
    self.optimizer.zero_grad(set_to_none = True)
```

## 11. Rules for multi-GPU code

These rules prevent code that works on one GPU and stops on more GPUs.

- **Put all values in the configuration.** Each process imports the entry script again. A change to a file-level variable does not get to the other processes.
- **Put all state on `self`.** Make it in `__init__()`.
- **Define functions at file level.** The worker function and a `collate_fn` must be compatible with pickle. A lambda or a function inside another function is not compatible.
- **Return the same metric names on all ranks.** `MetricAccumulator.reduce()` adds the values of all ranks. If one rank has a different name, the training stops, or waits until the NCCL timeout. Do not add a metric only for some batches.
- **Call collective operations on all ranks.** `train_epoch()` uses them. `evaluate()` and `report()` do not use them, because only rank 0 calls them.
- **Write files only on rank 0.** The other ranks have a `NullLogger`. Its methods do nothing.
- **Do not use CUDA in `build_config()`.** See section 5.

## 12. Singularity and SLURM

A SLURM job with 4 GPUs and one task:

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32

singularity exec --nv container.sif python main.py
```

Use one task. The template starts one process for each GPU itself. Give enough CPU cores for the loader workers: approximately `(total_workers + 1)` x the number of GPUs.

If a problem occurs, do the applicable step:

| Problem | Step |
|---|---|
| `environment.txt` shows 0 visible GPUs | Add the `--nv` flag to the container command. |
| The script stops with "An external launcher started this script" | Start the script one time, without torchrun and without `srun` with more than one task. |
| A loader worker stops with "bus error" | Decrease `total_workers`, or increase the size of `/dev/shm`. The flags `--contain` and `--containall` give a small `/dev/shm`. |
| The NCCL communication between the GPUs stops | Set `NCCL_P2P_DISABLE=1` before you start the script. |
| The NCCL communication still stops | Also set `NCCL_IB_DISABLE=1`. |
| The data wait is high, and `environment.txt` shows a NOTE about CPU cores | Decrease `total_workers`, or ask for more CPU cores. |
| The NCCL timeout stops the training during the evaluation | Increase `ddp_timeout_seconds`, or make the validation set smaller. |

## 13. Writing rules for this repository

All docstrings, comments, and log messages use ASD-STE100 Simplified Technical English. The text is then clear for readers who are not native English speakers.

- Descriptive sentences have 25 words or fewer. Instructions have 20 words or fewer.
- Instructions use the imperative. A condition comes before its instruction.
- The only modal verbs are `can`, `must`, and `will`.
- The text has no semicolons, no contractions, and no "e.g.", "i.e.", or "etc.".
- One word has one meaning. The glossary gives the terms:

| Concept | Term |
|---|---|
| The dataclass and its values | configuration |
| A process that trains | process |
| The number of a process | rank |
| A DataLoader subprocess | loader worker |
| Make sure of a state | make sure that |
| Start the script | start |
| Model quality on held-out data | evaluation |
| Saved training state | checkpoint |

The code also uses these conventions:

- Type annotations on all parameters and return values.
- Aligned `=` and `:` in blocks of related lines.
- No module-level mutable state.
