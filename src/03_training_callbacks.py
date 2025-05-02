"""
Notebook 3: Callbacks, Hooks & Training

Objective: Learn about PyTorch Lightning Callbacks, implement built-in and
           custom callbacks, set up and run a full training loop using the
           Trainer, and export the model using TorchScript.
"""

# --- Imports ---
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import os

# Assume DataModule and LightningModule can be imported from previous scripts
# You might need to adjust sys.path or package structure for this to work
# Example (adjust paths as needed):
# from pathlib import Path
# import sys
# script_dir = Path(__file__).parent
# sys.path.append(str(script_dir.parent)) # Add project root to path
# from src.01_Dataset_Creation_Augmentation import PatchDataModule, train_transforms, val_test_transforms, class_to_idx # Need to refactor these out or pass config
# from src.02_model_training import HistologyClassifier

# --- Configuration (Placeholders - Load these properly) ---
# These would typically come from config files, CLI args, or imported modules
CHECKPOINT_DIR = "./lightning_checkpoints"
LOG_DIR = "./lightning_logs"
NUM_CLASSES = 7
LEARNING_RATE = 1e-4

# Example: Assume these are loaded or defined
# class_to_idx = {'TER': 0, 'Necrotic': 1, ...} # Load from previous step or config
# train_transforms = ... # Load from previous step or config
# val_test_transforms = ... # Load from previous step or config
# DATA_DIR = '../data/raw/patch_classification_dataset'

# ==============================================================================
# Section 1: Introduction to Callbacks & Built-in Examples
# ==============================================================================

"""
## 📞 Introduction to Callbacks

Callbacks are self-contained programs that can be added to your PyTorch Lightning
`Trainer`. They allow you to add custom logic at various stages of the training
process (e.g., at the beginning/end of an epoch, before/after a batch) without
cluttering your `LightningModule`.

**Why use Callbacks?**
- **Modularity:** Keep training logic separate from model definition.
- **Reusability:** Easily reuse common logic like checkpointing or early stopping across projects.
- **Extensibility:** Hook into specific points in the training loop for monitoring, logging, or other actions.

Lightning provides several useful built-in callbacks, and you can easily create
your own.
"""

# --- Built-in Callback: ModelCheckpoint ---
"""
### `ModelCheckpoint`

This callback saves your model's weights periodically during training.
Key parameters:
- `dirpath`: Directory to save checkpoints.
- `filename`: Naming pattern for checkpoint files (can include metrics).
- `monitor`: Metric to monitor for saving the 'best' model (e.g., 'val_loss_epoch').
- `mode`: 'min' or 'max' depending on whether the monitored metric should be minimized or maximized.
- `save_top_k`: Save the top 'k' best models according to the monitored metric.
- `save_last`: Save the latest model checkpoint at the end of every epoch.
"""

# Example configuration for ModelCheckpoint
# Saves the best model based on validation accuracy (higher is better)
checkpoint_callback_acc = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename='best-model-acc-{epoch:02d}-{val_acc:.2f}',
    monitor='val_acc', # Assuming 'val_acc' is logged in LightningModule
    mode='max',
    save_top_k=1, # Save only the single best model
    save_last=True, # Also save the latest model state
    verbose=True
)

# Example configuration for ModelCheckpoint
# Saves the best model based on validation loss (lower is better)
checkpoint_callback_loss = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename='best-model-loss-{epoch:02d}-{val_loss_epoch:.2f}',
    monitor='val_loss_epoch', # Monitor the epoch validation loss
    mode='min',
    save_top_k=1,
    save_last=True,
    verbose=True
)

print("ModelCheckpoint callbacks configured.")

# --- Built-in Callback: EarlyStopping ---
"""
### `EarlyStopping`

This callback stops training early if a monitored metric stops improving,
preventing overfitting and saving computation time.
Key parameters:
- `monitor`: Metric to monitor (e.g., 'val_loss_epoch').
- `mode`: 'min' or 'max'.
- `patience`: Number of epochs to wait for improvement before stopping.
- `min_delta`: Minimum change in the monitored quantity to qualify as an improvement.
- `verbose`: Print messages when stopping.
"""

# Example configuration for EarlyStopping
# Stops training if validation loss doesn't improve for 5 consecutive epochs
early_stopping_callback = EarlyStopping(
    monitor='val_loss_epoch',
    mode='min',
    patience=10, # Increase patience for potentially noisy validation loss
    min_delta=0.001,
    verbose=True
)

print("EarlyStopping callback configured.")

# --- Built-in Callback: LearningRateMonitor ---
"""
### `LearningRateMonitor`

Automatically logs the learning rate used by the optimizer(s) at each step or epoch.
Very useful when using learning rate schedulers.
Key parameters:
- `logging_interval`: 'step' or 'epoch'.
"""

lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

print("LearningRateMonitor callback configured.")


# --- Using Callbacks with the Trainer (Placeholder) ---
"""
To use these callbacks, you pass them as a list to the `callbacks` argument
when initializing the `pytorch_lightning.Trainer`:

```python
# Assuming 'model' is an instance of HistologyClassifier
# Assuming 'data_module' is an instance of PatchDataModule

# Create a list of callbacks to use
my_callbacks = [
    checkpoint_callback_loss, # Save based on loss
    # checkpoint_callback_acc, # Or save based on accuracy
    early_stopping_callback,
    lr_monitor_callback
    # Add custom callbacks here later (Task 5.2)
]

# Configure the Trainer
trainer = pl.Trainer(
    max_epochs=50, # Example
    callbacks=my_callbacks,
    logger=pl.loggers.TensorBoardLogger(save_dir=LOG_DIR, name="histology_classification"), # Example logger
    # Add other trainer arguments (accelerator, devices, etc.) as needed
    # accelerator='gpu', devices=1,
)

# Start training (Example call)
# trainer.fit(model, datamodule=data_module)

print("\nPlaceholder showing how callbacks are passed to the Trainer.")
```
"""

# --- Example: Dummy Model and DataModule for testing structure ---
# In a real scenario, you would import these from other files
class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self): return None
    def val_dataloader(self): return None

class DummyModel(pl.LightningModule):
    def training_step(self, batch, batch_idx): return None
    def validation_step(self, batch, batch_idx): return None
    def configure_optimizers(self): return None

if __name__ == '__main__':
    print("\n--- Testing Trainer with Callbacks (Structure Check) ---")
    try:
        dummy_model = DummyModel()
        dummy_data = DummyDataModule()

        trainer_test = pl.Trainer(
            callbacks=[
                checkpoint_callback_loss,
                early_stopping_callback,
                lr_monitor_callback
            ],
            max_epochs=1, # Just to test instantiation
            logger=False, # Disable logging for this quick test
            enable_checkpointing=False, # Disable actual checkpointing for this test
            enable_progress_bar=False
        )
        print("Trainer instantiated successfully with callbacks.")
        # We won't call .fit() here as it requires real data

    except Exception as e:
        print(f"Error during Trainer testing: {e}")

"""
*Next Steps:*
*- Implement a custom callback for logging images (Task 5.2).
*- Set up the complete Trainer and run the training loop (Task 5.3).
""" 

# ==============================================================================
# Section 2: Custom Callback for Image Logging
# ==============================================================================
from pytorch_lightning.callbacks import Callback
import torchvision # For make_grid
import math

# Need denormalize function - might be better to put this in a utils file
# Or redefine it here if needed, assuming IMG_MEAN/IMG_STD are accessible
# Example denormalize function (ensure mean/std are correct)
IMG_MEAN = [0.485, 0.456, 0.406] # Placeholder
IMG_STD = [0.229, 0.224, 0.225]  # Placeholder
def denormalize(tensor, mean=IMG_MEAN, std=IMG_STD):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


"""
## 🖼️ Custom Callback: Image Prediction Logger

While built-in callbacks are useful, sometimes we need custom logic. Here, we
create a callback to log sample image predictions during validation. This helps
visually inspect how the model is performing on actual data over time.

This callback will:
- Run at the end of each validation epoch (`on_validation_epoch_end`).
- Get a batch of data from the validation dataloader.
- Perform inference using the current model (`pl_module`).
- Create a grid of images showing the input, true label, and predicted label.
- Log this grid to the Lightning logger (e.g., TensorBoard).
"""

class ImagePredictionLogger(Callback):
    """
    Logs a grid of validation images, true labels, and predictions to the logger.
    """
    def __init__(self, num_samples: int = 8, log_every_n_epochs: int = 5):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        print(f"ImagePredictionLogger configured to log {num_samples} samples every {log_every_n_epochs} epochs.")

    def _log_image_predictions(self, trainer, pl_module):
        """Helper function to perform the logging."""
        if not trainer.datamodule:
            print("ImagePredictionLogger: Trainer has no datamodule, cannot log images.")
            return

        val_dataloader = trainer.datamodule.val_dataloader()
        if not val_dataloader:
             print("ImagePredictionLogger: No validation dataloader found in datamodule.")
             return

        # Get idx_to_class mapping from datamodule or model hparams if available
        idx_to_class = {}
        if hasattr(trainer.datamodule, 'get_idx_to_label'):
            idx_to_class = trainer.datamodule.get_idx_to_label()
        elif hasattr(pl_module, 'hparams') and 'idx_to_class' in pl_module.hparams:
             idx_to_class = pl_module.hparams.idx_to_class # Assuming it was saved
        else:
            # Fallback: Create a simple map if NUM_CLASSES is known
             if hasattr(pl_module, 'hparams') and 'num_classes' in pl_module.hparams:
                 idx_to_class = {i: f'Class_{i}' for i in range(pl_module.hparams.num_classes)}
             else:
                 print("Warning: Cannot determine class names for image logging.")


        # Get a batch from the validation dataloader
        try:
            batch = next(iter(val_dataloader))
            images, labels = batch
        except StopIteration:
            print("ImagePredictionLogger: Validation dataloader is empty.")
            return
        except Exception as e:
             print(f"ImagePredictionLogger: Error fetching validation batch: {e}")
             return

        # Move images to the same device as the model
        images = images.to(pl_module.device)
        labels = labels.to(pl_module.device)

        # Get model predictions
        pl_module.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits = pl_module(images)
        preds = torch.argmax(logits, dim=1)
        pl_module.train() # Set model back to training mode

        # Select samples to log
        images_to_log = images[:self.num_samples]
        labels_to_log = labels[:self.num_samples]
        preds_to_log = preds[:self.num_samples]

        # Denormalize images for visualization
        denormalized_images = [denormalize(img) for img in images_to_log]

        # Create the grid
        try:
            grid = torchvision.utils.make_grid(denormalized_images, nrow=int(math.sqrt(self.num_samples)))
        except Exception as e:
            print(f"ImagePredictionLogger: Error creating image grid: {e}")
            return

        # Generate captions (TrueLabel / PredLabel)
        captions = []
        for i in range(self.num_samples):
             true_label = idx_to_class.get(labels_to_log[i].item(), '?')
             pred_label = idx_to_class.get(preds_to_log[i].item(), '?')
             captions.append(f"T:{true_label}/P:{pred_label}")

        # Log the image grid using the trainer's logger
        # Note: TensorBoardLogger expects CHW format
        if trainer.logger:
            trainer.logger.experiment.add_image(
                f"Validation Predictions (Epoch {trainer.current_epoch})",
                grid,
                global_step=trainer.current_epoch # Log per epoch
                # dataformats='CHW' # Usually default for add_image
            )
            # Add text captions (might need specific logger handling)
            try:
                 # Basic text logging - format might vary depending on logger
                 caption_text = "\n".join([f"Img {i+1}: {cap}" for i, cap in enumerate(captions)])
                 trainer.logger.experiment.add_text(
                     f"Validation Predictions Captions (Epoch {trainer.current_epoch})",
                     caption_text,
                     global_step=trainer.current_epoch
                 )
            except Exception as text_log_e:
                 print(f"ImagePredictionLogger: Could not log text captions: {text_log_e}")

        else:
            print("ImagePredictionLogger: Trainer has no logger configured.")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of the validation epoch."""
        # Log only every n epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            print(f"\nLogging image predictions for epoch {trainer.current_epoch + 1}...")
            self._log_image_predictions(trainer, pl_module)


# --- Update Placeholder for Using Callbacks ---
"""
To use the custom callback, simply add an instance of it to the list:

```python
# Assuming previous callbacks (checkpoint_callback_loss, etc.) are defined

image_logger_callback = ImagePredictionLogger(num_samples=16, log_every_n_epochs=5)

my_callbacks = [
    checkpoint_callback_loss,
    early_stopping_callback,
    lr_monitor_callback,
    image_logger_callback # Add the custom callback here
]

# Configure the Trainer (as before)
trainer = pl.Trainer(
    # ... other args ...
    callbacks=my_callbacks,
    logger=pl.loggers.TensorBoardLogger(save_dir=LOG_DIR, name="histology_classification"),
    # ...
)

# trainer.fit(model, datamodule=data_module)
```
"""