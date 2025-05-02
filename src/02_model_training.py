# src/02_model_training.py
"""
Notebook 2: Model + LightningModule + Logging

Objective: Implement model architectures (Custom CNN, ResNet), encapsulate them
           in a PyTorch Lightning LightningModule, and set up comprehensive
           logging for training and evaluation.
"""

# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim import AdamW # Example optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau # Example scheduler
# Add metrics imports later if needed (e.g., from torchmetrics)

# --- Configuration ---
# These might be loaded from a config file or CLI args in a real script
NUM_CLASSES = 7 # From our NSCLC dataset
LEARNING_RATE = 1e-4
# ... other relevant configurations ...

# ==============================================================================
# Section 1: Introduction to Histology Modeling & Custom CNN Architecture
# ==============================================================================

"""
## 🧠 Introduction to Histology Image Modeling

Histology image classification involves training models to recognize patterns
in tissue samples, often stained (e.g., H&E). Key challenges include:
- **Large Image Size:** Whole Slide Images (WSIs) are massive, requiring patch-based approaches.
- **Subtle Features:** Distinguishing between classes often relies on fine-grained details.
- **Stain Variability:** Differences in staining protocols can affect image appearance.
- **Class Imbalance:** Some tissue types might be much rarer than others.

**Common Approaches:**
- **Binary Classification:** Distinguishing between two states (e.g., tumor vs. normal).
- **Multi-Class Classification:** Assigning patches to one of several predefined categories (like our 7 region types).
- **Transfer Learning:** Leveraging models pre-trained on large datasets (like ImageNet) and fine-tuning them for the specific histology task (often beneficial).
- **Custom Architectures:** Designing CNNs tailored to the specific characteristics of histology data.

This section focuses on implementing a custom CNN as a baseline.
"""

# --- Custom CNN Architecture ---

class SimpleHistologyCNN(nn.Module):
    """
    A basic Convolutional Neural Network for histology patch classification.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Define convolutional layers
        # Example: (Input channels, Output channels, Kernel size, Stride, Padding)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dims by half

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Placeholder for calculating the flattened size dynamically
        # We'll need to pass a dummy input through the conv layers once
        self._feature_size = None
        self._calculate_feature_size() # Calculate size during init

        # Define fully connected layers
        self.fc1 = nn.Linear(self._feature_size, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_feature_size(self):
        """Helper to determine the size of the flattened features after conv layers."""
        # Create a dummy input matching expected dimensions (Batch, Channels, Height, Width)
        # Assuming input patches are, for example, 256x256
        # Adjust the dummy input size if your patches are different!
        dummy_input = torch.randn(1, 3, 256, 256) # Example size
        x = self.pool1(self.relu1(self.conv1(dummy_input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        # Flatten the output
        self._feature_size = x.view(x.size(0), -1).shape[1]
        print(f"Calculated feature size after conv layers: {self._feature_size}")


    def forward(self, x):
        # Convolutional part
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the features
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch

        # Fully connected part
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output layer (raw scores/logits)
        return x

# --- Example Instantiation (for testing) ---
if __name__ == '__main__':
    print("\n--- Testing Model Instantiation ---")
    try:
        model = SimpleHistologyCNN(num_classes=NUM_CLASSES)
        print("Model instantiated successfully.")

        # Test forward pass with a dummy input
        # Assuming input size 256x256, adjust if needed
        dummy_batch = torch.randn(4, 3, 256, 256) # Batch of 4 images
        output = model(dummy_batch)
        print(f"Output shape: {output.shape}") # Should be [batch_size, num_classes]
        assert output.shape == (4, NUM_CLASSES), "Output shape mismatch!"
        print("Forward pass successful.")

    except Exception as e:
        print(f"Error during model testing: {e}")

"""
*Next Steps:*
*- Adapt a pre-trained ResNet model (Task 4.2).
*- Implement the LightningModule to wrap these models (Task 4.2).
"""

# ==============================================================================
# Section 2: LightningModule Implementation
# ==============================================================================

"""
## ⚡ LightningModule: Encapsulating Model & Training Logic

A `pytorch_lightning.LightningModule` organizes the PyTorch code related to training,
validation, and testing. It encapsulates:
- The model architecture (our CNN or ResNet).
- The forward pass logic.
- The calculation of the loss.
- The code performed during training, validation, and test steps.
- The configuration of optimizers and learning rate schedulers.

This makes the code cleaner, more reusable, and integrates seamlessly with the
Lightning `Trainer`.
"""

class HistologyClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for histology image classification.
    """
    def __init__(self,
                 model_name: str = 'resnet18', # Or 'custom_cnn'
                 num_classes: int = NUM_CLASSES,
                 learning_rate: float = LEARNING_RATE,
                 optimizer: str = 'AdamW',
                 lr_scheduler: str = 'ReduceLROnPlateau',
                 pretrained: bool = True): # Added pretrained flag
        super().__init__()
        # Save hyperparameters for logging and potential loading later
        # Important: Don't save the actual model instance here if it's large or complex
        self.save_hyperparameters()

        # Initialize the chosen model architecture
        if model_name == 'resnet18':
            self.model = PretrainedResNetGAP(num_classes=num_classes, backbone='resnet18', pretrained=pretrained)
        elif model_name == 'resnet34':
             self.model = PretrainedResNetGAP(num_classes=num_classes, backbone='resnet34', pretrained=pretrained)
        elif model_name == 'custom_cnn':
            self.model = SimpleHistologyCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Loss function (CrossEntropyLoss is common for multi-class classification)
        self.criterion = nn.CrossEntropyLoss()

        # Example placeholder for metrics (we'll add proper ones in Task 4.3)
        # from torchmetrics.classification import MulticlassAccuracy
        # self.accuracy = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        # The forward pass is simply delegated to the underlying model
        return self.model(x)

    def _calculate_loss(self, batch, batch_idx, stage='train'):
        """Helper function to calculate and log loss for a step."""
        images, labels = batch
        logits = self.forward(images) # Get model predictions (raw scores)
        loss = self.criterion(logits, labels)

        # Log the loss for this step
        self.log(f'{stage}_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # Log the loss for accumulation over the epoch
        self.log(f'{stage}_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Placeholder for logging metrics (Task 4.3)
        # preds = torch.argmax(logits, dim=1)
        # acc = self.accuracy(preds, labels)
        # self.log(f'{stage}_acc_step', acc, on_step=True, on_epoch=False)
        # self.log(f'{stage}_acc_epoch', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss # Return loss for the optimizer

    def training_step(self, batch, batch_idx):
        # Defines the logic for a single training step
        return self._calculate_loss(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        # Defines the logic for a single validation step
        # This is used to monitor performance on unseen data during training
        return self._calculate_loss(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        # Defines the logic for a single testing step
        # This is used after training to evaluate the final model performance
        return self._calculate_loss(batch, batch_idx, stage='test')

    def configure_optimizers(self):
        """
        Choose optimizers and learning-rate schedulers to use during training.
        """
        if self.hparams.optimizer.lower() == 'adamw':
            optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer.lower() == 'adam':
             optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Add other optimizers like SGD if needed
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        # Configure learning rate scheduler if specified
        if self.hparams.lr_scheduler.lower() == 'reducelronplateau':
            # Reduce learning rate when validation loss plateaus
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss_epoch', # Metric to monitor
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        elif self.hparams.lr_scheduler.lower() == 'none':
            return optimizer # Return only the optimizer if no scheduler
        else:
             raise ValueError(f"Unsupported lr_scheduler: {self.hparams.lr_scheduler}")


# --- Update Example Instantiation (for testing) ---
if __name__ == '__main__':
    print("\n--- Testing Model Instantiation ---")
    # ... (previous tests for SimpleHistologyCNN and PretrainedResNetGAP remain) ...

    print("\n--- Testing LightningModule Instantiation ---")
    try:
        # Test with ResNet18
        lightning_model_resnet = HistologyClassifier(model_name='resnet18', num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)
        print("LightningModule (ResNet18) instantiated successfully.")

        # Test with Custom CNN
        lightning_model_custom = HistologyClassifier(model_name='custom_cnn', num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)
        print("LightningModule (Custom CNN) instantiated successfully.")

        # Test optimizer configuration
        optimizers_config = lightning_model_resnet.configure_optimizers()
        print("Optimizer configuration successful.")
        # print(optimizers_config) # Uncomment to inspect config

        # We could add a dummy forward pass test here too if needed
        # dummy_batch = torch.randn(4, 3, 256, 256)
        # output = lightning_model_resnet(dummy_batch)
        # print(f"LightningModule Output shape: {output.shape}")

    except Exception as e:
        print(f"Error during LightningModule testing: {e}")


"""
*Next Steps:*
*- Implement comprehensive logging (metrics like accuracy, precision, recall) (Task 4.3).
*- Add visualization for misclassified samples (Task 4.4).
*- Integrate with external loggers like W&B/TensorBoard (Task 4.5).
"""



# Add more sections below for subsequent tasks... 