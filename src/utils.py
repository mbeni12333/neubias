
# --- Common Imports ---
import os
# Disable CUDA devices (force CPU only)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Important for Albumentations with PyTorch
from PIL import Image # For loading images
import cv2 # Often used by Albumentations backend
from sklearn.model_selection import train_test_split
from PIL import Image
# --- Modified PatchDataset Class ---
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import math

import seaborn as sns

# --- Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim import AdamW # Example optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau # Example scheduler
# Add metrics imports later if needed (e.g., from torchmetrics)
# --- Imports ---
# ... (existing imports) ...
import torchmetrics # Make sure torchmetrics is installed (pip install torchmetrics)

# Specific metrics (adjust average strategy as needed)
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, ConfusionMatrix

# --- Configuration ---
# These might be loaded from a config file or CLI args in a real script
NUM_CLASSES = 7 # From our NSCLC dataset
LEARNING_RATE = 1e-4
# ... other relevant configurations ...

Image.MAX_IMAGE_PIXELS = None  # disables the decompression bomb protection
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


# --- Visualization (using Plotly) ---
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import torchvision.utils
# --- LightningDataModule Implementation ---
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob # Ensure glob is imported if not already in this cell context
from pathlib import Path # Ensure Path is imported


# --- LightningDataModule Implementation ---
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob # Ensure glob is imported if not already in this cell context
from pathlib import Path # Ensure Path is imported

# Set default Plotly template
pio.templates.default = "plotly_white"

# --- Reproducibility ---
# Set random seeds for reproducibility
# pl.seed_everything(42, workers=True)

# --- Configuration ---
# Define the path to your raw dataset
# Ensure your dataset (e.g., NSCLC IHC images) is in this directory
# with subfolders for each class ('TER', 'Necrotic', etc.)


def denormalize(tensor, mean=IMG_MEAN, std=IMG_STD):
    """Denormalizes a tensor image with mean and standard deviation."""
    # Clone to avoid modifying the original tensor
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization: (tensor * std) + mean
    # We need to clamp values to [0, 1] after denormalization
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# --- Visualize a batch from the DataModule's DataLoader ---

# Ensure the show_batch_grid function is defined earlier
# Make sure it uses the potentially updated idx_to_class_from_module map

def show_batch_grid(dataloader, num_images=16, title="Sample Batch", idx_map=None): # Added idx_map
    """Fetches one batch and displays it using torchvision.utils.make_grid and matplotlib."""
    if not dataloader:
        print("DataLoader is None, cannot show batch.")
        return
    if not idx_map:
        print("Index-to-class map not provided.")
        idx_map = {} # Default to empty map

    try:
        images, labels = next(iter(dataloader))
    except StopIteration:
        print("DataLoader is empty or exhausted.")
        return
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return

    images_to_show = images[:num_images]
    labels_to_show = labels[:num_images]
    denormalized_images = [denormalize(img) for img in images_to_show]
    grid = torchvision.utils.make_grid(denormalized_images, nrow=int(math.sqrt(num_images)))
    grid_np = grid.numpy()
    grid_display = np.transpose(grid_np, (1, 2, 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_display)
    plt.title(title)
    plt.axis('off')
    plt.show()

    print("Labels for displayed batch:")
    print([idx_map.get(l.item(), "Unknown") for l in labels_to_show])



def get_dataset_info(data_dir='../data/Regions/'):
    data_dir = Path(data_dir)
    try:
        class_names = sorted([p.name for p in data_dir.glob('*') if p.is_dir()])
        if not class_names:
            raise FileNotFoundError
        num_classes = len(class_names)

        print(f'Dataset directory: {data_dir}')
        print(f'Found {num_classes} classes: {class_names}')

        class_to_idx = {name: i for i, name in enumerate(class_names)}
        idx_to_class = {i: name for name, i in class_to_idx.items()}

        print(f'Class to index mapping: {class_to_idx}')

    except FileNotFoundError:
        print(f'Error: Dataset directory not found or no class subfolders found at {data_dir}')
        print('Please ensure your dataset is placed correctly as per the README instructions.')
        class_names = []
        num_classes = 0
        class_to_idx = {}
        idx_to_class = {}

    return class_names, num_classes, class_to_idx, idx_to_class

# Example usage:
# CLASS_NAMES, NUM_CLASSES, class_to_idx, idx_to_class = get_dataset_info()


# --- Define Transforms ---

# Define image size if resizing is needed
# IMG_HEIGHT = 224
# IMG_WIDTH = 224

def get_transforms(img_mean=None, img_std=None, min_size=256, crop_size=256):
    """
    Returns train and validation/test transforms using Albumentations.
    Optionally override mean/std, min_size, and crop_size.
    """
    # Default normalization constants (ImageNet stats)
    IMG_MEAN = img_mean if img_mean is not None else [0.485, 0.456, 0.406]
    IMG_STD = img_std if img_std is not None else [0.229, 0.224, 0.225]

    train_transforms = A.Compose([
        A.PadIfNeeded(min_height=min_size, min_width=min_size, p=1.0, border_mode=cv2.BORDER_REFLECT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.1), p=0.2),
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

    val_test_transforms = A.Compose([
        A.PadIfNeeded(min_height=min_size, min_width=min_size, p=1.0, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])

    print("Defined train_transforms and val_test_transforms.")
    return train_transforms, val_test_transforms

# Example usage:
# train_transforms, val_test_transforms = get_transforms()

# Optional: Visualize an image after applying train_transforms
def visualize_transform(sample_image_np, train_transforms):
    if sample_image_np is not None:
        transformed_sample = train_transforms(image=sample_image_np)['image']
        print("Sample image shape after train_transforms (should be CHW tensor):", transformed_sample.shape)
        # Visualization of the normalized tensor might look strange directly with imshow
    else:
        print("Cannot apply train_transforms as sample_image_np is None.")






class PatchDataset(Dataset):
    """
    PyTorch Dataset for patch classification.
    Accepts a list of filepaths and corresponding labels.
    """
    def __init__(self, filepaths, labels, transform=None, label_map=None):
        """
        Args:
            filepaths (list): List of paths to image files.
            labels (list): List of corresponding labels (strings).
            transform (callable, optional): Optional transform to be applied on a sample.
            label_map (dict, optional): Dictionary mapping string labels to integer indices.
        """
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

        # Create or use label map
        if label_map:
            self.label_map = label_map
        else:
            # Create map from unique labels found
            unique_labels = sorted(list(set(labels)))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        self.idx_to_label = {v: k for k, v in self.label_map.items()} # For potential reverse lookup

        print(f"Initialized Dataset. Label map: {self.label_map}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label_str = self.labels[idx]
        label_idx = self.label_map[label_str]

        # Load image using PIL (ensure RGB)
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if they exist
        if self.transform:
            # Albumentations requires numpy array
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image'] # Albumentations returns a dict
        else:
            # Basic transform to tensor if no augmentation
            # Note: Albumentations ToTensorV2 normalizes by default,
            # this basic conversion does not. Add normalization if needed.
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image_tensor, label_idx

    def get_label_map(self):
        return self.label_map

    def get_idx_to_label(self):
        return self.idx_to_label
    





class PatchDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 class_to_idx: dict,
                 train_transform: A.Compose,
                 val_test_transform: A.Compose,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Placeholders for datasets and file paths after setup
        self.train_paths, self.val_paths, self.test_paths = None, None, None
        self.train_labels, self.val_labels, self.test_labels = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.label_map = None # Will be determined by train_dataset

        # Save hyperparameters for logging (optional but good practice)
        self.save_hyperparameters(ignore=['train_transform', 'val_test_transform', 'class_to_idx']) # Avoid saving complex objects

    def prepare_data(self):
        # Called only on 1 GPU/TPU in distributed settings.
        # Use this to download data, check existence etc.
        if not self.data_dir.exists():
             raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        print(f"Data directory check passed: {self.data_dir}")
        # We could add more checks here if needed

    def setup(self, stage: str = None):
        # Called on every GPU/TPU in distributed settings.
        # Assign train/val/test datasets for use in dataloaders
        # `stage` can be 'fit', 'validate', 'test', 'predict'
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            print(f"Setting up data for stage: {stage}")

            # --- Scan & Split ---
            all_image_paths = []
            all_labels_str = [] # Use string labels for splitting/map creation
            class_names = sorted(list(self.class_to_idx.keys()))

            print("Scanning for all image paths and string labels...")
            if self.data_dir.exists() and class_names:
                 for class_name in class_names:
                     class_dir = self.data_dir / class_name
                     if class_dir.is_dir():
                         for img_path in glob.glob(str(class_dir / '*.png')): # Adjust pattern if needed
                            all_image_paths.append(Path(img_path))
                            all_labels_str.append(class_name) # Store string label
                 print(f"Found {len(all_image_paths)} total images.")
            else:
                 print("Error: Cannot perform split. Data directory or class folders not found/empty.")
                 # Handle error or return if necessary
                 return

            if not all_image_paths:
                print("No images found, cannot proceed with setup.")
                return

            # --- Perform Splits ---
            try:
                # Split 1: Train vs Temp (Val+Test)
                self.train_paths, temp_paths, self.train_labels, temp_labels = train_test_split(
                    all_image_paths, all_labels_str, # Use string labels
                    test_size=(self.val_ratio + self.test_ratio),
                    random_state=self.seed,
                    stratify=all_labels_str
                )
                # Split 2: Val vs Test from Temp
                relative_test_size = self.test_ratio / (self.val_ratio + self.test_ratio)
                self.val_paths, self.test_paths, self.val_labels, self.test_labels = train_test_split(
                    temp_paths, temp_labels, # Use string labels
                    test_size=relative_test_size,
                    random_state=self.seed,
                    stratify=temp_labels
                )
                print("Dataset split completed.")
                print(f"Train size: {len(self.train_paths)}, Val size: {len(self.val_paths)}, Test size: {len(self.test_paths)}")

            except ValueError as e:
                print(f"Error during stratified split: {e}. Check class distribution and split ratios.")
                # Implement fallback or raise error
                return

            # --- Instantiate Datasets ---
            if self.train_paths:
                self.train_dataset = PatchDataset(
                    filepaths=self.train_paths,
                    labels=self.train_labels,
                    transform=self.train_transform
                    # Let PatchDataset create the label_map from these string labels
                )
                self.label_map = self.train_dataset.get_label_map() # Get the map

            if self.val_paths and self.label_map is not None:
                self.val_dataset = PatchDataset(
                    filepaths=self.val_paths,
                    labels=self.val_labels,
                    transform=self.val_test_transform,
                    label_map=self.label_map # Use map from train set
                )

            if self.test_paths and self.label_map is not None:
                self.test_dataset = PatchDataset(
                    filepaths=self.test_paths,
                    labels=self.test_labels,
                    transform=self.val_test_transform,
                    label_map=self.label_map # Use map from train set
                )

            print("Datasets instantiated.")

    def train_dataloader(self):
        if self.train_dataset:
            return DataLoader(self.train_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              pin_memory=True,
                              persistent_workers=True if self.num_workers > 0 else False) # Good practice
        return None

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(self.val_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers,
                              pin_memory=True,
                              persistent_workers=True if self.num_workers > 0 else False)
        return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers,
                              pin_memory=True,
                              persistent_workers=True if self.num_workers > 0 else False)
        return None

    # Helper to get label map easily
    def get_label_map(self):
        if self.label_map is None and self.train_dataset:
             self.label_map = self.train_dataset.get_label_map()
        return self.label_map

    def get_idx_to_label(self):
         label_map = self.get_label_map()
         if label_map:
             return {v: k for k, v in label_map.items()}
         return {}



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

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling

        # Placeholder for calculating the flattened size dynamically
        # We'll need to pass a dummy input through the conv layers once
        self._feature_size = None
        self._calculate_feature_size() # Calculate size during init

        # # Define fully connected layers
        # self.fc1 = nn.Linear(self._feature_size, 128)
        # self.relu4 = nn.ReLU()
        # self.dropout = nn.Dropout(0.5) # Dropout for regularization
        # self.fc2 = nn.Linear(128, num_classes)

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
        # Global Average Pooling
        x = self.gap(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        return x

class HistologyClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for histology image classification.
    """
    def __init__(self,
                 encoder_name: str = 'resnet18', # Or 'custom_cnn'
                 num_classes: int = NUM_CLASSES,
                 learning_rate: float = LEARNING_RATE,
                 optimizer: str = 'AdamW',
                 lr_scheduler: str = 'ReduceLROnPlateau',
                 pretrained: bool = True,
                 classWeights=None): # Added pretrained flag
        super().__init__()
        # Save hyperparameters for logging and potential loading later
        # Important: Don't save the actual model instance here if it's large or complex
        self.save_hyperparameters()

        if classWeights is None:
            classWeights = torch.ones(num_classes)
        else:
            classWeights = torch.tensor(classWeights, dtype=torch.float32)
            # Ensure class weights are a tensor
            if not isinstance(classWeights, torch.Tensor):
                raise ValueError("classWeights must be a tensor or None.")
            if len(classWeights) != num_classes:
                raise ValueError(f"Length of classWeights must match num_classes ({num_classes}).")

        if encoder_name == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained) # Load pretrained ResNet18
            # remove the fully connected, and give the number of channels of the output layer
            num_ftrs = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity() # Remove the final fully connected layer
        elif encoder_name == 'custom_cnn':
            self.encoder = SimpleHistologyCNN(num_classes=num_classes)
            # For custom CNN, we need to define the final fully connected layer
            num_ftrs = self.encoder._feature_size # Get the calculated feature size
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")
            # Use a sequential model for the classifier head
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        

        # Loss function (CrossEntropyLoss is common for multi-class classification)
        self.criterion = nn.CrossEntropyLoss(weight=classWeights) # Use class weights if provided

        # Example placeholder for metrics (we'll add proper ones in Task 4.3)
        # from torchmetrics.classification import MulticlassAccuracy
        # self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        metric_args = {'num_classes': num_classes, 'average': 'macro'}

        self.train_metrics = torchmetrics.MetricCollection({
            'acc': MulticlassAccuracy(**metric_args),
            'precision': MulticlassPrecision(**metric_args),
            'recall': MulticlassRecall(**metric_args),
            'f1': MulticlassF1Score(**metric_args),
            # AUC requires probabilities or logits
            # 'auroc': MulticlassAUROC(**metric_args, thresholds=None) # Use default thresholds
        })
        self.val_metrics = self.train_metrics.clone(prefix='val_') # Clone with prefix
        self.test_metrics = self.train_metrics.clone(prefix='test_') # Clone with prefix

        # Separate AUROC metric as it requires logits/probabilities directly
        auroc_args = {'num_classes': num_classes, 'average': 'macro', 'thresholds': None}
        self.train_auroc = MulticlassAUROC(**auroc_args)
        self.val_auroc = MulticlassAUROC(**auroc_args)
        self.test_auroc = MulticlassAUROC(**auroc_args)


        self.val_confusion_matrix = ConfusionMatrix(num_classes=num_classes, task='multiclass')
        self.test_confusion_matrix = ConfusionMatrix(num_classes=num_classes, task='multiclass')

    def forward(self, x):
        # The forward pass is simply delegated to the underlying model
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    def common_step(self, batch, batch_idx, stage="train"):
        """
        Common step for training, validation, and testing.
        """
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)

        # Log the loss
        self.log(f'{stage}_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log(f'{stage}_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        getattr(self, f'{stage}_metrics').update(preds, labels)
        getattr(self, f'{stage}_auroc').update(probs, labels)
        if stage == "val" or stage == "test":
            # Update confusion matrix for validation
            # self.val_confusion_matrix.update(preds, labels)  # Uncommented this line
            getattr(self, f'{stage}_confusion_matrix').update(preds, labels)

        # Log metrics
        self.log_dict(getattr(self, f'{stage}_metrics'), on_step=False, on_epoch=True, logger=True)
        self.log(f'{stage}_auroc', getattr(self, f'{stage}_auroc'), on_step=False, on_epoch=True, logger=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="test")
    

    def on_validation_epoch_end(self):
        # Log confusion matrix for validation
        val_confusion = self.val_confusion_matrix.compute()

        fig, ax = plt.subplots(figsize=(10, 10))
        # use seaborn to plot the confusion matrix
        sns.heatmap(val_confusion.cpu().numpy(), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Validation Confusion Matrix')
        # plt.show()
        self.logger.experiment.add_figure('val_confusion_matrix', fig, self.current_epoch)
        # self.log('val_confusion_matrix', val_confusion, on_step=False, on_epoch=True, logger=True)
        # close the figure to avoid display issues
        plt.close(fig)
        self.val_confusion_matrix.reset()

        
    def on_test_epoch_end(self):
        # Log confusion matrix for test
        test_confusion = self.test_confusion_matrix.compute()
        fig, ax = plt.subplots(figsize=(10, 10))
        # use seaborn to plot the confusion matrix
        sns.heatmap(test_confusion.cpu().numpy(), annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Test Confusion Matrix')
  
        self.logger.experiment.add_figure('test_confusion_matrix', fig, self.current_epoch)
        # self.log('test_confusion_matrix', test_confusion, on_step=False, on_epoch=True, logger=True)
        # close the figure to avoid display issues
        plt.close(fig)
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        """
        Choose optimizers and learning-rate schedulers to use during training.
        """
        # If pretrained is True, freeze encoder parameters (do not train encoder)
        if self.hparams.pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Collect parameters to optimize (only trainable ones)
        params_to_optimize = (
            list(filter(lambda p: p.requires_grad, self.encoder.parameters())) +
            list(self.fc.parameters())
        )

        if self.hparams.optimizer.lower() == 'adamw':
            optimizer = AdamW(
            params_to_optimize,
            lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.hparams.learning_rate
            )
        # Add other optimizers like SGD if needed
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        # Configure learning rate scheduler if specified
        if self.hparams.lr_scheduler.lower() == 'reducelronplateau':
            # Reduce learning rate when validation loss plateaus
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
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

