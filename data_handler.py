import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2  # Import v2 for CutMix
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import random
from datasets import load_dataset
os.system("pip install -q kagglehub")
import kagglehub

class CIFAR100CDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.images[index]
        # Images are already in (H, W, C) format, no need to transpose
        img = Image.fromarray(img.astype(np.uint8))
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[index]
    
    def __len__(self):
        return len(self.images)
    
class DataHandler:
    """
    Data handling class for loading and preprocessing datasets for MixMo experiments.
    
    Handles:
    - CIFAR-10/100 loading
    - TinyImageNet loading
    - CIFAR-100-C loading (corrupted test data)
    - Augmentation pipeline (standard, CutMix, AugMix)
    - Batch repetition (for b=2, 4 experiments)
    """
    
    # CIFAR-10 stats
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    
    # CIFAR-100 stats
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    
    # TinyImageNet stats
    TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
    TINY_IMAGENET_STD = (0.2770, 0.2691, 0.2821)
    
    def __init__(self, data_root='./data'):
        """Initialize the data handler."""
        self.data_root = data_root
        if not os.path.exists(data_root):
            os.makedirs(data_root)
    
    def get_cifar10(self, batch_size=128, batch_repetitions=1):
        """
        Load CIFAR-10 dataset with standard augmentations.
        
        Args:
            batch_size: Batch size for dataloaders
            batch_repetitions: Number of times to repeat each sample in a batch (b in the paper)
        
        Returns:
            train_loader, test_loader
        """
        train_transform = self._get_train_transform(
            mean=self.CIFAR10_MEAN, 
            std=self.CIFAR10_STD, 
            size=32
        )
        
        test_transform = self._get_test_transform(
            mean=self.CIFAR10_MEAN,
            std=self.CIFAR10_STD
        )
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_transform)
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=test_transform)
        
        return self._create_data_loaders(train_dataset, test_dataset, batch_size, batch_repetitions)
    
    def get_cifar100(self, batch_size=128, batch_repetitions=1):
        """
        Load CIFAR-100 dataset with standard augmentations.
        
        Args:
            batch_size: Batch size for dataloaders
            batch_repetitions: Number of times to repeat each sample in a batch (b in the paper)
        
        Returns:
            train_loader, test_loader
        """
        train_transform = self._get_train_transform(
            mean=self.CIFAR100_MEAN, 
            std=self.CIFAR100_STD, 
            size=32
        )
        
        test_transform = self._get_test_transform(
            mean=self.CIFAR100_MEAN,
            std=self.CIFAR100_STD
        )
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root, train=True, download=True, transform=train_transform)
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root, train=False, download=True, transform=test_transform)
        
        return self._create_data_loaders(train_dataset, test_dataset, batch_size, batch_repetitions)
    
    def get_cifar100c(self, corruption_type, batch_size=128, data_path=None, severity=3):
        """
        Load CIFAR-100-C test dataset with specific corruption type and severity level.
        
        Args:
            corruption_type: str, name of corruption (e.g., 'gaussian_noise', 'frost', etc.)
            batch_size: int, batch size for the dataloader
            data_path: str, path to CIFAR-100-C directory, if None will download from Kaggle
            severity: int, severity level (1-5), default is 3
        
        Returns:
            test_loader for corrupted test data
        """
        # Download from Kaggle if path not provided
        if data_path is None:
            data_path = kagglehub.dataset_download("ahmedyradwan02/cifar100-c")
            data_path = os.path.join(data_path, "CIFAR-100-C")
            print(f"Downloaded CIFAR-100-C to {data_path}")
        
        # Check if corruption type is valid (by checking if file exists)
        corruption_file = os.path.join(data_path, f"{corruption_type}.npy")
        if not os.path.exists(corruption_file):
            raise ValueError(f"Corruption file {corruption_file} does not exist.")
        
        # Load corrupted images
        corrupted_data = np.load(corruption_file)
        
        # Calculate indices for the specified severity level
        # Each severity level has 10,000 images
        # Severity 1: 0-9999, Severity 2: 10000-19999, ..., Severity 5: 40000-49999
        start_idx = (severity - 1) * 10000
        end_idx = start_idx + 10000
        
        # Extract images for the specified severity level
        corrupted_images = corrupted_data[start_idx:end_idx]
        
        # Load labels (same for all corruptions)
        labels_file = os.path.join(data_path, "labels.npy")
        if os.path.exists(labels_file):
            # Labels are the same for all severity levels, so we take the first 10000
            labels = np.load(labels_file)[:10000]
        else:
            raise ValueError(f"Labels file not found at {labels_file}")
        
        # Create dataset from numpy arrays
        test_transform = self._get_test_transform(
            mean=self.CIFAR100_MEAN,
            std=self.CIFAR100_STD
        )
        
        # Create the dataset and dataloader
        test_dataset = CIFAR100CDataset(corrupted_images, labels, transform=test_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        return test_loader
    
    def get_tiny_imagenet(self, batch_size=128, batch_repetitions=1):
        """
        Load TinyImageNet dataset with standard augmentations.
        
        Args:
            batch_size: Batch size for dataloaders
            batch_repetitions: Number of times to repeat each sample in a batch (b in the paper)
        
        Returns:
            train_loader, val_loader
        """
        train_transform = self._get_train_transform(
            mean=self.TINY_IMAGENET_MEAN, 
            std=self.TINY_IMAGENET_STD, 
            size=64
        )
        
        val_transform = self._get_test_transform(
            mean=self.TINY_IMAGENET_MEAN,
            std=self.TINY_IMAGENET_STD
        )
    
        # Load datasets using the TinyImageNetDataset class
        train_dataset = TinyImageNetDataset(root=self.data_root, transform=train_transform, train=True)
        val_dataset = TinyImageNetDataset(root=self.data_root, transform=val_transform, train=False)
        
        return self._create_data_loaders(train_dataset, val_dataset, batch_size, batch_repetitions)
    
    def get_cutmix_loader(self, dataset, batch_size=128, alpha=1.0, batch_repetitions=1, num_classes=10):
        """
        Create a DataLoader with CutMix augmentation using torchvision's implementation.
        
        Args:
            dataset: Dataset to apply CutMix to
            batch_size: Batch size for dataloader
            alpha: Alpha parameter for Beta distribution
            batch_repetitions: Number of times to repeat each sample
            num_classes: Number of classes in the dataset
            
        Returns:
            DataLoader with CutMix augmentation
        """
        # Create the DataLoader
        if batch_repetitions > 1:
            loader = self._create_repeated_dataloader(
                dataset, batch_size, batch_repetitions)
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        # Create the CutMix transform
        cutmix_transform = v2.CutMix(num_classes=num_classes, alpha=alpha)
        
        # Create a function that applies CutMix to each batch
        def collate_fn(batch):
            from torch.utils.data._utils.collate import default_collate
            batch = default_collate(batch)
            return cutmix_transform(*batch)
        
        # Create a new DataLoader that uses our custom collate function
        cutmix_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn if batch_repetitions == 1 else None
        )
        
        # If using batch repetitions, we'll need to apply CutMix after batch creation
        if batch_repetitions > 1:
            # Return a wrapper that applies CutMix to each batch
            original_loader = loader
            
            class CutMixLoaderWrapper:
                def __init__(self, loader, transform):
                    self.loader = loader
                    self.transform = transform
                
                def __iter__(self):
                    for batch in self.loader:
                        yield self.transform(*batch)
                
                def __len__(self):
                    return len(self.loader)
            
            return CutMixLoaderWrapper(original_loader, cutmix_transform)
        
        return cutmix_loader
    
    def get_augmix_loader(self, dataset, batch_size=128, batch_repetitions=1, severity=3, mixture_width=3, chain_depth=-1, alpha=1.0):
        """
        Create a DataLoader with AugMix augmentation using torchvision's implementation.
        
        Args:
            dataset: Dataset to apply AugMix to
            batch_size: Batch size for dataloader
            batch_repetitions: Number of times to repeat each sample
            severity: The severity of base augmentation operators
            mixture_width: The number of augmentation chains
            chain_depth: The depth of augmentation chains (-1 for random between 1-3)
            alpha: The hyperparameter for probability distributions
            
        Returns:
            DataLoader with AugMix augmentation
        """
        # Create a new dataset with AugMix transform
        original_transform = dataset.transform if hasattr(dataset, 'transform') else None
        
        # Create AugMix transform
        augmix_transform = transforms.AugMix(
            severity=severity,
            mixture_width=mixture_width,
            chain_depth=chain_depth,
            alpha=alpha,
            all_ops=True
        )
        
        # Create a new transform that includes AugMix
        if original_transform:
            # For datasets that already have transforms
            # Need to apply normalization after AugMix
            if isinstance(original_transform, transforms.Compose):
                # Extract normalization transform if it exists
                norm_transform = None
                other_transforms = []
                
                for t in original_transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        norm_transform = t
                    elif not isinstance(t, transforms.ToTensor):  # Skip ToTensor as AugMix outputs tensor
                        other_transforms.append(t)
                
                # Create new transform sequence
                if norm_transform:
                    new_transform = transforms.Compose([
                        *other_transforms,
                        augmix_transform,
                        norm_transform
                    ])
                else:
                    new_transform = transforms.Compose([
                        *other_transforms,
                        augmix_transform
                    ])
            else:
                # Simple case - just apply AugMix after original transform
                new_transform = transforms.Compose([
                    original_transform,
                    augmix_transform
                ])
        else:
            # For datasets without transforms, just use AugMix
            new_transform = augmix_transform
        
        # Create a new dataset with the updated transform
        class AugMixWrappedDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
                
            def __getitem__(self, index):
                img, label = self.base_dataset[index]
                
                # Apply transform
                if self.transform:
                    img = self.transform(img)
                
                return img, label
                
            def __len__(self):
                return len(self.base_dataset)
        
        augmix_dataset = AugMixWrappedDataset(dataset, new_transform)
        
        # Create dataloader
        if batch_repetitions > 1:
            loader = self._create_repeated_dataloader(
                augmix_dataset, batch_size, batch_repetitions)
        else:
            loader = DataLoader(
                augmix_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        return loader
    
    def _get_train_transform(self, mean, std, size=32):
        """Create standard training transform with provided normalization stats."""
        return transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def _get_test_transform(self, mean, std):
        """Create standard testing transform with provided normalization stats."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def _create_data_loaders(self, train_dataset, test_dataset, batch_size, batch_repetitions):
        """Create data loaders with optional batch repetition."""
        if batch_repetitions > 1:
            train_loader = self._create_repeated_dataloader(
                train_dataset, batch_size, batch_repetitions)
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, test_loader
        
    def _create_repeated_dataloader(self, dataset, batch_size, repetitions):
        """
        Create a DataLoader that implements batch repetition as described in the MixMo paper.
        
        With batch repetition b=2, the dataloader produces batches where:
        - The total batch size remains the same (e.g. batch_size=128)
        - Each unique sample appears b times consecutively
        - Each batch will contain batch_size samples, but only batch_size/b unique samples
        
        Args:
            dataset: Dataset to create loader for
            batch_size: Total batch size for dataloader
            repetitions: Number of times each sample should be repeated (b in the paper)
            
        Returns:
            DataLoader with repeated samples
        """
        # Use a custom BatchSampler
        from torch.utils.data.sampler import BatchSampler
        
        # Create a sampler that repeats each index consecutively
        base_sampler = RepeatedSampler(dataset, repetitions)
        
        # Create a batch sampler to properly group indices
        batch_sampler = BatchSampler(
            base_sampler, 
            batch_size=batch_size,  # Keep the full batch size
            drop_last=True
        )
        
        # Create the dataloader using the batch sampler
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return loader

class RepeatedSampler(Sampler):
    """
    Sampler that generates indices for batch repetition as described in the MixMo paper.
    
    The sampler ensures each unique sample appears repetitions times consecutively,
    which ensures they'll be grouped together in the same batch by the dataloader.
    """
    def __init__(self, dataset, repetitions):
        self.dataset = dataset
        self.repetitions = repetitions
        self.num_samples = len(dataset)
    
    def __iter__(self):
        # Get random permutation of indices - these are the unique samples
        indices = torch.randperm(self.num_samples).tolist()
        
        # Create a list where each index appears repetitions times consecutively
        repeated_indices = []
        for idx in indices:
            # Add the same index repetitions times consecutively
            repeated_indices.extend([idx] * self.repetitions)
        
        return iter(repeated_indices)
    
    def __len__(self):
        return self.num_samples * self.repetitions

class TinyImageNetDataset(Dataset):
    """
    Dataset for TinyImageNet.
    """
    def __init__(self, root, transform=None, train=True):
        self.data = load_dataset("zh-plus/tiny-imagenet")
        self.root = root
        self.transform = transform
        self.train = train
        
        # Set the appropriate split based on train flag
        self.split = 'train' if self.train else 'valid'
        
    def __getitem__(self, index):
        # Get image and label from the dataset
        img = self.data[self.split]['image'][index]
        target = self.data[self.split]['label'][index]
        
        # Apply transform if available
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        # Return the length of the dataset for the current split
        return len(self.data[self.split]['image'])