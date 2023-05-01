from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def cifar10_dataloaders(BATCH_SIZE, val=False, WORKERS=2):
    """
    Downloads CIFAR-10 dataset and creates DataLoader objects.

    Args:
        batch_size (int): The batch size for the DataLoader.
        val (bool, optional): Whether to create a validation DataLoader. Defaults to False.
        workers (int, optional): Number of workers for DataLoader. Defaults to 2.

    Returns:
        dict: A dictionary containing DataLoader objects for train, test, and optionally, validation set.
    """

    # Training set data transformation pipeline
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # Resize to imagenet size
            transforms.RandomHorizontalFlip(),  # Random horizontal flipping
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalize to ImageNet for better comparison
        ]
    )

    # Testing set data transformation pipeline
    transform_test = transforms.Compose(
        [
            transforms.Resize(224),  # Resize to imagenet size
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Normalize to ImageNet for better comparison
        ]
    )

    ds = datasets.CIFAR10(
        root="./.data", train=True, download=True, transform=transform_train
    )

    test_ds = datasets.CIFAR10(
        root="./.data", train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
    )

    # Case with validation set
    if val:
        train_ds, val_ds = random_split(ds, [45000, 5000])

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        )

        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
        )

        return {"train": train_loader, "val": val_loader, "test": test_loader}

    # Case without validation set
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
    )

    return {"train": train_loader, "test": test_loader}
