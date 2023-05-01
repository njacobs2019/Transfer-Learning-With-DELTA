from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def cifar10_datasets(val=False):
    """
    Downloads CIFAR-10 dataset and returns a dictionary of dataset objects.

    This function applies data transformation pipelines for the training and testing sets,
    and optionally creates a validation set from the training data.

    Args:
        val (bool, optional): Whether to create a validation set. Defaults to False.

    Returns:
        dict: A dictionary containing dataset objects for the train, test, and
              optionally, validation set. The keys in the dictionary are "train",
              "test", and "val" (if a validation set is requested).
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

    # Case with validation set
    if val:
        train_ds, val_ds = random_split(ds, [45000, 5000])
        return {"train": train_ds, "val": val_ds, "test": test_ds}

    # Case without validation set
    return {"train": ds, "test": test_ds}
