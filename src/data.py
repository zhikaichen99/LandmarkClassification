import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    Inputs:
        batch_size (int) - size of the mini-batches
        valid_size (float) - fraction of the dataset to use for validation. 
        num_workers (int) - number of workers to use in the data loaders. Use -1 to mean "use all my cores"
        limit (int) - maximum number of data points to consider
    Outputs:
        dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()

    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256), # resize image to 256
            transforms.RandomCrop(224), # crop image to 224
            # Apply random affine transformations (rotation, translation, shear)
            transforms.RandomAffine(scale = (0.9, 1.1), translate = (0.1, 0.1), degrees = 10),
            # Apply horizontal flip
            transforms.RandomHorizontalFlip(0.5),
            # Convert to tensor
            transforms.ToTensor(),
            # Apply normalization
            transforms.Normalize(mean,std)
        ]
        ),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        ),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        ),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        # the data_transforms dictionary
        transform = data_transforms["train"]
    )
    valid_data = datasets.ImageFolder(
        base_path / "train",
        # YOUR CODE HERE: add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform = data_transforms['valid']
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size = batch_size,
        sampler = valid_sampler,
        num_workers = num_workers,
    )

    # create the test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        transform = data_transforms['test']
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size = batch_size,
        sampler = test_sampler,
        num_workers = num_workers,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    Inputs:
        data_loaders - dictionary containing data loaders
        max_n (int) - maximum number of images to show
    Output:
        None
    """


    dataiter  = iter(data_loaders['train'])

    images, labels  = dataiter.next()

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names  = data_loaders['train'].dataset.classes

    # Convert from BGR (the format used by pytorch) to RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])





