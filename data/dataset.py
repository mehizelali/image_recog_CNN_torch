import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_dataset(name: str, root: str = './data', img_res: int = 32, download: bool = True, to_rgb: bool = True):
    """
    Load CIFAR10, CIFAR100, SVHN, MNIST, FashionMNIST, or KMNIST datasets with optional augmentation.
    
    Args:
        name (str): Dataset name: "cifar10", "cifar100", "svhn", "mnist", "fashion-mnist", or "kmnist".
        root (str): Directory to store the data.
        img_res (int): Target image resolution.
        download (bool): Whether to download the dataset.
        to_rgb (bool): Convert 1-channel grayscale images to 3-channel RGB.

    Returns:
        Tuple[train_dataset, test_dataset, num_classes]
    """
    name = name.lower()

    # Add grayscale-to-RGB conversion if needed
    maybe_grayscale_to_rgb = [transforms.Grayscale(num_output_channels=3)] if to_rgb else []

    # Training transform
    train_transform = transforms.Compose([
        transforms.Resize((img_res, img_res)),
        *maybe_grayscale_to_rgb,
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor()
    ])

    # Testing transform
    test_transform = transforms.Compose([
        transforms.Resize((img_res, img_res)),
        *maybe_grayscale_to_rgb,
        transforms.ToTensor()
    ])

    if name == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=download)
        num_classes = 10

    elif name == "cifar100":
        train_dataset = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.CIFAR100(root=root, train=False, transform=test_transform, download=download)
        num_classes = 100

    elif name == "svhn":
        train_dataset = datasets.SVHN(root=root, split='train', transform=train_transform, download=download)
        test_dataset = datasets.SVHN(root=root, split='test', transform=test_transform, download=download)
        num_classes = 10

    elif name == "mnist":
        train_dataset = datasets.MNIST(root=root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.MNIST(root=root, train=False, transform=test_transform, download=download)
        num_classes = 10

    elif name == "fashion-mnist":
        train_dataset = datasets.FashionMNIST(root=root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.FashionMNIST(root=root, train=False, transform=test_transform, download=download)
        num_classes = 10

    elif name == "kmnist":
        train_dataset = datasets.KMNIST(root=root, train=True, transform=train_transform, download=download)
        test_dataset = datasets.KMNIST(root=root, train=False, transform=test_transform, download=download)
        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {name}. Choose from 'cifar10', 'cifar100', 'svhn', 'mnist', 'fashion-mnist', or 'kmnist'.")

    return train_dataset, test_dataset, num_classes
