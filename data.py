from torchvision import datasets, transforms
import numpy as np
import torch
import gc
import psutil

DATASET_NAMES = ["cifar10", "mnist", "fmnist", "svhn"]#, "stl10"]


def print_memory():
    # To get to memory and cpu information
    split_bar = "=" * 20
    memory_info = psutil.virtual_memory()._asdict()
    print(f"{split_bar} Memory Usage {split_bar}")
    for k, v in memory_info.items():
        print(k, v)


######## BINARIZERS #########
def make_binary_label_geq5(label):
    if isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
        label = 1 * (label >= 5)
    elif isinstance(label, list):
        label = [1 * (l >= 5) for l in label]
    return label


def make_binary_label_last(label):
    if isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
        label = 1 * (label == 9)
    elif isinstance(label, list):
        label = [1 * (l == 9) for l in label]
    return label


def make_binary_label_odd(label):
    if isinstance(label, torch.Tensor) or isinstance(label, np.ndarray):
        label = 1 * (label % 2 == 1)
    elif isinstance(label, list):
        label = [1 * (l % 2 == 1) for l in label]
    return label


def get_binarizer_fn(binarizer):
    if binarizer is None:
        binary_transform = torch.tensor
    elif binarizer == "geq5":
        binary_transform = make_binary_label_geq5
    elif binarizer == "last":
        binary_transform = make_binary_label_last
    elif binarizer == "odd":
        binary_transform = make_binary_label_odd
    else:
        raise ValueError(f"Invalid binarizer {binarizer}")
    return binary_transform


##### slow implementation #####
class EnumeratedUnlabeledWrapperSlow(torch.utils.data.Dataset):
    def __init__(self, wrapped_dataset):
        self.wrapped_dataset = wrapped_dataset

    def __getitem__(self, index):
        data, label = self.wrapped_dataset[index]
        return data, index

    def __len__(self):
        return len(self.wrapped_dataset)


def get_labeled_unlabeled_slow(train_dataset, labeled_indices):
    labeled_ds = torch.utils.data.Subset(train_dataset, labeled_indices)
    unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))
    unlabeled_ds = torch.utils.data.Subset(
        EnumeratedUnlabeledWrapperSlow(train_dataset), unlabeled_indices
    )  # indices are needed for selection and we unlabel the dataset to prove we aren't cheating
    return labeled_ds, unlabeled_ds


def get_torchvision_dataset(dataset_name, binarizer=None):
    # as in distil library
    binary_transform = get_binarizer_fn(binarizer)
    dataset_name = dataset_name.lower()

    # print(data_config)
    if dataset_name == "cifar10":
        download_path = "./downloaded_data/"

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            download_path,
            download=True,
            train=True,
            transform=train_transform,
            target_transform=binary_transform,
        )
        test_dataset = datasets.CIFAR10(
            download_path,
            download=True,
            train=False,
            transform=test_transform,
            target_transform=binary_transform,
        )

    elif dataset_name == "mnist":
        download_path = "./downloaded_data/"
        image_dim = 28
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_dim, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_dim, image_dim)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = datasets.MNIST(
            download_path,
            download=True,
            train=True,
            transform=train_transform,
            target_transform=binary_transform,
        )
        test_dataset = datasets.MNIST(
            download_path,
            download=True,
            train=False,
            transform=test_transform,
            target_transform=binary_transform,
        )

    elif dataset_name == "fmnist":
        download_path = "./downloaded_data/"

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )  # Use mean/std of MNIST

        train_dataset = datasets.FashionMNIST(
            download_path,
            download=True,
            train=True,
            transform=train_transform,
            target_transform=binary_transform,
        )
        test_dataset = datasets.FashionMNIST(
            download_path,
            download=True,
            train=False,
            transform=test_transform,
            target_transform=binary_transform,
        )

    elif dataset_name == "svhn":
        download_path = "./downloaded_data/"

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )  # ImageNet mean/std

        train_dataset = datasets.SVHN(
            download_path,
            download=True,
            split="train",
            transform=train_transform,
            target_transform=binary_transform,
        )
        test_dataset = datasets.SVHN(
            download_path,
            download=True,
            split="test",
            transform=test_transform,
            target_transform=binary_transform,
        )

    elif dataset_name == "stl10":
        download_path = "./downloaded_data/"

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )  # ImageNet mean/std

        train_dataset = datasets.STL10(
            download_path,
            download=True,
            split="train",
            transform=train_transform,
            target_transform=binary_transform,
        )
        test_dataset = datasets.STL10(
            download_path,
            download=True,
            split="test",
            transform=test_transform,
            target_transform=binary_transform,
        )

    return train_dataset, test_dataset


######## efficient implementation #########


class InMemoryDataset:
    def __init__(
        self, torchvision_dataset, binarizer_fn=None, use_only_first=False, device=None
    ):
        if isinstance(torchvision_dataset.data, np.ndarray):
            self.data = torch.tensor(torchvision_dataset.data)
        else:
            self.data = torchvision_dataset.data.clone().detach().requires_grad_(False)
        if len(self.data.shape) == 3:
            self.data = self.data.unsqueeze(3)
        self.data = (
            self.data.permute(0, 3, 1, 2)
            if not torchvision_dataset.__class__.__name__ in ["SVHN", "STL10"]
            else self.data
        )
        self.targets = (
            torchvision_dataset.targets
            if hasattr(torchvision_dataset, "targets")
            else torchvision_dataset.labels
        )
        if binarizer_fn is not None:
            self.targets = binarizer_fn(self.targets)
        if isinstance(self.targets, torch.Tensor):
            self.targets = self.targets.detach().requires_grad_(False).float()
        else:
            self.targets = torch.tensor(self.targets).float()
        if use_only_first:
            self.data = self.data[:use_only_first]
            self.targets = self.targets[:use_only_first]
        self.indices = torch.arange(len(self.data))
        if self.data.shape[1] == 1:
            self.data = transforms.Normalize((0.1307,), (0.3081,))(
                self.data / self.data.max()
            )
        elif self.data.shape[1] == 3:
            self.data = transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )(self.data / self.data.max())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def __len__(self):
        assert len(self.data) == len(self.indices)
        return len(self.data)

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        self.targets = self.targets.to(device) if self.targets is not None else None
        self.indices = self.indices.to(device)
        return self

    def select_indices(self, selected_indices):
        data = self.data[selected_indices]
        targets = self.targets[selected_indices] if self.targets is not None else None
        indices = self.indices[selected_indices]
        return data, targets, indices

    def update_indices(self, selected_indices):
        self.data, self.targets, self.indices = self.select_indices(selected_indices)

    def get_new_ds_from_indices(self, selected_indices):
        return InMemoryDatasetFromAttributes(*self.select_indices(selected_indices))


class InMemoryDatasetFromAttributes(InMemoryDataset):
    def __init__(self, data: torch.Tensor, targets, indices):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.device = data.device


######### API ###########
def get_dataset(
    dataset_name,
    binarizer=None,
    use_only_first=5000,
    use_only_first_test=100,
    device=None,
):
    train_ds, test_ds = get_torchvision_dataset(dataset_name, binarizer)
    binarizer_fn = get_binarizer_fn(binarizer)
    train_ds = InMemoryDataset(
        train_ds,
        binarizer_fn=binarizer_fn,
        use_only_first=use_only_first,
        device=device,
    )
    test_ds = InMemoryDataset(
        test_ds,
        binarizer_fn=binarizer_fn,
        use_only_first=use_only_first_test,
        device=device,
    )
    return train_ds, test_ds


class UnlabeledWrapper(InMemoryDataset):
    def __init__(self, wrapped_dataset):
        self.data = wrapped_dataset.data
        self.indices = wrapped_dataset.indices
        self.device = wrapped_dataset.device
        self.targets = None


def get_labeled_unlabeled(train_dataset: InMemoryDataset, labeled_indices: list[int], torch_version=False):
    labeled_ds = train_dataset.get_new_ds_from_indices(labeled_indices)
    unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))
    unlabeled_ds = UnlabeledWrapper(train_dataset).get_new_ds_from_indices(
        unlabeled_indices
    )  # indices are needed for selection and we unlabel the dataset to prove we aren't cheating
    if torch_version:
        return labeled_ds, unlabeled_ds, torch.tensor(unlabeled_indices)
    return labeled_ds, unlabeled_ds, np.array(unlabeled_indices)


###### testing #######3
def download_datasets():
    for ds_name in DATASET_NAMES:
        train_ds, test_ds = get_dataset(ds_name, 'geq5')
    print('all downloaded')
 
def prepare_datasets(binarizer="geq5"):
    print_memory()
    for ds_name in DATASET_NAMES:
        train_ds, test_ds = get_dataset(ds_name, binarizer)
        inmemory_shape = (len(train_ds),) + train_ds[0][0].shape
        print("-" * 10)
        print(f"inmemory shape {inmemory_shape}")
        aa = torch.rand(inmemory_shape)  # try to allocate tensor of that size
        print("everything is fine")
        print_memory()
        gc.collect()


### Intelligent annotation ###
def get_initial_info(train_dataset, labeled_indices):
    # this will be given as input to the joint model
    # in general this is a set of possible leabelings but in fact can and should be reduced
    # we have two cases, the case with the guess over n and the general case. They are so different computationally that we will treat them separately
    # for the case using guesses we have that every positive answer gives us some labels over the data and every negative answer gives us some incorrect labels. Then we can fit the model to the labels so far and to the incorrect labels too. We should treat all answers equally. To do this we should build the set of correct labels and the set of incorrect labels. Then we sum the BCE losses for both cases. In this case the info will be {'labeled': ..., 'incorrect': ...}
    raise NotImplementedError
