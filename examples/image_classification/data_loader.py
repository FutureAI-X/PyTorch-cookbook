from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_training_dataloader(batch_size:int = 64):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    return DataLoader(training_datasets, batch_size=batch_size)

def get_test_dataloader(batch_size:int = 64):
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return DataLoader(test_datasets, batch_size=batch_size)