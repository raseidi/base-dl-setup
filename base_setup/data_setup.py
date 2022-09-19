import torch
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def get_test_data(batch_size=16):
    x = torch.rand((128, 32))
    y = torch.randint(0, 9, (128,))
    data_loader = DataLoader(TestDataset(x, y), batch_size=batch_size)
    return data_loader
