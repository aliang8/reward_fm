import torch

from rfm.data.datasets.base import BaseDataset


class InfiniteDataset(torch.utils.data.Dataset):
    """
    Wrapper that makes any dataset appear infinite.

    The dataset reports a very large length and wraps around when accessing items.
    This is useful for training loops that need an "infinite" dataset.

    Usage:
        dataset = RFMDataset(config, ...)
        infinite_dataset = InfiniteDataset(dataset)
    """

    def __init__(self, dataset: BaseDataset):
        """
        Args:
            dataset: The dataset instance to wrap (should be a BaseDataset or compatible)
        """
        self.dataset = dataset
        self._max_length = 1_000_000

    def __len__(self):
        return self._max_length

    def __getitem__(self, idx):
        # Use modulo to wrap around when index exceeds dataset length
        return self.dataset[idx % len(self.dataset)]

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class RepeatedDataset(torch.utils.data.Dataset):
    """
    Wrapper that repeats a dataset when exhausted.

    The dataset will cycle through its items when the index exceeds the dataset length.
    The length of this dataset is the same as the wrapped dataset.

    Usage:
        dataset = RFMDataset(config, ...)
        repeated_dataset = RepeatedDataset(dataset)
    """

    def __init__(self, dataset: BaseDataset):
        """
        Args:
            dataset: The dataset instance to wrap (should be a BaseDataset or compatible)
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Use modulo to wrap around when index exceeds dataset length
        return self.dataset[idx % len(self.dataset)]

    def __getattr__(self, name):
        return getattr(self.dataset, name)
