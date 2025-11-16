import torch

from rfm.data.datasets.base import BaseDataset


class InfiniteDataset(torch.utils.data.Dataset):
    """
    Wrapper that makes any dataset appear infinite.

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
        return self.dataset[idx]

    def __getattr__(self, name):
        return getattr(self.dataset, name)
