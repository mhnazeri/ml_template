import os.path as osp
import uuid

from diskcache import Index
from typing import Callable, Optional
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class CustomDataset(Dataset):

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
        download: bool = False, read_only: int = 10):
        if transform is None:
            # transform = transforms.ToTensor()
            self.transform = transforms.ToTensor()
        self.dataset = MNIST(root, train=train, transform=self.transform, download=download)

        if read_only > 0:
            self.dataset.data = self.dataset.data[:read_only]
            self.dataset.targets = self.dataset.targets[:read_only]

        mode = 'train' if train else 'val'
        self.create_uuid(root + f'/{mode}' + '-cache')

    def create_uuid(self, directory: Optional[str] = './cache'):
        """Create UID for samples to get the actual name for later use cases"""
        print("START caching file names.")
        if osp.exists(directory):
            self.cache_names = Index(directory)
        else:
            self.cache_names = Index(directory, {
                    str(index): str(uuid.uuid5(uuid.NAMESPACE_X500, str(index)))  for index, _ in enumerate(self.dataset)
                })
            # use values as keys
            # self.cache_names.update({
            #         value: key for key, value in self.cache_names.items()
            #     })
        print("END caching file names.")

    def get_uuid(self, file_name):
        """Works both ways, given filename returns uuid and vice versa"""
        return self.cache_names[file_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return a sample in the form: (uuid, image, label)"""
        return self.get_uuid(str(idx)), self.dataset.data[idx].unsqueeze(0) / 255.0, self.dataset.targets[idx]


if __name__ == "__main__":
    pass
