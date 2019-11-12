from data.image import GAPED
from data.pmemo import PMEmo
from data.pmemo import AudioTransform
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Composite(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 cache=False, shuffle=True):
        # Check for valid size
        assert size == 128 or size == 256 or size == 512
        chunks = PMEmo.full_length // PMEmo.length(size)

        # Choose which subset of music to use
        self.gaped = GAPED(size, image_channels, cache=cache)
        self.pmemo = ConcatDataset([PMEmo(size, audio_channels, i, cache)
            for i in range(chunks)])

        # Number of in/out channels for neural network
        self.in_channels = self.gaped.channels
        self.out_channels = self.pmemo.datasets[0].channels

        # Set up loaders and iterators
        self.gaped_loader = DataLoader(self.gaped, shuffle=shuffle)
        self.pmemo_loader = DataLoader(self.pmemo, shuffle=shuffle)
        self.gaped_iter = iter(self.gaped_loader)
        self.pmemo_iter = iter(self.pmemo_loader)

    def __next__(self, iterator, loader):
        try:
            data, emotion = iterator.next()
        except StopIteration:
            iterator = iter(loader)
            data, emotion = iterator.next()
        while data.min() == 0 and data.max() == 0:
            data, emotion = iterator.next()
        data = torch.squeeze(data, 0)
        emotion = torch.squeeze(emotion, 0)
        return (iterator, data, emotion)

    def __getitem__(self, i):
        gaped_tuple = self.__next__(self.gaped_iter, self.gaped_loader)
        pmemo_tuple = self.__next__(self.pmemo_iter, self.pmemo_loader)

        self.gaped_iter, image, image_emotion = gaped_tuple
        self.pmemo_iter, audio, audio_emotion = pmemo_tuple

        return [image, image_emotion], [audio, audio_emotion]

    def __len__(self):
        return max(self.gaped.__len__(), self.pmemo.__len__())
