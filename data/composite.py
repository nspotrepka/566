from data.gaped import GAPED
from data.pmemo import PMEmo
from data.pmemo import AudioTransform
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Composite(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 cache=False, type='train'):
        # Check for valid size
        # Should be either 128 or 256
        chunks = PMEmo.full_length // PMEmo.length(size)
        assert chunks >= 3

        # Just use all images
        self.gaped = GAPED(size, image_channels, cache=cache)

        # 70/30 split
        num_test = int(chunks * 0.3 / 2)
        num_train = chunks - 2 * num_test

        # Choose which subset of music to use
        if type == 'train':
            self.pmemo = ConcatDataset([PMEmo(size, audio_channels, i, cache)
                for i in range(num_train)])
        elif type == 'validation':
            self.pmemo = ConcatDataset([PMEmo(size, audio_channels, i, cache)
                for i in range(num_train, num_train + num_test)])
        elif type == 'test':
            self.pmemo = ConcatDataset([PMEmo(size, audio_channels, i, cache)
                for i in range(num_train + num_test, num_train + num_test * 2)])

        # Number of in/out channels for neural network
        self.in_channels = self.gaped.channels
        self.out_channels = self.pmemo.datasets[0].channels

        self.gaped_iter = iter(DataLoader(self.gaped, shuffle=True))
        self.pmemo_iter = iter(DataLoader(self.pmemo, shuffle=True))

    def __getitem__(self, i):
        image, image_emotion = next(self.gaped_iter)
        audio, audio_emotion = next(self.pmemo_iter)
        return [image[0], image_emotion[0]], [audio[0], audio_emotion[0]]

    def __len__(self):
        return max(self.gaped.__len__(), self.pmemo.__len__())
