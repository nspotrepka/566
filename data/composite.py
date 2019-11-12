from data.gaped import GAPED
from data.pmemo import PMEmo
from data.pmemo import AudioTransform
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Composite(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 cache=False, type='train', shuffle=True):
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

        self.gaped_loader = DataLoader(self.gaped, shuffle=shuffle)
        self.pmemo_loader = DataLoader(self.pmemo, shuffle=shuffle)

        self.gaped_iter = iter(self.gaped_loader)
        self.pmemo_iter = iter(self.pmemo_loader)

    def __getitem__(self, i):
        try:
            image, image_emotion = self.gaped_iter.next()
        except StopIteration:
            self.gaped_iter = iter(self.gaped_loader)
            image, image_emotion = self.gaped_iter.next()

        try:
            audio, audio_emotion = self.pmemo_iter.next()
        except StopIteration:
            self.pmemo_iter = iter(self.pmemo_loader)
            audio, audio_emotion = self.pmemo_iter.next()

        image = torch.squeeze(image, 0)
        image_emotion = torch.squeeze(image_emotion, 0)
        audio = torch.squeeze(audio, 0)
        audio_emotion = torch.squeeze(audio_emotion, 0)
        return [image, image_emotion], [audio, audio_emotion]

    def __len__(self):
        return max(self.gaped.__len__(), self.pmemo.__len__())
