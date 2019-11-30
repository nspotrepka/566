from data.audio import Audio
from data.gaped import GAPED
from data.pmemo import PMEmo
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CompositeEmotion(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 spectrogram=False, cache=False, shuffle=True,
                 validation=False):
        self.composite = Composite(size, image_channels, audio_channels,
                                   spectrogram, cache, shuffle, validation)
        self.in_channels = self.composite.in_channels + 4
        self.out_channels = self.composite.out_channels + 4

    def __getitem__(self, i):
        image_batch, audio_batch = self.composite.__getitem__(i)
        image, image_emotion = image_batch
        audio, audio_emotion = audio_batch
        image_emotion = image_emotion[:2]
        audio_emotion = audio_emotion[:2]
        shape = [image.shape[2], image.shape[1], image_emotion.shape[0]]
        image_emotion = image_emotion.expand(shape).T
        audio_emotion = audio_emotion.expand(shape).T
        image_emotion += torch.randn(image_emotion.shape) * 1e-9
        audio_emotion += torch.randn(audio_emotion.shape) * 1e-9
        image = torch.cat([image, image_emotion, audio_emotion])
        audio = torch.cat([audio, audio_emotion, image_emotion])
        return [image, []], [audio, []]

    def __len__(self):
        return self.composite.__len__()

class Composite(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 spectrogram=False, cache=False, shuffle=True,
                 validation=False):
        # Check for valid size
        assert size == 128 or size == 256 or size == 512
        chunks = int(Audio.full_length // Audio.length(size, spectrogram))
        if size == 128:
            train_chunks = int(chunks * 0.9)
        elif size == 256:
            train_chunks = int(chunks * 0.8)
        elif size == 512:
            train_chunks = int(chunks * 0.5)
        val_chunks = chunks - train_chunks

        # Choose which subset of music to use
        self.gaped = GAPED(size, image_channels, cache=cache)
        if validation:
            self.pmemo = ConcatDataset(
                [PMEmo(size, audio_channels, spectrogram, i, cache)
                for i in range(train_chunks, train_chunks + val_chunks)])
        else:
            self.pmemo = ConcatDataset(
                [PMEmo(size, audio_channels, spectrogram, i, cache)
                for i in range(train_chunks)])

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
        return min(self.gaped.__len__(), self.pmemo.__len__())
