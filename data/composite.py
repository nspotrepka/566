from data.audio import Audio
from data.gaped import GAPED
from data.gaped import GAPED2
from data.lakh import Lakh
from data.pmemo import PMEmo
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CompositeEmotion(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 spectrogram=False, cache=False, shuffle=True,
                 validation=False, midi=False):
        self.composite = Composite(size, image_channels, audio_channels,
                                   spectrogram, cache, shuffle, validation,
                                   midi)
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
                 validation=False, midi=False, blur=False):
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

        self.midi = midi
        self.blur = blur

        # Choose which subset of music to use
        self.gaped = GAPED(size, image_channels, cache=cache,
            validation=validation)
        if midi:
            self.lakh = Lakh(size, 1, cache=cache, validation=validation)
        elif:
            self.gaped2 = GAPED2(size, image_channels, cache=cache,
            validation=validation)
        else:
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
        if midi:
            self.out_channels = self.lakh.channels
        elif blur:
            self.out_channels = self.gaped2.channels
        else:
            self.out_channels = self.pmemo.datasets[0].channels

        # Set up loaders and iterators
        self.image_loader = DataLoader(self.gaped, shuffle=shuffle)
        self.image_iter = iter(self.image_loader)
        if midi:
            self.audio_loader = DataLoader(self.lakh, shuffle=shuffle)
        elif blur:
            self.audio_loader = DataLoader(self.gaped2, shuffle=shuffle)
        else:
            self.audio_loader = DataLoader(self.pmemo, shuffle=shuffle)
        self.audio_iter = iter(self.audio_loader)

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
        image_tuple = self.__next__(self.image_iter, self.image_loader)
        audio_tuple = self.__next__(self.audio_iter, self.audio_loader)

        self.image_iter, image, image_emotion = image_tuple
        self.audio_iter, audio, audio_emotion = audio_tuple

        return [image, image_emotion], [audio, audio_emotion]

    def __len__(self):
        if self.midi:
            return min(self.gaped.__len__(), self.lakh.__len__())
        elif self.blur:
            return min(self.gaped.__len__(), self.gaped2.__len__())
        else:
            return min(self.gaped.__len__(), self.pmemo.__len__())
