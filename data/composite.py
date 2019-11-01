from data.gaped import GAPED
from data.pmemo import PMEmo
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Composite(Dataset):
    def __init__(self, size=256, image_channels=3, audio_channels=2,
                 type='train'):
        self.gaped = GAPED(size)

        chunks = PMEmo.full_length // PMEmo.length(size)
        assert chunks >= 3
        # 70/30 split
        num_test = int(chunks * 0.3 / 2)
        num_train = chunks - 2 * num_test

        if type == 'train':
            self.pmemo = ConcatDataset([PMEmo(size, i, audio_channels)
                for i in range(num_train)])
        elif type == 'validation':
            self.pmemo = ConcatDataset([PMEmo(size, i, audio_channels)
                for i in range(num_train, num_train + num_test)])
        elif type == 'test':
            self.pmemo = ConcatDataset([PMEmo(size, i, audio_channels)
                for i in range(num_train + num_test, num_train + 2 * num_test)])

        self.in_channels = self.gaped.channels
        self.out_channels = self.pmemo.datasets[0].channels

        self.gaped_iter = iter(DataLoader(self.gaped, shuffle=True))
        self.pmemo_iter = iter(DataLoader(self.pmemo, shuffle=True))

    def __getitem__(self, i):
        image, image_emotion = next(self.gaped_iter)
        audio, audio_emotion = next(self.pmemo_iter)
        print(audio[0].shape)
        return [image[0], image_emotion[0]], [audio[0], audio_emotion[0]]

    def __len__(self):
        return min(self.gaped.__len__(), self.pmemo.__len__())
