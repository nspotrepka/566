from data.gaped import GAPED
from data.pmemo import PMEmo
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Composite(Dataset):
    def __init__(self, size=256, type='train'):
        gaped_data = GAPED(size)
        if type == 'train':
            pmemo_data = ConcatDataset([PMEmo(size, i) for i in range(5)])
        elif type == 'validation':
            pmemo_data = PMEmo(size, 5)
        elif type == 'test':
            pmemo_data = PMEmo(size, 6)
        self.gaped = iter(DataLoader(gaped_data, shuffle=True))
        self.pmemo = iter(DataLoader(pmemo_data, shuffle=True))

    def __getitem__(self, i):
        image, image_emotion = next(self.gaped)
        audio, audio_emotion = next(self.pmemo)
        return [image[0], image_emotion[0]], [audio[0], audio_emotion[0]]

    def __len__(self):
        return min(self.gaped.__len__(), self.pmemo.__len__())
