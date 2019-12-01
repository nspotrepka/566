import common.setup as setup
from data.lakh import midi_paths
from music21.midi import MidiException
from scripts.convert_midi_to_image import midi2image
import time
from torch.utils.data import Dataset

class ConvertLakh(Dataset):
    def __init__(self):
        self.paths = midi_paths()
        self.names = list(self.paths.keys())

    def __getitem__(self, i):
        key = self.names[i]
        try:
            midi2image(self.paths[key])
        except Exception as e:
            print(e)
        return i

    def __len__(self):
        return len(self.names)

def main():
    device = setup.device()

    batch_size = 8
    num_workers = 12
    dataset = ConvertLakh()
    loader = setup.load(dataset, batch_size, num_workers)

    start_time = time.time()

    count = 0
    for batch in loader:
        count = min(count + batch_size, dataset.__len__())
        print('Converted', count, '/', dataset.__len__())

    end_time = time.time()
    print('Time:', end_time - start_time, 'sec')

if __name__ == '__main__':
    main()
