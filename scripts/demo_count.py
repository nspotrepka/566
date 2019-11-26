from data.pmemo import PMEmo
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

def main():
    size = 256
    audio_channels = 2
    cache = True
    train_chunks = 8
    shuffle = True

    pmemo = ConcatDataset([PMEmo(size, audio_channels, i, cache)
        for i in range(train_chunks)])
    pmemo_loader = DataLoader(pmemo, shuffle=shuffle)

    all = 767 * 8

    i = 0
    count = 0
    for batch in pmemo_loader:
        data, emotion = batch
        if data.min() == 0 and data.max() == 0:
            count += 1
        if i % 100 == 0:
            print('Iteration', i, '/', all, 'count =', count)
        i += 1
    print('count =', count)

if __name__ == '__main__':
    main()
