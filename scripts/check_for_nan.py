import common.setup as setup
from data.composite import Composite
import torch

def main():
    size = 256
    image_channels = 3
    audio_channels = 2
    batch_size = 1
    train = Composite(size, image_channels, audio_channels, shuffle=False)
    train_loader = setup.load(train, batch_size)
    setup.init_audio()
    i = 0
    for batch in train_loader:
        image_batch, audio_batch = batch
        image, image_emotion = image_batch
        audio, audio_emotion = audio_batch
        print("Batch:", i)
        if torch.isnan(image).any():
            print('Invalid image in batch:', i)
            print(batch)
        if torch.isnan(image_emotion).any():
            print('Invalid image emotion in batch:', i)
            print(batch)
        if torch.isnan(audio).any():
            print('Invalid audio in batch:', i)
            print(batch)
        if torch.isnan(audio_emotion).any():
            print('Invalid audio emotion in batch:', i)
            print(batch)
        i += 1
    setup.shutdown_audio()

if __name__ == '__main__':
    main()
