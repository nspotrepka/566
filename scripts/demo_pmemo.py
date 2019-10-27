import numpy as np
import common.setup as setup
from data.pmemo import PMEmo
import torchaudio

def main():
    setup.print_torch_version()

    batch_size = 8
    num_workers = 8
    dataset = PMEmo()
    loader = dataset.loader(batch_size, num_workers)
    device = setup.device()

    count = 0
    torchaudio.initialize_sox()
    for batch in loader:
        audio, emotion = batch
        audio.to(device)
        emotion.to(device)
        count = min(count + batch_size, dataset.__len__())
        print("Loaded", count, "/", dataset.__len__())
    torchaudio.shutdown_sox()

if __name__ == "__main__":
    main()
