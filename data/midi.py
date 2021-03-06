from scripts.convert_image_to_midi import image2midi
from scripts.convert_midi_to_image import midi2image
from skimage import io, transform, filters
import torch

class MidiTransform:
    def __init__(self, size, image_channels):
        self.size = size
        self.image_channels = image_channels
        self.channels = image_channels

    def __call__(self, image, reverse=False):
        if reverse:
            image = torch.flip(image, [2])
            image = torch.squeeze(image, 0)
            image = transform.resize(image, (128, 128))
            threshold = filters.threshold_otsu(image)
            image = image > threshold
            image = image.astype(int) * 255
            image = image.T

        else:
            image = transform.resize(image, (self.size, self.size),
                anti_aliasing=False)
            # Transpose dimensions
            image = image.T
            # Scale
            image = image * 2 - 1
            image = torch.from_numpy(image).float()
            if len(image.shape) == 2:
                image = torch.unsqueeze(image, 0)
            image = torch.flip(image, [2])

        return image

class Midi:
    def __init__(self, size=256, image_channels=1):
        self.transform = MidiTransform(size, image_channels)
        self.channels = self.transform.channels

class MidiReader(Midi):
    def __init__(self, size=256, image_channels=1):
        super(MidiReader, self).__init__(size, image_channels)

    def __call__(self, path):
        image = io.imread(path)
        image = self.transform(image)
        return image

class MidiWriter(Midi):
    def __init__(self, size=256, image_channels=1):
        super(MidiWriter, self).__init__(size, image_channels)

    def __call__(self, path, image):
        image = self.transform(image, reverse=True)
        image2midi(image, path)
