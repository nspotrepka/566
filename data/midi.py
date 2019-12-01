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
            image = transform.resize(image, (128, 128))
            image = filters.threshold_otsu(image)

        else:
            image = transform.resize(image, (self.size, self.size),
                anti_aliasing=False)

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
