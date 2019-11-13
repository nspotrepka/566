import torch
from skimage import io, transform, util

class ImageTransform:
    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels
        self.padding = (self.width - self.height) // 2

    def __call__(self, image, reverse=False):
        if reverse:
            # Unscale
            image = (image + 1) / 2
            # Crop height
            # Skip to observe artifacts
            # image = image[:, self.padding:self.padding + self.height, :]
            # Transpose dimensions
            image = image.T
        else:
            # Resize width and height
            image = transform.resize(image, (self.width, self.height))
            # Transpose dimensions
            image = image.T
            # Pad channels and height
            image = util.pad(
                image,
                ((0, self.channels), (self.padding, self.padding), (0, 0)),
                mode='reflect')
            # Crop channels
            image = image[:self.channels, :, :]
            # Scale
            image = image * 2 - 1
        return image

class Image:
    def __init__(self, size=256, image_channels=3):
        self.transform = ImageTransform(size, size * 3 // 4, image_channels)
        self.channels = image_channels

class ImageReader(Image):
    def __init__(self, size=256, image_channels=3):
        super(ImageReader, self).__init__(size, image_channels)

    def __call__(self, path):
        image = io.imread(path)
        image = self.transform(image)
        image = torch.from_numpy(image)
        return image.float()

class ImageWriter(Image):
    def __init__(self, size=256, image_channels=3):
        super(ImageWriter, self).__init__(size, image_channels)

    def __call__(self, path, image):
        image = self.transform(image, reverse=True)
        io.imsave(path, image)
