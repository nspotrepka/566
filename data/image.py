import torch
from skimage import io, transform, util

class ImageTransform:
    def __init__(self, size, image_channels):
        self.size = size
        self.image_channels = image_channels

    def __call__(self, image, reverse=False):
        if reverse:
            # Pad channels
            channel_pad = max(0, self.image_channels - image.shape[0])
            image = util.pad(
                image,
                ((0, channel_pad), (0, 0), (0, 0)),
                mode='constant'
            )
            # Unscale
            image = (image + 1) / 2
            # Transpose dimensions
            image = image.T
        else:
            # Calculate padding
            lr_pad = 0
            tb_pad = 0
            if image.shape[0] < image.shape[1]:
                # Portrait
                lr_pad = (image.shape[1] - image.shape[0]) // 2
            else:
                # Landscape
                tb_pad = (image.shape[0] - image.shape[1]) // 2
            # Pad channels and width/height
            image = util.pad(
                image,
                ((lr_pad, lr_pad), (tb_pad, tb_pad), (0, self.image_channels)),
                mode='reflect'
            )
            # Crop channels
            image = image[:, :, :self.image_channels]
            # Resize width and height
            image = transform.resize(image, (self.size, self.size))
            # Transpose dimensions
            image = image.T
            # Scale
            image = image * 2 - 1
        return image

class Image:
    def __init__(self, size=256, image_channels=3):
        self.transform = ImageTransform(size, image_channels)
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
        image = image / (1e-9 + torch.max(image.min().abs(), image.max().abs()))
        image = self.transform(image, reverse=True)
        io.imsave(path, image)
