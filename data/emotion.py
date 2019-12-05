import numpy as np
import torch

class EmotionTransform:
    def __call__(self, emotion, reverse=False):
        new_emotion = np.array(emotion)
        if reverse:
            # Add 1 to mean
            for i in range(2):
                new_emotion[i] = new_emotion[i] + 1
            # Unscale
            new_emotion *= 0.5
        else:
            # Scale
            new_emotion *=  2
            # Subtract 1 from mean
            for i in range(0, 2):
                new_emotion[i] = new_emotion[i] - 1
        return new_emotion

class Emotion:
    def __init__(self):
        self.transform = EmotionTransform()

class EmotionReader(Emotion):
    def __init__(self):
        super(EmotionReader, self).__init__()

    def __call__(self, emotion):
        emotion = self.transform(emotion)
        emotion = torch.from_numpy(emotion)
        print(emotion)
        return emotion.float()
