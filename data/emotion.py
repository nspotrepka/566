import torch

class EmotionTransform:
    def __call__(self, emotion, reverse=False):
        if reverse:
            for i in range(2):
                emotion[i] = emotion[i] + 1
            emotion *= 0.5
        else:
            emotion *= 2
            for i in range(0, 2):
                emotion[i] = emotion[i] - 1
        return emotion

class Emotion:
    def __init__(self):
        self.transform = EmotionTransform()

class EmotionReader(Emotion):
    def __init__(self):
        super(EmotionReader, self).__init__()

    def __call__(self, emotion):
        emotion = self.transform(emotion)
        emotion = torch.from_numpy(emotion)
        return emotion.float()
