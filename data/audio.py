import numpy as np
import numpy.fft as fft
from scipy.interpolate import interp1d
import torch
import torchaudio

class AudioTransform:
    def __init__(self, size, audio_channels, log_scale=False):
        self.size = size
        self.audio_channels = audio_channels

    def __call__(self, audio, reverse=False):
        if reverse:
            # To 1-D
            audio = audio.contiguous().view(
                self.audio_channels, self.size * self.size)
        else:
            # Clip length
            audio = audio[:, :self.size * self.size]
            # To 2-D
            audio = audio.contiguous().view(
                self.audio_channels, self.size, self.size)
            # Normalize
            min = audio.min().abs()
            max = audio.max().abs()
            audio = audio / (1e-9 + torch.max(min, max))

        return audio

class Audio:
    rate = 44100
    full_length = 30

    def __init__(self, size=256, audio_channels=2):
        self.length = Audio.length(size)
        self.transform = AudioTransform(size, audio_channels)
        self.channels = audio_channels

    # Get length in seconds
    def length(size):
        return 3 * (size / 256) ** 2

class AudioReader(Audio):
    def __init__(self, size=256, audio_channels=2, offset=0):
        super(AudioReader, self).__init__(size, audio_channels)
        assert offset >= 0 and offset < Audio.full_length // self.length
        cue = offset * self.length
        self.chain = torchaudio.sox_effects.SoxEffectsChain()
        self.chain.append_effect_to_chain('rate', [str(Audio.rate)])
        self.chain.append_effect_to_chain('channels', [str(audio_channels)])
        self.chain.append_effect_to_chain('pad', ['0', str(Audio.full_length)])
        self.chain.append_effect_to_chain('trim', [str(cue), str(self.length)])
        self.dim = [audio_channels, self.length * Audio.rate]

    def __call__(self, path):
        self.chain.set_input_file(path)
        try:
            audio, _ = self.chain.sox_build_flow_effects()
        except RuntimeError:
            audio = torch.zeros(self.dim)
        audio = self.transform(audio)
        return audio.float()

class AudioWriter(Audio):
    output_rate = 44100

    def __init__(self, size=256, audio_channels=2):
        super(AudioWriter, self).__init__(size, audio_channels)
        self.resample = torchaudio.transforms.Resample(
            Audio.rate, AudioWriter.output_rate)

    def __call__(self, path, audio):
        audio = self.transform(audio, reverse=True)
        # This takes a long time
        audio = self.resample(audio)
        audio = audio / (1e-9 + torch.max(audio.min().abs(), audio.max().abs()))
        torchaudio.save(path, audio, AudioWriter.output_rate)
