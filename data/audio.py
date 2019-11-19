import numpy as np
import numpy.fft as fft
from scipy.interpolate import interp1d
import torch
import torchaudio

class AudioTransform:
    def __init__(self, size, audio_channels, rate, log_scale=False):
        self.size = size
        self.fft_size = size * 2
        self.hop_size = size * 2
        self.fft_out = self.fft_size // 2 + 1
        self.audio_channels = audio_channels
        self.rate = rate
        self.log_scale = log_scale

    def __call__(self, audio, reverse=False):
        y = np.abs(np.fft.fftfreq(self.fft_size)[1:self.fft_out]) * self.rate
        y = torch.tensor(y).float()
        scale = torch.tensor(range(1, self.fft_out + 1))
        if reverse:
            # Unscale
            audio = audio * self.size * self.size
            # Convert frequency axis to linear scale
            if self.log_scale:
                log_y = np.log2(y + 1)
                new_y = np.linspace(min(log_y), max(log_y), self.size)
                interpolation = interp1d(new_y, audio, fill_value='extrapolate')
                audio = torch.tensor(interpolation(log_y), dtype=torch.float32)
            # Extend time to account for samples that were clipped in FFT
            new_audio = torch.zeros(
                self.audio_channels * 2, self.size + 1, self.fft_out)
            new_audio[:, :self.size, :self.size] = audio
            audio = new_audio
            # Separate audio channels with real/imag
            audio = audio.contiguous().view(
                self.audio_channels, 2, -1, self.fft_out)
            # Permute dimensions
            audio = audio.permute(0, 3, 2, 1)
            # Scale values
            audio = audio / scale[None, :, None, None]
            # Perform inverse FFT
            audio = torchaudio.functional.istft(
                audio, self.fft_size, self.hop_size, center=False,
                length = 2 * self.fft_out * self.size)
        else:
            # Perform FFT
            audio = torch.stft(
                audio, self.fft_size, self.hop_size, center=False)
            # Scale values
            audio = audio * scale[None, :, None, None]
            # Permute dimensions
            audio = audio.permute(0, 3, 2, 1)
            # Combine audio channels with real/imag
            audio = audio.contiguous().view(
                self.audio_channels * 2, -1, self.fft_out)
            # Clip length and Nyquist frequency
            audio = audio[:, :self.size, :self.size]
            # Convert frequency axis to log scale
            if self.log_scale:
                log_y = np.log2(y + 1)
                new_y = np.linspace(min(log_y), max(log_y), self.size)
                interpolation = interp1d(log_y, audio, fill_value='extrapolate')
                audio = torch.tensor(interpolation(new_y), dtype=torch.float32)
            # Scale
            audio = audio / self.size / self.size

        return audio

class Audio:
    rate = 44100
    full_length = 30

    def __init__(self, size=256, audio_channels=2):
        self.length = Audio.length(size)
        self.transform = AudioTransform(size, audio_channels, Audio.rate)
        self.channels = audio_channels * 2

    # Get length in seconds
    def length(size):
        return 30#int(4 * (size / 256) ** 2)

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
