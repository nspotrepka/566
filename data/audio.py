import math
import numpy as np
import numpy.fft as fft
from scipy.interpolate import interp1d
import torch
import torchaudio

class AudioTransformSpectrogram:
    def __init__(self, size, audio_channels, mel_scale=False):
        self.size = size
        self.audio_channels = audio_channels
        self.channels = audio_channels * 2
        self.mel_scale = mel_scale

        self.fft_size = size * 2
        self.hop_size = size * 2
        self.fft_out = self.fft_size // 2 + 1

    def __call__(self, audio, reverse=False):
        y = np.abs(np.fft.fftfreq(self.fft_size)[1:self.fft_out]) * Audio.rate
        y = torch.tensor(y).float()
        mel_y = np.log(1 + y / 1000) * 1000 / np.log(2)
        new_y = np.linspace(min(mel_y), max(mel_y), self.size)
        if reverse:
            # Flip y
            audio = torch.flip(audio, [2])
            # Convert frequency axis to linear scale
            if self.mel_scale:
                interpolate = interp1d(new_y, audio, fill_value='extrapolate')
                audio = torch.tensor(interpolate(mel_y), dtype=torch.float32)
            # Separate
            mag = audio[0:2]
            phase = audio[2:4]
            # Unscale
            mag = torch.exp(mag * math.log(self.size))
            phase = phase * math.pi
            # Extend time to account for samples that were clipped in FFT
            new_mag = torch.zeros(
                self.audio_channels, self.size + 1, self.fft_out)
            new_phase = torch.zeros(
                self.audio_channels, self.size + 1, self.fft_out)
            new_mag[:, :self.size, :self.size] = mag
            new_phase[:, :self.size, :self.size] = phase
            mag = new_mag
            phase = new_phase
            # Permute dimensions
            mag = mag.permute(0, 2, 1)
            phase = phase.permute(0, 2, 1)
            # Combine magnitude and phase
            sine = mag * torch.sin(phase)
            cosine = mag * torch.cos(phase)
            audio = torch.stack([cosine, sine]).permute(1, 2, 3, 0);
            # Perform inverse FFT
            audio = torchaudio.functional.istft(
                audio, self.fft_size, self.hop_size, center=False,
                length = 2 * self.fft_out * self.size)
        else:
            # Perform FFT
            audio = torch.stft(
                audio, self.fft_size, self.hop_size, center=False)
            # Get magnitude and phase
            mag, phase = torchaudio.functional.magphase(audio)
            # Permute dimensions
            mag = mag.permute(0, 2, 1)
            phase = phase.permute(0, 2, 1)
            # Clip length and Nyquist frequency
            mag = mag[:, :self.size, :self.size]
            phase = phase[:, :self.size, :self.size]
            # Scale
            mag = torch.log(mag) / math.log(self.size)
            mag = torch.clamp(mag, -1, 1)
            phase = phase / math.pi
            # Combine
            audio = torch.cat([mag, phase])
            # Convert frequency axis to mel scale
            if self.mel_scale:
                interpolate = interp1d(mel_y, audio, fill_value='extrapolate')
                audio = torch.tensor(interpolate(new_y), dtype=torch.float32)
            # Flip y
            audio = torch.flip(audio, [2])

        return audio

class AudioTransform:
    def __init__(self, size, audio_channels):
        self.size = size
        self.audio_channels = audio_channels
        self.channels = audio_channels

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

    def __init__(self, size=256, audio_channels=2, spectrogram=False):
        self.length = Audio.length(size, spectrogram)
        t_func = AudioTransformSpectrogram if spectrogram else AudioTransform
        self.transform = t_func(size, audio_channels)
        self.channels = self.transform.channels

    # Get length in seconds
    def length(size, spectrogram):
        return 3 * (size / 256) ** 2 * (1 if spectrogram else 0.5)

class AudioReader(Audio):
    def __init__(self, size=256, audio_channels=2, spectrogram=False, offset=0):
        super(AudioReader, self).__init__(size, audio_channels, spectrogram)
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
        return audio

class AudioWriter(Audio):
    output_rate = 44100

    def __init__(self, size=256, audio_channels=2, spectrogram=False):
        super(AudioWriter, self).__init__(size, audio_channels, spectrogram)
        self.resample = torchaudio.transforms.Resample(
            Audio.rate, AudioWriter.output_rate)

    def __call__(self, path, audio):
        audio = self.transform(audio, reverse=True)
        # This takes a long time
        audio = self.resample(audio)
        audio = audio / (1e-9 + torch.max(audio.min().abs(), audio.max().abs()))
        torchaudio.save(path, audio, AudioWriter.output_rate)
