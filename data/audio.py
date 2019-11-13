import torch
import torchaudio

class AudioTransform:
    def __init__(self, size, audio_channels):
        self.size = size
        # This is not an intuitive FFT size, but we get good dimensions from it
        self.fft_size = 2 * size - 1
        self.audio_channels = audio_channels

    def __call__(self, audio, reverse=False):
        if reverse:
            # Unscale
            audio = audio * self.fft_size
            # Separate audio channels with real/imag
            audio = audio.contiguous().view(
                self.audio_channels, 2, self.size, self.size)
            # Permute dimensions
            audio = audio.permute(0, 2, 3, 1)
            # Extend time to account for samples that were clipped in FFT
            new_audio = torch.zeros(
                self.audio_channels, self.size, self.size + 1, 2)
            new_audio[:, :, :self.size, :] = audio
            # Perform inverse FFT
            audio = torchaudio.functional.istft(
                new_audio, self.fft_size, self.fft_size, center=False,
                length = 2 * self.size * self.size)
        else:
            # Perform FFT
            audio = torch.stft(
                audio, self.fft_size, self.fft_size, center=False)
            # Permute dimensions
            audio = audio.permute(0, 3, 1, 2)
            # Combine audio channels with real/imag
            audio = audio.contiguous().view(
                self.audio_channels * 2, self.size, self.size)
            # Scale
            audio = audio / self.fft_size
        return audio

class Audio:
    rate = 32768
    full_length = 30

    def __init__(self, size=256, audio_channels=2):
        self.length = Audio.length(size)
        self.transform = AudioTransform(size, audio_channels)
        self.channels = audio_channels * 2

    # Get length in seconds
    def length(size):
        return int(4 * (size / 256) ** 2)

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
