# Data

**UPDATE SOON!**

## GAPED

### Setup

Download the [GAPED](https://www.unige.ch/cisa/index.php/download_file/view/288/296/)
dataset and copy the `GAPED` folder into this directory.

### Description

Data | Dimension | Description
--- | --- | ---
Images | `(730, 3, 128, 128)` <br> `(730, 3, 256, 256)` <br> `(730, 3, 512, 512)` | 128 x 128, 3 channels <br> 256 x 256, 3 channels <br> 512 x 512, 3 channels
Emotion | `(730, 4)` | arousal, valence, arousal SD, valence SD

Image is resized and padded to square of `size` in `{128, 256, 512}`.

Arousal and valence values are scaled to `[-1, 1]`.

## PMEmo

### Setup

Download the [PMEmo](https://drive.google.com/file/d/1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ/view)
dataset and copy the `PMEmo2019` folder into this directory.

### Description

Data | Dimension | Description
--- | --- | ---
Audio | `(767, 4, 128, 128)` <br> `(767, 4, 256, 256)` <br> `(767, 4, 512, 512)` | 1 second, 2 channels, 32768 Hz <br> 4 seconds, 2 channels, 32768 Hz <br> 16 seconds, 2 channels, 32768 Hz
Emotion | `(767, 4)` | arousal, valence, arousal SD, valence SD

Audio is converted to a sample rate of 32768 Hz, padded or clipped to `length` in seconds, and processed using a STFT with `fft_size = 2 * size - 1`.

Arousal and valence values are scaled to `[-1, 1]`.
