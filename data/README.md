# Data

## PMEmo

### Setup

1. Download the [PMEmo](https://drive.google.com/drive/folders/1qDk6hZDGVlVXgckjLq9LvXLZ9EgK9gw0) dataset and copy the
  `PMEmo2019` folder into this directory.
2. Convert MP3 files to WAV.
    ```
    python scripts/convert_audio.py
    ```

### Description

Data | Function | Dimension | Description
--- | --- | --- | ---
Static Emotion | `static()` | (767, 4) | arousal, valence, arousal SD, valence SD
Dynamic Emotion | `dynamic()` | (767, 30, 4) | 30 samples of arousal, valence, arousal SD, valence SD
Audio Samples | `audio()` | (767, 1323000, 2) | 30 seconds, 44.1 kHz, 2 channels

Values for arousal and valence are between 0 and 1.

There is no dynamic data for the first 15 seconds of each song.
The 30 data points of dynamic emotion correspond to half-seconds
between the 15 second mark and 30 second mark.

## GAPED

### Setup

1. Download the [GAPED](https://www.unige.ch/cisa/index.php/download_file/view/288/296/) dataset and copy the
  `GAPED` folder into this directory.

### Description

Data | Function | Dimension | Description
--- | --- | --- | ---
Emotion | `emotion()` | (730, 4) | arousal, valence, arousal SD, valence SD
Images | `images()` | (730, 480, 640, 3) | 640 x 480, 3 channels

Values for arousal and valence are between 0 and 1.

Note that the width and height dimensions are flipped.
