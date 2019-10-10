# Data

## PMEmo

### Setup

1. Download the [PMEmo](http://pmemo.hellohui.cn/) dataset and copy the
  `PMEmo2019` folder into this directory.
2. Convert MP3 files to WAV.
    ```
    python scripts/convert_audio.py
    ```

### Description

Data | Function | Dimension | Description
--- | --- | --- | ---
Static Emotion | `static()` | (767, 2) | arousal/valence
Static Emotion SD | `static_std()` | (767, 2) | arousal/valence standard deviation
Dynamic Emotion | `dynamic()` | (767, 30, 2) | half-second arousal/valence
Dynamic Emotion SD | `dynamic_std()` | (767, 30, 2) | half-second arousal/valence standard deviation
Audio Samples | `audio()` | (767, 1323000, 2) | 30 seconds, 44.1 kHz, 2 channels

Arousal and valence values are between 0 and 1.

There is no dynamic data for the first 15 seconds of each song.
The 30 data points of dynamic emotion correspond to half-seconds
between the 15 second mark and 30 second mark.
