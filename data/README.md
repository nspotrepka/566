# Data

## FMA

### Setup

```
cd data

curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -

unzip fma_metadata.zip
unzip fma_small.zip
```

If `unzip` does not work, try using 7zip:
```
brew install p7zip

7z x fma_metadata.zip
7z x fma_small.zip
```

### Description

Data | Function | Dimension | Description
--- | --- | --- | ---
Audio Samples | (767, 2, 1323000) | 2 channels, 30 seconds, 44.1 kHz
Genre | (8000, 8) | one-hot encoding of 8 genres

Here are the 8 genres:
```
['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
```

## GAPED

### Setup

1. Download the [GAPED](https://www.unige.ch/cisa/index.php/download_file/view/288/296/) dataset and copy the
  `GAPED` folder into this directory.

### Description

Data | Dimension | Description
--- | --- | ---
Images | (730, 640, 480, 3) | 640 x 480, 3 channels
Emotion | (730, 4) | arousal, valence, arousal SD, valence SD

Arousal and valence values are between 0 and 1.

## PMEmo

### Setup

1. Download the [PMEmo](https://drive.google.com/drive/folders/1qDk6hZDGVlVXgckjLq9LvXLZ9EgK9gw0) dataset and copy the
  `PMEmo2019` folder into this directory.
2. Convert MP3 files to WAV.
    ```
    python scripts/convert_audio.py
    ```

### Description

Data | Dimension | Description
--- | --- | ---
Audio Samples | (767, 2, 1323000) | 2 channels, 30 seconds, 44.1 kHz
Static Emotion | (767, 4) | arousal, valence, arousal SD, valence SD

Arousal and valence values are between 0 and 1.
