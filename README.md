# Mario

**Mario** is a research project exploring the relationships between music and
art through emotion.

## Setup

1. Download [PMEmo](http://pmemo.hellohui.cn/) to the `data` folder.
2. Convert MP3 files to WAV.
```
python scripts/convert_audio.py
```

## Demo

```
python scripts/demo_audio.py
```

## Development

1. Clone this repository.
2. Install the following dependencies:
```
pip install numpy
pip install tensorflow
pip install pydub
```
3. Write some code.
4. Pull any changes that may have happened while you were coding.
```
git pull
```
If there are conflicts, merge your changes.

5. Double check what you have modified.
```
git status
```
Don't add unnecessary files or folders. If you see a file you don't want to
commit, add it to the `.gitignore`.

6. Add your files, commit, and push.
```
git add --all
git commit -m "Change learning rate for image GAN"
git push
```

## License

Apache License 2.0
