# Mario

**Mario** is a research project exploring the relationships between music and
art through emotion.

## Setup

See `data/README.md` for instructions on how to download the datasets.

## Demo

To confirm that PMEmo is properly installed, run the following demo:
```
python scripts/demo_pmemo.py
```

To confirm that GAPED is properly installed, run the following demo:
```
python scripts/demo_gaped.py
```

## Development

1. Clone this repository.
    ```
    git clone https://github.com/nspotrepka/mario.git
    cd mario
    ```
2. Install the following dependencies:
    ```
    pip install numpy
    pip install tensorflow
    pip install pydub
    ```
3. Write some code!
4. Pull any changes that may have been pushed while you were coding.
    ```
    git pull
    ```
5. Resolve all merge conflicts. Consult Stack Overflow for help, if needed.
6. Double check the modifications you have made.
    ```
    git status
    ```
    Don't add unnecessary files or folders. If you see something that shouldn't
    be committed, add it to the `.gitignore`.
7. Add your files, commit, and push.
    ```
    git add --all
    git commit -m "Change learning rate for image GAN"
    git push
    ```

## License

Apache License 2.0
