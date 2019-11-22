# MARIO

**MARIO** is a research project exploring the relationships between music and
art through deep learning.

## Setup

Before training, you need to install dependencies and download the data:
```
pip install -r requirements.txt
python download.py
```

## Training

To train your model using the default hyperparameters, run the following:
```
python train.py
```

Please consult the help documentation for more information:
```
python train.py --help
```

## Evaluation

To generate image or audio from a trained model, run the following:

```
python generate.py --checkpoint /PATH/TO/MODEL/CHECKPOINT --image /PATH/TO/IMAGE
python generate.py --checkpoint /PATH/TO/MODEL/CHECKPOINT --audio /PATH/TO/AUDIO
```

Please consult the help documentation for more information:
```
python generate.py --help
```

## Development

1. Clone this repository.
    ```
    git clone https://github.com/nspotrepka/mario.git
    cd mario
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
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
