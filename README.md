# Mario

**Mario** is a research project exploring the relationships between art and
music through deep learning.

## Setup

Before training, you need to install dependencies and download the data:
```
pip install -r requirements.txt
python download.py
```

## Training

To train your model using the default hyperparameters, run the following:
```
python train.py --prefix some_prefix_for_checkpoint
```

Please consult the help documentation for more information:
```
python train.py --help
```

## Evaluation

To generate image or music from a trained model, run the following:

```
python evaluate.py --checkpoint /PATH/TO/MODEL/CHECKPOINT --data_a /PATH/TO/IMAGE
python evaluate.py --checkpoint /PATH/TO/MODEL/CHECKPOINT --data_b /PATH/TO/AUDIO
```

Please consult the help documentation for more information:
```
python evaluate.py --help
```

## License

Apache License 2.0
