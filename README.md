# Deep Modules For Abstract Reasoning Corpus
Contact: Logan Bruns <logan@gedanken.org>

### Abstract from project paper

### Example predictions on test dataset

### Repository layout

#### Source files

_Makefile_: top level makefile to run training, start tensorboard and start notebook server

_deep\_module\_arc.ipynb_: top level notebook

_environment.yml_: conda environment yaml for creating python virtual environment

#### Directories

_data/_ directory to download dataset to and store transformed forms

_experiments/_ directory to hold experiment checkpoints and tensorboard logs

### Environment preparation steps

```
$ conda env create --name deep-module-arc --file environment.yml
$ conda activate deep-module-arc
```

### Data preparation steps

#### Download dataset from kaggle
Download ARC kaggle dataset zip from kaggle:

https://www.kaggle.com/c/abstraction-and-reasoning-challenge/data

#### Unzip into input directory

```
$ cd data
$ unzip /path/to/abstraction-and-reasoning-challenge.zip 
```

#### Convert from JSON to TFRECORD

```
$ python json_to_tfrecord.py
```

### Training steps

#### Start tensorboard

```
$ make tensorboard
```

#### Start training

```
$ make train
```

#### Monitor training and results in tensorboard

Go to http://localhost:6006/ or http://hostname:6006/ and click images
tab to see examples of the training context images, predictions, and
ground truth.
