# CityNet - a neural network for urban sounds

CityNet is a machine-learned system for estimating the level of **biotic** and **anthropogenic** sound at each moment in time in an audio file.

The system has been trained and validated on human-labelled audio files captured from green spaces around London.

CityNet comprises a neural network classifier, which operates on audio spectrograms to produce a measure of biotic or anthropogenic activity level.

More details of the method are available from the paper:

> **[CityNet - Deep Learning Tools for Urban Ecoacoustic Assessment](https://doi.org/10.1101/248708)**
>
> Alison J Fairbrass, Michael Firman, Carol Williams, Gabriel J Brostow, Helena Titheridge and Kate E Jones
>
> **doi**: https://doi.org/10.1101/248708


An overview of predictions of biotic and anthropogenic activity on recordings of London sounds can be seen at our website [londonsounds.org](http://londonsounds.org).

[![Screenshot of urban sounds website](website/website.png)](http://londonsounds.org)





## Requirements

The system has been tested using the dependencies in `environment.yml`. Our code works with python 3.

You can create an environment with all the dependencies installed using:

```bash
conda env create -f environment.yml -n citynet
conda activate citynet
```

## How to classify a new audio file with CityNet

- Run `python demo.py` to classify an example audio file.
- Predictions should be saved in the folder `demo`.
- Your newly-created file `demo/prediction.pdf` should look identical to the provided file `demo/reference_prediction.pdf`:

## How to classify multiple audio files

You can run CityNet on a folder of audio files with:

```bash
python multi_predict.py path/to/audio/files
```

## Hardware requirements

For training and testing we used a 2GB NVIDIA GPU. The computation requirements for classification are pretty low though, so a GPU should not be required.
