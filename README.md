# CityNet - a neural network for urban sounds

CityNet is a machine-learned system for estimating the level of **biotic** and **anthropogenic** sound at each moment in time in an audio file.

The system has been trained and validated on human-labelled audio files captured from green spaces around London.

CityNet comprises a neural network classifier, which operates on audio spectrograms to produce a measure of biotic or anthropogenic activity level.


# Requirements

The system has been tested with `python 2.7`, using the `anaconda` distribution (see [here](https://www.anaconda.com/download/) for download details)

Then run the following commands to get suitable versions of the required libraries:


    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/v0.1/requirements.txt
    pip install Lasagne==0.1
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
    pip install nolearn
    pip install librosa
    pip install easydict
    pip install tqdm
    pip install git+git://github.com/mdfirman/ml_helpers.git@master


# How to classify a new audio file with CityNet

- Run `python demo.py` to classify an example audio file. 
- Predictions should be saved in the folder `demo`. 
- Your newly-created file `demo/prediction.pdf` should look identical to the provided file `demo/reference_prediction.pdf`.

Editing `demo.py` should allow you to classify your own audio files.

