import os
import sys
import yaml
import zipfile
import numpy as np
import pickle
from six.moves import urllib
import matplotlib.pyplot as plt
import sys
sys.path.append('lib')

from prediction.tf_classifier import TFClassifier, HOP_LENGTH


# getting the list of all the files to predict for:
where_to_search = sys.argv[1]
filenames = os.listdir(where_to_search)

assert os.path.exists("tf_models/biotic/network_opts.yaml"), \
    "Pretrained model not found - run demo.py first to download model"


############################################################

print("->  Making predictions for biotic and anthropogenic separately")

preds = {}

for classifier_type in ['biotic', 'anthrop']:

    with open('tf_models/%s/network_opts.yaml' % classifier_type) as f:
        options = yaml.load(f)

    model_path = 'tf_models/%s/weights_99.pkl-1' % classifier_type

    predictor = TFClassifier(options, model_path)


    for count, filename in enumerate(filenames):

        if not filename.endswith('.wav'):
            print("Skipping {}, not a wav file".format(filename))
            continue

        print("Classifying file {} of {}".format(count, len(filenames)))
        if filename not in preds:
            preds[filename] = {}
        preds[filename][classifier_type] = predictor.classify(
            os.path.join(where_to_search, filename))

print("-> ...Done")

############################

print("-> Saving raw predictions to {}".format("predictions.pkl"))

# save comprehensive data
with open('predictions.pkl', 'wb') as f:
    pickle.dump(preds, f, -1)

# save summaries to csv file

print("-> Saving prediction summaries to {}".format("prediction_summaries.csv"))

with open("prediction_summaries.csv", "w") as f:
    f.write("Filename,Average biotic sound,Average anthropogenic sound\n")
    for fname in preds:
        f.write("{},{:5.3f},{:5.3f}\n".format(
            fname,
            preds[fname]['biotic'].mean(),
            preds[fname]['anthrop'].mean()))


print("-> ...Done")
