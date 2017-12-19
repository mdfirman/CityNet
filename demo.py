import os
import wget
import yaml
import zipfile
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('lib')

from prediction.classifier import Classifier, HOP_LENGTH

# where to download the pretrained models from
models_dl_path = 'https://www.dropbox.com/s/6mhv7ddfhievx8x/models.zip?dl=1'


#############################################

print "-> Downloading and unzipping pre-trained model..."

if not os.path.exists('models/biotic/network_opts.yaml'):
    
    if not os.path.exists('models/models.zip'):
        wget.download(models_dl_path, 'models')

    with zipfile.ZipFile('models/models.zip', 'r') as zip_ref:
        zip_ref.extractall('models/')

print "-> ...Done"


############################################################

print "->  Making predictions for biotic and anthropogenic separately"

preds = {}

for classifier_type in ['biotic', 'anthrop']:

    with open('models/%s/network_opts.yaml' % classifier_type) as f:
        options = yaml.load(f)

    predictor = Classifier(options, 'models/%s/weights_99.pkl' % classifier_type)
    preds[classifier_type] = predictor.classify('demo/SW154LA-3527_20130705_0909.wav')

print "-> ...Done"


############################

print "-> Saving predictions to disk"

with open('demo/predictions.pkl', 'wb') as f:
    pickle.dump(preds, f, -1)

print "-> ...Done"


######################

print "-> Plotting predictions"

plt.figure(figsize=(15, 5))
cols = {'anthrop': 'b', 'biotic': 'g'}

for key, val in preds.items():
    len_in_s = val.shape[0] * HOP_LENGTH / predictor.sample_rate
    print len_in_s
    
    x = np.linspace(0, len_in_s, val.shape[0])
    plt.plot(x, val[:, 1], cols[key], label=key)

    plt.xlabel('Time (s)')
    plt.ylabel('Activity level')

plt.ylim(0, 1.2)
plt.legend()
plt.savefig('demo/predictions.pdf')

print "-> ...Done"
