import os
import sys
import yaml
import pickle
import numpy as np
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
        options = yaml.full_load(f)

    model_path = 'tf_models/%s/weights_99.pkl-1' % classifier_type

    predictor = TFClassifier(options, model_path)


    for count, filename in enumerate(filenames):

        if not filename.lower().endswith('.wav'):
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


######################

plots_savedir = "plots"
print("-> Saving prediction plots to {}".format(plots_savedir))

os.makedirs(plots_savedir, exist_ok=True)
cols = {'anthrop': 'b', 'biotic': 'g'}

for fname, this_fname_preds in preds.items():
    plt.figure(figsize=(15, 5))

    for key, val in this_fname_preds.items():
        len_in_s = val.shape[0] * HOP_LENGTH / predictor.sample_rate
        print(len_in_s)

        x = np.linspace(0, len_in_s, val.shape[0])
        plt.plot(x, val, cols[key], label=key)

        plt.xlabel('Time (s)')
        plt.ylabel('Activity level')

    plt.ylim(0, 1.2)
    plt.xlim(0, 60)
    plt.legend()
    plt.title(fname)

    save_fname = os.path.splitext(fname)[0]
    plt.savefig(os.path.join(plots_savedir, save_fname))
    plt.close()

######################

print("-> ...Done")
