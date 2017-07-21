import os
import sys
import numpy as np
from tqdm import tqdm
import time
sys.path.append('..')
sys.path.append('../..')
import classifier
import utils
from easydict import EasyDict as edict
import yaml

search_locations = utils.get_search_locations()

classname = 'anthrop'
extra = {'biotic': '', 'anthrop': '_anthrop'}[classname]
best_member = {'biotic': 0, 'anthrop': 3}[classname]
base_savedir = '/media/michael/SeagateData/alison_data/predictions_%s/' % classname
spec_basedir = '/media/michael/SeagateData/alison_data/spectrograms/'


models_dir = '/media/michael/Engage/data/audio/alison_data/golden_set/predictions/ensemble_train%s/%d/%s/' % (
    extra, best_member, classname)
weights_path = os.path.join(models_dir, 'results/weights_99.pkl')
options_path = os.path.join(models_dir, 'network_opts.yaml')

opts = edict(yaml.load(open(options_path)))
model = classifier.Classifier(opts=opts, weights_path=weights_path)


def proc_file(paths):
    spec_loadpath, savepath = paths

    try:
        model.spec = np.load(spec_loadpath)
        # model.spec
        preds = model.classify()
        np.save(savepath, preds.astype(np.float16))
    #
    # try:
    # except ParameterError:
    #     with open('./failure_log.txt', 'w+') as f:
    #         f.write(loadpath + "\n")
    #     return
    except:
        with open('./classification_failure_log.txt', 'w+') as f:
            f.write(spec_loadpath + "\n")
        return


from multiprocessing import Pool


def batch_process_files(loaddir, fnames, savedir):
    print loaddir, savedir, fnames[0:5]

    # Inner loop
    paths = []
    for fname in fnames:
        if not fname.endswith('.wav'):
            continue

        spec_loadpath = loaddir + '/' + fname.replace('.wav', '.npy')
        pred_savepath = savedir + '/' + fname.replace('.wav', '.npy')
        if not os.path.exists(pred_savepath):
            paths.append([spec_loadpath, pred_savepath])

    #p = Pool(4, maxtasksperchild=10)
    map(proc_file, paths)


# Loop over a load of hdds
all_fnames = []
endnow = 0
for hd_idx, search_location in search_locations:

    for root, dirnames, filenames in os.walk(search_location):

        filtered_fnames = [fname for fname in filenames
                          if 'BAT+' not in root and 'Random' not in root and fname.endswith('.wav')]

        if len(filtered_fnames):
            all_fnames.extend(filtered_fnames)

            spec_loaddir = spec_basedir + ('/%d/' % hd_idx) + root.split('Fieldwork_Data')[1]
            savedir = base_savedir + ('/%d/' % hd_idx) + root.split('Fieldwork_Data')[1]

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            batch_process_files(spec_loaddir, filtered_fnames, savedir)
