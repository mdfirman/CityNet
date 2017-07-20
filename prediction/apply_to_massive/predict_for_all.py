import os
import sys
import numpy as np
from tqdm import tqdm
import time


# Find chronological ordering of the HDDs
unordered_hdds = ['/media/michael/Elements/', '/media/michael/Elements1/', '/media/michael/Elements2/']
ordered_hdds = [None, None, None]

for hdd in unordered_hdds:
    if os.path.exists(hdd + 'Fieldwork_Data/2013/') and os.path.exists(hdd + 'Fieldwork_Data/2014/'):
        ordered_hdds[0] = hdd
    elif os.path.exists(hdd + 'Fieldwork_Data/2014/') and os.path.exists(hdd + 'Fieldwork_Data/2015/'):
        ordered_hdds[1] = hdd
    elif os.path.exists(hdd + 'Fieldwork_Data/2015') and not os.path.exists(hdd + 'Fieldwork_Data/2014'):
        ordered_hdds[2] = hdd

print ordered_hdds


# # Define all search locations
search_locations = [
    (0, ordered_hdds[0] + 'Fieldwork_Data/2013/'),
    (0, ordered_hdds[0] + 'Fieldwork_Data/2014/'),
    (1, ordered_hdds[1] + 'Fieldwork_Data/2014/'),
    (1, ordered_hdds[1] + 'Fieldwork_Data/2015/'),
    (2, ordered_hdds[2] + 'Fieldwork_Data/2015/')
]

base_savedir = '/media/michael/SeagateData/alison_data/predictions/'
spec_basedir = '/media/michael/SeagateData/alison_data/spectrograms/'

sys.path.append('..')
sys.path.append('../..')
import classifier
model = classifier.Classifier()

print "TOdo - just do per second"

def proc_file(paths):
    print "TOdo - just do per second"
    spec_loadpath, savepath = paths

    # try:
    model.spec = np.load(spec_loadpath)
    preds = model.classify()
    np.save(savepath, preds.astype(np.float16))
    #
    # try:
    # except ParameterError:
    #     with open('./failure_log.txt', 'w+') as f:
    #         f.write(loadpath + "\n")
    #     return
    # except ValueError:
    #     with open('./classification_failure_log.txt', 'w+') as f:
    #         f.write(loadpath + "\n")
    #     return



from multiprocessing import Pool


def batch_process_files(loaddir, fnames, savedir):
    print loaddir, savedir, fnames[0:5]


    # Inner loop
    paths = []
    for fname in fnames:
        if not fname.endswith('.wav'):
            continue

        loadpath = loaddir + fname
        spec_loadpath = savedir + '/' + fname.replace('.wav', '.npy')
        if not os.path.exists(savepath):
            paths.append([loadpath, savepath])

    #p = Pool(4, maxtasksperchild=10)
    map(proc_file, paths)
    sds


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
