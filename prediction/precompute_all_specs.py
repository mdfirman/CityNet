
# coding: utf-8

# In[ ]:

# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')
# get_ipython().magic(u'matplotlib inline')
import os
import sys
import numpy as np


import cPickle as pickle
import yaml
import collections
from tqdm import tqdm
import time
from scipy.ndimage.interpolation import zoom


# # Find chronological ordering of the HDDs

# In[ ]:

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

# In[ ]:

search_locations = [
    (0, ordered_hdds[0] + 'Fieldwork_Data/2013/'),
    (0, ordered_hdds[0] + 'Fieldwork_Data/2014/'),
    (1, ordered_hdds[1] + 'Fieldwork_Data/2014/'),
    (1, ordered_hdds[1] + 'Fieldwork_Data/2015/'),
    (2, ordered_hdds[2] + 'Fieldwork_Data/2015/')
]


# In[ ]:

base_savedir = '/media/michael/SeagateData/alison_data/spectrograms/'


# In[ ]:

import classifier
model = classifier.Classifier()


# In[ ]:

def compute_spec(loadpath, savepath):
    model.load_wav(loadpath, loadmethod='wavfile')
    model.compute_spec()
    np.save(savepath, model.spec.astype(np.float16))


def batch_process_files(loaddir, fnames, savedir):

    # Inner loop
    for fname in fnames:
        loadpath = loaddir + fname
        savepath = savedir + '/' + fname.replace('.wav', '.npy')
        compute_spec(loadpath, savepath)


# Loop over a load of hdds
all_fnames = []
endnow = 0
for hd_idx, search_location in search_locations:

    for root, dirnames, filenames in os.walk(search_location):

        filtered_fnames = [fname for fname in filenames
                          if 'BAT+' not in root and 'Random' not in root and fname.endswith('.wav')]

        if len(filtered_fnames):
            endnow = 1
            break
            all_fnames.extend(filtered_fnames)

            savedir = base_savedir + ('/%d/' % hd_idx) + root.split('Fieldwork_Data')[1]

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            tic = time.time()
            batch_process_files(root + '/', filtered_fnames[:6], savedir)
            print time.time() - tic
        if endnow:
            break
    if endnow:
        break


print len(all_fnames)


# In[ ]:

savedir = base_savedir + ('/%d/' % hd_idx) + root.split('Fieldwork_Data')[1]

if not os.path.exists(savedir):
    os.makedirs(savedir)

tic = time.time()
batch_process_files(root + '/', filtered_fnames[:10], savedir)
print time.time() - tic





def proc_file(paths):
    loadpath, savepath = paths
    model.load_wav(loadpath, loadmethod='wavfile')
    model.compute_spec()
    np.save(savepath, model.spec.astype(np.float16))


from multiprocessing import Pool

def batch_process_files(loaddir, fnames, savedir):

    # Inner loop
    paths = []
    for fname in fnames:
        loadpath = loaddir + fname
        savepath = savedir + fname.replace('.wav', '.npy')
        paths.append([loadpath, savepath])

    p = Pool(4)
    p.map(proc_file, paths)
    # for path in paths:
    #     proc_file(path)


tic = time.time()
savedir = base_savedir + '/pooled/' + ('/%d/' % hd_idx) + root.split('Fieldwork_Data')[1]

if not os.path.exists(savedir):
    os.makedirs(savedir)

batch_process_files(root + '/', filtered_fnames[:10], savedir)
print (time.time() - tic)
print (time.time() - tic) / 10 * 32223
