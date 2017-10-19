import os
from collections import Counter
import random
import yaml

base = yaml.load(open('../CONFIG.yaml'))['base_dir']
spec_pkl_dir = base + 'extracted/specs/mel/'


files = [xx for xx in os.listdir(spec_pkl_dir) if xx.endswith('.pkl')]
file_sites = [xx.split('_')[0] for xx in files]
print len(files)
print len(set(file_sites))

site_counts = Counter(file_sites)
print site_counts

num_folds = 6

for seed in range(1000):
    print "Seed is ", seed
    sites = sorted(list(set(file_sites)))
    random.seed(seed)
    random.shuffle(sites)

    fold_size = len(sites) / num_folds
    file_fold_size = len(files) / num_folds

    # manually getting the 3 folds
    folds = []
    folds.append(sites[:fold_size])
    folds.append(sites[fold_size:2*fold_size])
    folds.append(sites[2*fold_size:3*fold_size])
    folds.append(sites[3*fold_size:4*fold_size])
    folds.append(sites[4*fold_size:5*fold_size])
    folds.append(sites[5*fold_size:])

    wav_folds = []

    passed = True

    for fold in folds:
        wav_fold_list = [xx.split('-sceneRect.csv')[0]
                         for xx in files
                         if xx.split('_')[0] in fold]
        wav_folds.append(wav_fold_list)

        num_files = sum([site_counts[xx] for xx in fold])
        print num_files
        if num_files < 6:
            passed = False
    if passed: break


# saving the folds to disk
savedir = base + 'splits/'

print "Code commented out to prevent accidently overwriting"
# for idx, (fold, wav_fold) in enumerate(zip(folds, wav_folds)):
#     savepath = savedir + 'fold_sites_%d.yaml' % idx
#     yaml.dump(fold, open(savepath, 'w'), default_flow_style=False)

#     savepath = savedir + 'folds_%d.yaml' % idx
#     yaml.dump(wav_fold, open(savepath, 'w'), default_flow_style=False)


print "Code commented out to prevent accidently overwriting"
# savepath = savedir + 'fold_sites_6.yaml'
# yaml.dump(folds, open(savepath, 'w'), default_flow_style=False)
#
# savepath = savedir + 'folds_6.yaml'
# yaml.dump(wav_folds, open(savepath, 'w'), default_flow_style=False)
