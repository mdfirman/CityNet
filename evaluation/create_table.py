import os
import yaml

base = yaml.load(open('../CONFIG.yaml'))['base_dir'] + '/predictions/'
classnames = ['biotic', 'anthrop']

metrics = ['recall_at_095_prec', 'avg_precision']

csv = []
for classname in classnames:

    csv.append(classname + ',,' + '\n')

    if classname == 'biotic':
        runs = ['ensemble_train',  'warblr_challenge_baseline',  'oneSec_ACI_baseline', 'oneSec_ADI_baseline', 'oneSec_BI_baseline', 'oneSec_NDSI_baseline']
    else:
        runs = ['ensemble_train_anthrop', 'oneSec_NDSI_baseline']

    scores = {}
    for run in runs:
        scores[run]  = yaml.load(open(base + run + '/' + classname + '/analysis/scores.yaml'))

    for run in runs:
        csv.append(run)
        for metric in metrics:
            csv.append(", %0.3f" % scores[run][metric],)
        csv.append("\n")

print "".join(csv)

with open('../plots/tables.csv', 'w') as f:
    f.write("".join(csv))
