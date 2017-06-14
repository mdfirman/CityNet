import yaml

run_types = [
    # 'mel32_train_large_hard_bootstrap',
    'warblr_challenge_baseline',
    'mel32_large_test_golden_fullsplit',
    'timestep_BI_baseline',
    'overlap_BI_baseline',
    'timestep_NDSI_baseline',
    'overlap_NDSI_baseline',
    'timestep_ACI_noRound_baseline',
    'overlap_ACI_noRound_baseline',
    'timestep_ACI_Round_baseline',
    'overlap_ACI_Round_baseline',
    'ensemble_train'
]

classname = 'biotic'

base_dir = '/media/michael/Engage/data/audio/alison_data/golden_set/'

mapp = {}
recall_at_095_prec = {}

for run_type in run_types:

    results_dir = base_dir + 'predictions/%s/%s/analysis/' % (run_type, classname)

    scores = yaml.load(open(results_dir + 'scores.yaml'))

    recall_at_095_prec[run_type] = float(scores['recall_at_095_prec'])
    mapp[run_type] = float(scores['avg_precision'])

def print_table(tab):
    for key, val in tab.iteritems():
        print "%s - %0.3f" % (key.ljust(35), val)

print "Average precision"
print_table(mapp)

print "\nRecall at 0.95"
print_table(recall_at_095_prec)
