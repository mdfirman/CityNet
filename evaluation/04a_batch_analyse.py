import subprocess as sp

run_types = [
    'warblr_challenge_baseline',
    'ensemble_train',
    'oneSec_BI_baseline',
    'oneSec_NDSI_baseline',
    'oneSec_ACI_baseline',
    'oneSec_ADI_baseline',
    ]

import socket
if socket.gethostname() == 'biryani':
    # to force python 2 on this machine
    python = '/home/michael/anaconda/bin/python'

for run_type in run_types:
    print "\n\n\n--- ", run_type, "---"
    sp.call([python, '04_analyse_results.py', run_type, 'biotic'])

for run_type in ['ensemble_train_anthrop', 'oneSec_NDSI_baseline']:
    print "\n\n\n--- ", run_type, "---"
    sp.call([python, '04_analyse_results.py', run_type, 'anthrop'])
