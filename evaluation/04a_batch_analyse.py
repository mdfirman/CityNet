import subprocess as sp

run_types = [
    # 'mel32_train_large_hard_bootstrap',
    'warblr_challenge_baseline',
    # 'mel32_large_test_golden_fullsplit',
    # 'timestep_BI_baseline',
    'oneSec_BI_baseline',
    # 'timestep_NDSI_baseline',
    'oneSec_NDSI_baseline',
    # 'timestep_ACI_noRound_baseline',
    'oneSec_ACI_baseline',
    'oneSec_ADI_baseline',
    # 'timestep_ACI_Round_baseline',
    'ensemble_train'
    ]

for run_type in run_types:
    print "--- ", run_type, "---"
    sp.call(['python', '04_analyse_results.py', run_type, 'biotic'])

for run_type in ['ensemble_train_2', 'oneSec_NDSI_baseline']:
    sp.call(['python', '04_analyse_results.py', run_type, 'anthrop'])
