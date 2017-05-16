import subprocess as sp

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
    'overlap_ACI_Round_baseline'
    ]

if 0:
    for run_type in run_types:
        print "--- ", run_type, "---"
        sp.call(['python', '04_analyse_results.py', run_type, 'biotic'])

else:
    for run_type in ['mel32_large_test_golden_fullsplit', 'overlap_NDSI_baseline', 'timestep_NDSI_baseline']:
        sp.call(['python', '04_analyse_results.py', run_type, 'anthrop'])
