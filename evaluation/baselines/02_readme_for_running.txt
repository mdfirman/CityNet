# Creating spectrograms and applying classifier

Go to: 
michael@biryani:~/projects/others_projects/bird_audio_detection_challenge_2017

Make sure config.inc is pointing to the right places

In run.sh, comment out lines 286-290 except:
stage1_prepare ${cmdargs}
stage1_predict ${cmdargs}

Do
./run.sh
