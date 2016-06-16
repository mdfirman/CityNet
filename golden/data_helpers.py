import pandas as pd
import librosa

labels_dir = '/media/michael/Seagate/engage/alison_data/golden_set/labels/Golden/'
wav_dir = '/media/michael/Seagate/engage/alison_data/golden_set/wavs/'


human_noises = set(['mix traffic', 'braking', 'voices', 'electrical',
                   'anthropogenic unknown', 'airplane', 'beep',
                   'metal', 'bus emitting', 'footsteps', 'mower', 'whistle',
                  'siren', 'coughing', 'music', 'horn', 'startthecar', 'bells',
                    'applause', 'dog bark', 'road traffic', 'braking vehicle (road or rail)',
                   'human voices', 'mechanical', 'vehicle horn (road or rail)',
                   'air traffic', 'vehicle alarm', 'human voice', 'machinery',
                   'church bell', 'breaking vehicle', 'deck lowering', 'car horn',
                   'rail traffic', 'alarm', 'vehicle horn',
                   'building ventilation system', 'car alarm', 'rock', 'church bells',
                   'train horn', 'mobile phone', 'train station announcement', 'hammering',
                   'door opening', 'dog barking', 'vehicle breaking', 'cat',
                   'glass into bins', 'barking dog', 'television', 'sweeping broom',
                   'ball bouncing', 'bat hitting ball', 'laughing', 'clapping', 'camera',
                   'train doors (beeping)', 'lawnmower'])

animal_noises = set(['bird', 'wing beats', 'bat', 'fox',
                     'grey squirrel', 'invertebrate', 'insect', 'animal',
                     'wings beating',  'russling leaves (animal)',
                     'amphibian', 'squirrel', 'russling vegetation (animal)'])

other = set(['rain', 'unknown sound', 'electrical disturbance', 'vegetation',
            'wind', 'unknown', 'metalic sound', 'dripping water', 'shower',
            'metalic', 'rubbish bag','water dripping', 'water splashing',
            'rainfall on vegetation'])


def load_annotations(fname, labels_dir=labels_dir, wav_dir=wav_dir):

    pd_annots = pd.read_csv(labels_dir + fname)

    # where we'll temp store the blank snippets for this file
    blank_snippets = []

    # load file and convert to spectrogram
    wav, sample_rate = librosa.load(wav_dir + fname.replace('-sceneRect.csv', '.wav'), 22050)

    # create label vector...
    biotic = 0 * wav
    anthropogenic = 0 * wav

    # loop over each annotation...
    for _, annot in pd_annots.iterrows():

        # fill in the label vector
        start_point = int(annot['LabelStartTime_Seconds'] * sample_rate)
        end_point =  int(annot['LabelEndTime_Seconds'] * sample_rate)

        if annot['Label'].lower() in human_noises:
            anthropogenic[start_point:end_point] = 1
        elif annot['Label'].lower() in animal_noises:
            biotic[start_point:end_point] = 1
        elif annot['Label'].lower() in other:
            pass
        else:
            raise Exception("Unknown label ", annot['Label'])

    return {'anthrop': anthropogenic, 'biotic': biotic}, wav, sample_rate
