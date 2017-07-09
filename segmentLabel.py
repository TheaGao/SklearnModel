import os

from baseZhang import wavread, calcMFCC
from sklearn.externals import joblib

from voteModels import voteIt
from preprocessData import get_accuracy


def singing_voice_detection(audio_path):
    predict_song_label = []
    true_song_label = []
    if '/train/' in audio_path:
        lab_path = audio_path.replace('/train/', '/lab/')[:-3] + 'lab'
    elif '/test/' in audio_path:
        lab_path = audio_path.replace('/test/', '/lab/')[:-3] + 'lab'
    elif '/valid/' in audio_path:
        lab_path = audio_path.replace('/valid/', '/lab/')[:-3] + 'lab'
    else:
        lab_path = 'null'

    label_file = open(lab_path, 'r')
    labels = label_file.readlines()
    label_file.close()
    audioData, fs = wavread(audio_path)

    for item_label in labels:
        startTime, endTime, labelY = item_label.split(' ')
        startTime = float(startTime)
        endTime = float(endTime)
        labelY = labelY[:-1]
        true_song_label.append(labelY)
        audio_part_data = audioData[int(startTime * fs):int(endTime * fs)]
        segment_mfcc = []
        mfcc = calcMFCC(audio_part_data, fs)
        for item_mfcc in mfcc:
            segment_mfcc.append(item_mfcc)
        models = ['dt0.58.pkl', 'gnb0.59.pkl', 'mlp0.60.pkl', 'nc0.57.pkl', 'sgd0.57.pkl']
        all_pre = []
        for model in models:
            # print model
            clf = joblib.load(model)
            predictY = clf.predict(segment_mfcc)
            all_pre.append(predictY)
        voteRes = voteIt(all_pre)
        if voteRes.count('sing') > voteRes.count('nosing'):
            segmentLabel = 'sing'
        else:
            segmentLabel = 'nosing'
        predict_song_label.append(segmentLabel)
    return predict_song_label, true_song_label


print singing_voice_detection('../data/pyT6/wav/train/01 - 10min.wav')


def batch_svd(dataset_dir='../data/pyT6/wav/test/'):
    all_pre_seg = []
    all_tru_seg = []
    for root, dirs, filenames in os.walk(dataset_dir):
        for audioFile in filenames:
            audio_path = os.path.join(root, audioFile)
            if '.wav' in audio_path:
                pre, tru = singing_voice_detection(audio_path)
                all_pre_seg.extend(pre)
                all_tru_seg.extend(tru)
    print get_accuracy(all_tru_seg, all_pre_seg)

    return 0


batch_svd()
