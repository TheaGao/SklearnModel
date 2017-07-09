from sklearn.externals import joblib
from preprocessData import *
from voteModels import voteIt

predict_song_label = []
true_song_label = []


def batch_singing_voice_detection(dataset_dir='../Data/wav/test/'):
    for root, dirs, filenames in os.walk(dataset_dir):
        for audioFile in filenames:
            audio_path = os.path.join(root, audioFile)
            if '.wav' in audio_path:
                pre_seg = []

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
                    for mfcc_item in mfcc:
                        segment_mfcc.append(mfcc_item)
                models = ['Models/dt0.58.pkl', 'Models/NB0.59.pkl', 'Models/NC0.57.pkl',
                          'Models/NNP0.61.pkl', 'Models/sgd0.54.pkl']
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

    print get_accuracy(predict_song_label, true_song_label)

    return 0

batch_singing_voice_detection()