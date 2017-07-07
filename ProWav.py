import numpy
from baseZhang import calcMFCC, wavread
from pydub import AudioSegment

lab = '01 - 01 Les Jardins Japonais.lab'
label = open(lab, 'r')
labels = label.readlines()
label.close()
# print labels[0]
startTime, endTime, labelY = labels[0].split(' ')
startTime = float(startTime)
endTime = float(endTime)
labelY = labelY[:-1]
# print labelY

song = AudioSegment.from_file(lab.replace('.lab', '.ogg'))
song.set_channels(1)
song = song.export(lab.replace('.lab', '.wav'), 'wav')
audioData, fs = wavread(lab.replace('.lab', '.wav'))
# print len(audioData)
# print fs
part1 = audioData[int(startTime * fs):int(endTime * fs)]
mfcc = calcMFCC(part1, fs)
print numpy.shape(mfcc)
