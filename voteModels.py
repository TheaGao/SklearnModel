from sklearn.externals import joblib
from preprocessData import getDataXY, get_accuracy

trainX, trainY, testX, testY, validX, validY = getDataXY()

print type(trainY)

models = ['Models/dt0.58.pkl', 'Models/NB0.59.pkl', 'Models/NC0.57.pkl',
          'Models/NNP0.61.pkl', 'Models/sgd0.54.pkl']

all_pre = []
for model in models:
    print model
    clf = joblib.load(model)
    predicY = clf.predict(validX)
    all_pre.append(predicY)


def voteIt(allResults):
    votRes = []
    for item in range(len(allResults[0])):
        all_item_same_set = []
        for clf_num in range(len(allResults)):
            all_item_same_set.append(allResults[clf_num][item])
            if all_item_same_set.count('sing') > all_item_same_set.count('nosing'):
                votRes.append('sing')
            else:
                votRes.append('nosing')
    return votRes


print get_accuracy(voteIt(all_pre), testY)











