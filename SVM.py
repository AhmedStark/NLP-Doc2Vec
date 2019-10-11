import matplotlib.pyplot as plt
from sklearn import svm, datasets
import numpy as np
import pandas as pd

TrainData = pd.read_csv("/home/ahmed/Desktop/python/Assignment2/T1/TrainingD2V.csv").drop(columns=["Unnamed: 0"])
TestData = pd.read_csv("/home/ahmed/Desktop/python/Assignment2/T1/TestingD2V.csv").drop(columns=["Unnamed: 0"])

xTrain = TrainData[['polarity', "body_len", "punctuation%", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]].reset_index(drop=True)
xTest = TestData[['polarity',"body_len","punctuation%", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]].reset_index(drop=True)

train_y = TrainData['label'].reset_index(drop=True).values.tolist()
test_y = TestData['label'].reset_index(drop=True).values.tolist()

def convertClassesToNums(ylist):
    result = []
    for i in ylist:
        if i == 'ham':
            result.append(0)
        else:
            result.append(1)
    return result
train_y = np.array(convertClassesToNums(train_y))
test_y = np.array(convertClassesToNums(test_y))
######################################################################################################################
polarity = xTrain['polarity']
PolMean = sum(polarity)/len(polarity)
stdPol = np.std(polarity)

zStatPol = []

for i in range(len(polarity)):
    zStatPol.append((polarity[i]-PolMean)/stdPol)


def standerdize(df):

    colnames = df.columns
    for i in colnames:
        colMean = sum(df[i])/len(df[i])
        colStd = np.std(df[i])

        for row in range(len(df[i])):
            df.at[row, i] = (df.at[row, i]-colMean)/colStd

    return df

xTrain=standerdize(xTrain)

xTest=standerdize(xTest)

names = xTrain.columns



from sklearn import preprocessing

X_scaled = preprocessing.scale(xTrain)

train_x = np.array(pd.DataFrame(X_scaled,columns=names).values.tolist())

X_scaled = preprocessing.scale(xTest)

test_x = np.array(pd.DataFrame(X_scaled,columns=names).values.tolist())


####################################################################################################################
def confusionHam(actual,pred):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(actual)):
        if actual[i]==0 and pred[i]==0:
            TP+=1
        elif actual[i]==1 and pred[i]==1:
            TN+=1
        elif actual[i]==1 and pred[i]==0:
            FP+=1
        else:
            FN+=1
    return TP, TN, FP, FN



def accuracy(actual, pred):
    TP, TN, FP, FN = confusionHam(actual,pred)
    result = (TP+TN)/len(actual)
    return result


def precisionHam(actual, pred):
    TP, TN, FP, FN = confusionHam(actual,pred)
    try:
        result = TP / (TP + FP)
    except:
        result = 0.404
    return result


def recallHam(actual, pred):
    TP, TN, FP, FN = confusionHam(actual,pred)
    try:
        result = TP / (TP + FN)
    except:
        result = 0.404
    return result



def confusionSpam(actual,pred):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(actual)):
        if actual[i]==0 and pred[i]==0:
            TN+=1
        elif actual[i]==1 and pred[i]==1:
            TP+=1
        elif actual[i]==1 and pred[i]==0:
            FN+=1
        else:
            FP+=1
    return TP, TN, FP, FN


def precisionSpam(actual, pred):
    TP, TN, FP, FN = confusionSpam(actual,pred)
    try:
        result = TP / (TP + FP)
    except:
        result = 0.404
    return result


def recallSpam(actual, pred):
    TP, TN, FP, FN = confusionSpam(actual,pred)
    try:
        result = TP / (TP + FN)
    except:
        result = 0.404
    return result


#################################################################################################################
gammas = [0.02, 0.025, 0.03, 0.035, 0.04]
cs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
gridSearchDF = pd.DataFrame(columns=['C value', 'gamma', 'accuracy', 'Ham precision', 'Ham recall', 'Spam precision', 'Spam recall'])

for g in gammas:
    for c in cs:
        svc = svm.SVC(kernel="rbf", gamma=g, C=c).fit(train_x, train_y)
        result = svc.predict(test_x)

        Accuracy = accuracy(test_y, result)

        HamPrecision = precisionHam(test_y, result)
        HamRecall = recallHam(test_y, result)

        SpamPrecision = precisionSpam(test_y, result)
        SpamRecall = recallSpam(test_y, result)

        gridSearchDF = gridSearchDF.append({

            'gamma': g,
            'C value': c,
            'accuracy': Accuracy,
            'Ham precision': HamPrecision,
            'Ham recall': HamRecall,
            'Spam precision': SpamPrecision,
            'Spam recall': SpamRecall
        }, ignore_index=True)

gridSearchDF.to_csv('T1/SVMGridSearch')