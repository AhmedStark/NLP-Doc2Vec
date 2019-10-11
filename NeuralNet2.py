# load libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd

TrainData = pd.read_csv("/home/ahmed/Desktop/python/Assignment2/T2/TrainingD2V.csv").drop(columns=["Unnamed: 0"])
TestData = pd.read_csv("/home/ahmed/Desktop/python/Assignment2/T2/TestingD2V.csv").drop(columns=["Unnamed: 0"])

xTrain = TrainData[['polarity', "body_len", "punctuation%", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]].reset_index(drop=True)
xTest = TestData[['polarity',"body_len","punctuation%", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]].reset_index(drop=True)

train_y = TrainData['label'].reset_index(drop=True).values.tolist()
test_y = TestData['label'].reset_index(drop=True).values.tolist()

def convertClassesToNums(ylist):
    result = []
    for i in ylist:
        if i == 'ham':
            result.append([1,0])
        else:
            result.append([0,1])
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


#################################################################################################################

names = xTrain.columns
from sklearn import preprocessing

X_scaled = preprocessing.scale(xTrain)

train_x = np.array(pd.DataFrame(X_scaled,columns=names).values.tolist())

X_scaled = preprocessing.scale(xTest)

test_x = np.array(pd.DataFrame(X_scaled,columns=names).values.tolist())


###################################################################################################################
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


# instantiate model
gridSearchDF = pd.DataFrame(columns=['Epoch', 'Neurons', 'batch size', 'learning rate', 'accuracy', 'Ham precision', 'Ham recall', 'Spam precision', 'Spam recall'])

for nn in range(50,150,20):#500
    print("Neurons: ", nn)
    for bat in range(1,52,10):
        for ep in range(5, 6, 10):
            lrs = [1,0.1,0.01, 0.001]
            for lrate in lrs:
                model = Sequential([
                    Dense(nn, activation='relu', input_shape=(23,)),
                    Dense(nn, activation='relu'),
                    Dense(nn, activation='relu'),
                    Dense(2, activation='softmax'),
                ])

                model.compile(
                    optimizer=Adam(lr=lrate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                )

                model.fit(
                    train_x, # training data
                    train_y, # training targets [0,1][1,0]
                    epochs=ep,
                    batch_size=bat,
                )

                # Predict on the first 5 test images.
                predictions = model.predict(test_x)

                # Print our model's predictions.
                pred = np.argmax(predictions, axis=1) # [1,1, .........,0,1] 1 for spam and 0 for ham
                actual = np.argmax(test_y, axis=1)  # [1,1, .........,0,1] 1 for spam and 0 for ham

                Accuracy = accuracy(actual, pred)

                HamPrecision = precisionHam(actual, pred)
                HamRecall = recallHam(actual, pred)

                SpamPrecision = precisionSpam(actual, pred)
                SpamRecall = recallSpam(actual, pred)

                gridSearchDF = gridSearchDF.append({
                    'Epoch': ep,
                    'Neurons': nn,
                    'batch size': bat,
                    'accuracy': Accuracy,
                    'learning rate': lrate,
                    'Ham precision': HamPrecision,
                    'Ham recall': HamRecall,
                    'Spam precision': SpamPrecision,
                    'Spam recall': SpamRecall
                }, ignore_index=True)


gridSearchDF.to_csv('T2/NeuralNetGridSearch.csv')
