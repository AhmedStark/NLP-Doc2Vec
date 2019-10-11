import pandas as pd
import numpy as np
print('sup')
TrainData = pd.read_csv("T2/TrainingD2V.csv").drop(columns=["Unnamed: 0"])
TestData = pd.read_csv("T2/TestingD2V.csv").drop(columns=["Unnamed: 0"])
print(TrainData)
xTrain = TrainData[['polarity', "body_len", "punctuation%", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]].reset_index(drop=True)
xTest = TestData[['polarity',"body_len","punctuation%", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]].reset_index(drop=True)

yTrain = TrainData['label'].reset_index(drop=True)
yTest = TestData['label'].reset_index(drop=True)

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


######################################################################################################################
from sklearn import preprocessing
# # Get column names first
# # Create the Scaler object
# scaler = preprocessing.StandardScaler()
# # Fit your data on the scaler object
# xTrain = scaler.fit_transform(xTrain)
# xTrain = pd.DataFrame(xTrain, columns=names)
#
# names = xTest.columns
# # Create the Scaler object
# scaler = preprocessing.StandardScaler()
# # Fit your data on the scaler object
# xTest = scaler.fit_transform(xTest)
# xTest = pd.DataFrame(xTest, columns=names)
# print("Standardisation results:")
#
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(xTrain, yTrain)
#
# print(clf.predict(xTest))
#
# print(clf.score(xTest, yTest))
#
# from sklearn.metrics import classification_report
#
# print(classification_report(yTest, clf.predict(xTest)))
#####################################################################################################################
# normalise

# names = xTrain.columns
# x = xTrain.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# xTrain = pd.DataFrame(x_scaled, columns=names)
#
#
#
# names = xTest.columns
# x = xTest.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# xTest = pd.DataFrame(x_scaled, columns=names)

#################################################################################################################

# xTrain = xTrain[['x18', 'x4', 'punctuation%', 'x20', 'x8', 'x11', 'x7', 'polarity', 'body_len', 'x15', 'x17', 'x19', 'x1', 'x3', 'x6', 'x5', 'x12', 'x16']]
# xTest = xTest[['x18', 'x4', 'punctuation%', 'x20', 'x8', 'x11', 'x7', 'polarity', 'body_len', 'x15', 'x17', 'x19', 'x1', 'x3', 'x6', 'x5', 'x12', 'x16']]
names = xTrain.columns
from sklearn import preprocessing

X_scaled = preprocessing.scale(xTrain)

xTrain=pd.DataFrame(X_scaled,columns=names)

X_scaled = preprocessing.scale(xTest)

xTest=pd.DataFrame(X_scaled,columns=names)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(xTrain, yTrain)

print(clf.predict(xTest))

print(clf.score(xTest, yTest))

from sklearn.metrics import classification_report
print('for Naive Bayes:')
print(classification_report(yTest, clf.predict(xTest)))

