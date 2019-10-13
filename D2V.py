import pandas as pd

TrainData = pd.read_csv("T1/Training.csv").drop(columns=["Unnamed: 0"])
TrainData = TrainData.drop([0]).reset_index(drop=True)

TestData = pd.read_csv("T1/Testing.csv").drop(columns=["Unnamed: 0"])
TestData = TestData.drop([0]).reset_index(drop=True)

TrainData = TrainData.sample(frac=1).reset_index(drop=True)
TestData = TestData.sample(frac=1).reset_index(drop=True)
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

data = TrainData["message"].tolist()

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")


model= Doc2Vec.load("d2v.model")
TrainD2VDataFrame = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"])

for i in range(len(model.docvecs)):
    TrainD2VDataFrame = TrainD2VDataFrame.append([{
        "x1":model.docvecs[i][0],
        "x2":model.docvecs[i][1],
        "x3":model.docvecs[i][2],
        "x4":model.docvecs[i][3],
        "x5":model.docvecs[i][4],
        "x6":model.docvecs[i][5],
        "x7":model.docvecs[i][6],
        "x8":model.docvecs[i][7],
        "x9":model.docvecs[i][8],
        "x10":model.docvecs[i][9],
        "x11":model.docvecs[i][10],
        "x12":model.docvecs[i][11],
        "x13":model.docvecs[i][12],
        "x14":model.docvecs[i][13],
        "x15":model.docvecs[i][14],
        "x16":model.docvecs[i][15],
        "x17":model.docvecs[i][16],
        "x18":model.docvecs[i][17],
        "x19":model.docvecs[i][18],
        "x20":model.docvecs[i][19]

    }],ignore_index=True)


allTrain = pd.concat([TrainData, TrainD2VDataFrame], axis=1)

#####################################################################################################



#to find the vector of a document which is not in training data

TestD2VDataFrame = pd.DataFrame(columns=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"])

for m in TestData["message"]:
    test_data = word_tokenize(m.lower())
    v1 = model.infer_vector(test_data)
    TestD2VDataFrame = TestD2VDataFrame.append([{
        "x1": v1[0],
        "x2": v1[1],
        "x3": v1[2],
        "x4": v1[3],
        "x5": v1[4],
        "x6": v1[5],
        "x7": v1[6],
        "x8": v1[7],
        "x9": v1[8],
        "x10": v1[9],
        "x11": v1[10],
        "x12": v1[11],
        "x13": v1[12],
        "x14": v1[13],
        "x15": v1[14],
        "x16": v1[15],
        "x17": v1[16],
        "x18": v1[17],
        "x19": v1[18],
        "x20": v1[19]

    }], ignore_index=True)

allTest = pd.concat([TestData, TestD2VDataFrame], axis=1)



allTrain.to_csv('T1/TrainingD2V.csv')
allTest.to_csv('T1/TestingD2V.csv')
# print("V1_infer", v1)

# to find most similar doc using tags

# mostSimilarToV1 = [(model.wv.similar_by_vector(v1)[i][1]) for i in range(len(model.wv.similar_by_vector(v1)))]

