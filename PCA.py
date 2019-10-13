import pandas as pd
PCA = pd.read_csv("PCA_Loadings.csv").drop([23, 24, 25, 26, 27, 28, 29])

names = PCA.columns.to_list()
names.remove("Unnamed: 0")

PickNo = 0.4
features = []
for i in names:

    for s in range(len(PCA[i])):
        if float(PCA.at[s,i])>PickNo or float(PCA.at[s,i])<(-1*PickNo):
            features.append(PCA.at[s,'Unnamed: 0'])

features = list(dict.fromkeys(features))
print(features)

# ['x18', 'x4', 'punctuation%', 'x20', 'x8', 'x11', 'x7', 'polarity', 'body_len', 'x15', 'x17', 'x19', 'x1', 'x3', 'x6', 'x5', 'x12', 'x16']