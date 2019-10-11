# run this in the terminal first:
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000

import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import os
from sklearn.model_selection import train_test_split
import timeit
import numpy as np
import spacy
from pycorenlp import StanfordCoreNLP

import string
import os
import timeit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

start = timeit.default_timer()

# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
sp = spacy.load('en', disable=['parser', 'ner'])

# Parse the sentence using the loaded 'en' model object `nlp`

en1 = []
en2 = []
en3 = []
en4 = []
en5 = []

for f in range(1,6):

    types = ["ham", "spam"]
    data = [["type", "message"]]
    for g in range(2):

        x=types[g]
        y=str(f)

        data_dir = "/home/ahmed/Desktop/python/Assignment2/Data/enron"+ y+ "/"+x+"/"

        num_of_files = 0

        for filename in os.listdir(data_dir):

            num_of_files+=1

        for n in range(1,num_of_files+1):

            completeName = os.path.join(data_dir, str(n) + ".txt")
            with open(completeName, 'rb') as file:
                contents = file.read()
                try:
                    row = [types[g], contents.decode()]

                    data.append(row)
                except:
                    print(contents)
                file.close()


    if f==1:
        en1 = data

    elif f==2:
        en2=data
    elif f==3:
        en3=data
    elif f==4:
        en4=data
    else:
        en5=data



en1=pd.DataFrame(en1)
en2=pd.DataFrame(en2)
en3=pd.DataFrame(en3)
en4=pd.DataFrame(en4)
en5=pd.DataFrame(en5)

enronCorpus = [en1,en2,en3,en4,en5]

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

for k in range(len(enronCorpus)):
    enronCorpus[k].columns = ['label', 'message']
    enronCorpus[k]['body_len'] = enronCorpus[k]['message'].apply(lambda x: len(x) - x.count(" "))
    enronCorpus[k]['punctuation%'] = enronCorpus[k]['message'].apply(lambda x: count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    text = " ".join(re.split('\W+', text))
    sentence = sp(text)
    text = [word.lemma_ for word in sentence if word.text not in stopwords]
    return text


TrainSet = pd.concat([en1, en3, en5], ignore_index=True)

TestSet = pd.concat([en2, en4], ignore_index=True)

###########################################################################################################

nlp = StanfordCoreNLP('http://localhost:9000')

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score["compound"]




def sentence_split(text, properties={'annotators': 'ssplit', 'outputFormat': 'json'}):
    """Split sentence using Stanford NLP"""
    annotated = nlp.annotate(text, properties)
    # print(annotated)
    sentence_split = list()
    try:
        for sentence in annotated['sentences']:
            s = [t['word'] for t in sentence['tokens']]
            sentence_split.append(s)
    except:
        print(annotated)
    return sentence_split


def AddPolarityFeature(DF):
    sentiment = pd.DataFrame(columns = ['label','message','polarity',"body_len","punctuation%"])

    properties={
      'annotators': 'ssplit',
      'outputFormat': 'json'
      }


    t=0

    for par in DF["message"]:
        # print(t)
        # par=par[0]
        w=len(par)
        sent_text = sentence_split(par, properties)
        values=[]
        for s in sent_text:
            res=sentiment_analyzer_scores(" ".join(s))

            values.append(res)

        summ = 0

        for v in values:
            summ+=v
        try:
            value=summ/len(values)
        except:
            value=0

        sentiment = sentiment.append({
            "label":DF.at[t,"label"],
            "message": DF.at[t, "message"],
            "polarity": value,
            "punctuation%": DF.at[t,"punctuation%"],
            "body_len":DF.at[t,"body_len"]
        },ignore_index=True)
        if t!=w:
            t += 1
    return sentiment


#################################################################################################################
TrainSet = AddPolarityFeature(TrainSet)
TestSet = AddPolarityFeature(TestSet)

TrainSet.to_csv('T2/Training.csv')
TestSet.to_csv('T2/Testing.csv')
