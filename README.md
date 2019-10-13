# NLP-Doc2Vec
This is part of a Data mining and knowledge engineering course I am taking this semester. I am experimenting with Doc2Vec.

# How to use:
#### 1. Run this in the stanford-corenlp-full-2018-10-05 file directory to start CoreNLP:

```python
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
```

#### 2. Put the data file in the same directory.

#### 3. Create files T1, T2 in that directory.

#### 4. Run the python file 'making_table.py' for task 1, or 'making_tables2.py' for task 2:
This will get the data from their files and put it into pandas dataframes (Training and Testing), then save those dataframes as csv files in T1 for Task 1 and T2 for Task 2.

#### 5. Run the python file 'D2V.py' for task 1, or 'D2V2.py' for task 2:
This will change the text in messages into vectors of 20 variables each using Doc2Vec and save that as a csv file in T1 for Task 1 and T2 for Task 2.

#### 6. Run the python file 'NB.py' for task 1, or 'NB2.py' for task 2:
This will show you accuracy, precision, and recall of both classes 'ham' and 'spam' after training and testing with Naive Bayes classifier.

#### 7. Run the python file 'NeuralNet.py' for task 1, or 'NeuralNet2.py' for task 2:
This will show you accuracy, precision, and recall of both classes 'ham' and 'spam' after training and testing with Neural network classifier.

#### 8. Run the python file 'SVM.py' for task 1, or 'SVM2.py' for task 2:
This will show you accuracy, precision, and recall of both classes 'ham' and 'spam' after training and testing with Support Vector Machine classifier.
