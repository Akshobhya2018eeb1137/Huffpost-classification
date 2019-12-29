import pandas as pd

df = pd.read_csv(r'C:\Users\Akshobhya\Documents\NLP\news_train.csv')
pd.set_option('max_colwidth',1000)

df.replace("ENVIRONMENT", "GREEN", inplace = True)
df.replace("PARENTING", "PARENTS", inplace = True)
df.replace("THE WORLDPOST", "WORLDPOST", inplace = True)

df.fillna(' ',inplace = True)
df['new'] = df['headline'] +' ' + df['short_description']
df.drop(['headline','short_description','date','link','authors'],axis = 1, inplace = True)

df.dropna(axis=0,inplace=True)
df.reset_index(inplace=True)
df.drop(['index'], axis = 1, inplace = True)

import string

def removePunctuations(s):
    out =  s.translate(str.maketrans('','',string.punctuation))
    return out


df['new'] = df['new'].apply(removePunctuations)   
arr = list(df.category.unique())
for i in arr:
    df.loc[df.category == i,"category"] = arr.index(i)
df.to_csv(r'C:\Users\Akshobhya\Documents\NLP\new.csv')
df1 = pd.read_csv(r'C:\Users\Akshobhya\Documents\NLP\news_test.csv')

pd.set_option('max_colwidth',1000)

df1.replace("ENVIRONMENT", "GREEN", inplace = True)
df1.replace("PARENTING", "PARENTS", inplace = True)
df1.replace("THE WORLDPOST", "WORLDPOST", inplace = True)

df1.fillna(' ',inplace = True)
df1['new'] = df1['headline'] +' ' + df1['short_description']
df1.drop(['headline','short_description','date','link','authors'],axis = 1, inplace = True)

df1.dropna(axis=0,inplace=True)
df1.reset_index(inplace=True)
df1.drop(['index'], axis = 1, inplace = True)

df1.fillna(' ',inplace = True)


df1['new'] = df1['new'].apply(removePunctuations)   

df1.to_csv(r'C:\Users\Akshobhya\Documents\NLP\new_t.csv')
df = pd.read_csv(r'C:\Users\Akshobhya\Documents\NLP\new.csv',index_col = [0])

df.info()
df1 =pd.read_csv(r'C:\Users\Akshobhya\Documents\NLP\new_t.csv',index_col=[0])
df1.info()

x = df['new']
y = df['category']

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 4)

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


sgd = Pipeline([('vect',CountVectorizer()),
               ('tfidf',TfidfTransformer()),
               ('clf',SGDClassifier(loss ='hinge',  alpha=1e-4, random_state = 42, max_iter = 10,tol=None))
               ])

sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
print('accuracy %s' %accuracy_score(y_pred,y_test))
