from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt


#read dataset
df=pd.read_csv('spam.csv')

df.head()

#check for null values
df.info()


#changing to 0 and 1
df['Category']=df['Category'].map({'ham':0,'spam':1})

#import countvectorizer
from sklearn.feature_extraction.text import CountVectorizer

#split data
count=CountVectorizer()
#fit into the vectorizer
train=count.fit_transform(df['Message'])

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(train,df['Category'])

#check accuracy
from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,pred))
filename='model.pkl'
_=joblib.dump(clf, filename, compress=9)

clf2 = joblib.load(filename)

model_columns = list(df.columns)
joblib.dump(count,'counts.pkl')
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
