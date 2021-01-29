import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df=pd.read_csv('BankNote_Authentication.csv')

### Independent and Dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

### Implement Random Forest classifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

## Prediction
y_pred=classifier.predict(X_test)

### Check Accuracy
score=accuracy_score(y_test,y_pred)
print(score)

### Predict
print(classifier.predict([[2,3,4,1]]))

### Create a Pickle file using serialization 
pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
