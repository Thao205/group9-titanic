import pandas as pd

# Load the dataset
train = pd.read_csv('/workspaces/group9-titanic/Titanic.csv')
#Data cleaning
# Drop the 'Cabin' column
train.drop(["Cabin"], axis=1, inplace=True)

# Fill missing values in the 'Age' column with the mean age
train["Age"] = train["Age"].fillna(train["Age"].mean())

# Fill missing values in the 'Embarked' column with 'S'
train["Embarked"] = train["Embarked"].fillna("S")

# Check for any remaining missing values
missing_values = train.isnull().sum()
print(missing_values)
#Encoding Categorical columns
train
train["Sex"].value_counts()
train["Embarked"].value_counts()
train.replace({"Sex":{"male":0,"female":1},
               "Embarked":{"S":0, "C":1, "Q":2}},inplace=True)
#Model Training
train
X = train.drop(["PassengerId","Survived","Name","Ticket"],axis=1)
Y = train["Survived"]

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test = tts(X,Y,test_size=0.25,random_state=42)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train,y_train)

#Model evaluation

from sklearn.metrics import accuracy_score

x_train_prediction = model.predict(x_train)
x_train_prediction


accuracy_train = accuracy_score(y_train,x_train_prediction)
accuracy_train

x_test_prediction = model.predict(x_test)
x_test_prediction

accuracy_test = accuracy_score(y_test,x_test_prediction)
accuracy_test

import pickle
with open("model.pkl","wb") as f:
    pickle.dump(model,f)
with open("model.pkl","rb") as f:
    loaded_model = pickle.load(f)
loaded_model.predict([[3,0,22,1,0,7,0]])