import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier


data = pd.read_csv('./heart.csv')
print("sklearn version : ")
print(sklearn.__version__)
features = ['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'
from sklearn.model_selection import train_test_split
X = data[features]
y = data[target]
print(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# random forest
rfc = RandomForestClassifier()  
rfc.fit(x_train, y_train) 
# logistic reg
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
# svm
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)
# XGB
xgb_model = xgb.XGBClassifier()
xgb_model.fit(x_train, y_train)
#stacking
estimators = [
    ('random_forst',rfc),
    ('logistic_reg',logreg),
    ('svm',svm),
    ('xgb_model',xgb_model)
]
stack_model = StackingClassifier(
    estimators = estimators,final_estimator = DecisionTreeClassifier()
)
stack_model.fit(x_train,y_train)

y_pred = stack_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
pr = precision_score(y_test,y_pred)
print("Precision:",pr)
recall = recall_score(y_test, y_pred)
print("recall:",recall)
f1 = f1_score(y_test, y_pred)
print("fl score:",f1)

warnings.filterwarnings("ignore")
pickle.dump(rfc,open('model1.pkl','wb'))