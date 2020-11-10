import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("term-deposit-marketing-2020.csv")
print(data.head())
print(data.columns)
print(data.count())
print(data.isnull().sum()) # Eksik veri var mı ? Kontrolü
#print(data.describe())
#print(data[['job', 'y']].groupby("job").mean().reset_index())
#print(data["job"].value_counts())
job = data.iloc[:,1:2].values
target = data.iloc[:,13:14].values
print(job)
le = LabelEncoder()      # Kategorik tipteki nitelikleri numeriğe çevirmek için.


for col in data.columns:
    if(data[col].dtype == 'object'):
        data.loc[:,col] = le.fit_transform(data.loc[:,col])
        
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

correlation = data.corr()       # HANGİ ÖZELLİK HEDEF SONUÇ İÇİN DAHA ETKİLİ
                                # duration = 0.46
correlation.drop(["y"], axis=0, inplace=True)
max_corr = correlation["y"].max()
################################### RANDOM FOREST #######################################
"""
clf = RandomForestClassifier(n_estimators=95,random_state=0)
clf.fit(x_train,y_train)
result2 = clf.predict(x_test)
cm2 = confusion_matrix(y_test,result2)                       # %70 ACCURACY
print(cm2)
accuracy1 = accuracy_score(y_test, result2)
print(accuracy1)
print(data["y"].value_counts())
"""

################### LOGISTIC REGRESSION ################################
"""
clf = LogisticRegression(random_state=0)
clf.fit(x_train,y_train)
result = clf.predict(x_test)                              # %92.45 ACCURACY
accuracy_lr = accuracy_score(y_test, result)
"""

#################### KNN #########################################

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)                     # %92,55 ACCURACY
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


############################ Decision Tree ################################
"""
from sklearn import metrics
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset                 # %41,7 ACCURACY
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""
 ########################### CROSS VALIDATION ################################

cv = cross_validate(clf, X, y, cv=5)
print(cv['test_score'])
print("Ortalama başarı = ",cv['test_score'].mean())

print("Hedef için en etkili nitelik -> Duration = ",max_corr)

