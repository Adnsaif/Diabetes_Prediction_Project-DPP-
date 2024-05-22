import os
from django.shortcuts import render
from django.urls import path

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score

 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix




def home(request):
    return render(request, 'home.html')

 
def predict(request):
    return render(request, 'predict.html') 

def result(request):
     data = pd.read_csv(r'C:/Users/Xpert/Downloads/diabetes.csv')
     x = data.drop('Outcome', axis=1)
     y = data['Outcome']
     x_train,x_test,y_train,y_test = train_test_split( x, y, test_size=0.30 )
     model = LogisticRegression() 
     model.fit(x_train,y_train)
     
     predictions = model.predict(x_test)






# def result(request):
#     df = pd.read_csv('C:/Users/Xpert/Downloads/diabetes.csv')
#     x = df.drop(columns='Outcome', axis=1)
#     y = df['Outcome']
#     scaler = StandardScaler()
#     scaler.fit(x)
#     standarized_data = scaler.transform(x)

#     X_train, X_test, Y_train,Y_test = train_test_split(x,y,test_size =0.2)
#     X_train.shape
#     clf  = svm.SVC(kernel= 'linear')
#     clf.fit(X_train,Y_train)

#     X_train_prediction = clf.predict(X_train)
#     accuracy_score(X_train_prediction,Y_train)

#     X_test_prediction = clf.predict(X_test)
#     accuracy_score(X_test_prediction,Y_test)

    
     val1 = float(request.GET['n1'])
     val2 = float(request.GET['n2'])
     val3 = float(request.GET['n3'])
     val4 = float(request.GET['n4'])
     val5 = float(request.GET['n5'])
     val6 = float(request.GET['n6']) 
     val7 = float(request.GET['n7'])
     val8 = float(request.GET['n8'])


  

     pred = model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
     result1 = ""
     if pred ==[1]:
        result1 =  "Positive"
     else:
        result1 = "Negative"

     return render(request, 'predict.html', {"result2" : result1})
