# -*- coding: utf-8 -*-
"""
Created on Fri May  7 01:31:40 2021

@author: dshar
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

#Read the dataset
df = pd.read_csv('winequalityN.csv')
#Create Target Variable
y = [ 1 if x=='white' else 0 for x in df.type] 

#Keep top 4 important columns
df.drop(['type', 'fixed acidity', 'citric acid',
       'residual sugar', 'free sulfur dioxide',
       'total sulfur dioxide', 'pH', 'sulphates',
       'quality'],axis=1, inplace = True)

#Fill mean for missing values
df.update(df.fillna(df.mean()))

#Chosen Random Forest Classifier from experiments
rnd = RandomForestClassifier()
rnd.fit(df,y)

#Create pickle file
pickle.dump(rnd,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6.3, 0.34,1.6, 0.5]]))