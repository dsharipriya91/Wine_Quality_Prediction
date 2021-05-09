# -*- coding: utf-8 -*-
"""
Created on Fri May  7 01:31:40 2021

@author: dshar
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('winequalityN.csv')
y = [ 1 if x=='white' else 0 for x in df.type] 
df.drop(['type','citric acid','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'],axis=1, inplace = True)
df.update(df.fillna(df.mean()))

rnd = RandomForestClassifier()
rnd.fit(df,y)


pickle.dump(rnd,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6.3, 0.34,1.6]]))