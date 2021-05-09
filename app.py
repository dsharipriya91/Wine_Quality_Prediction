# -*- coding: utf-8 -*-
"""
Created on Fri May  7 02:02:39 2021

@author: dshar
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = 'white' if prediction[0]==1 else 'red'
    
    return render_template('index.html', prediction_text='Wine Prediction: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
