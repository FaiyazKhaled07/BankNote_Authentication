#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request
import numpy as np
import pickle
import pandas as pd


app=Flask(__name__)


file_in = open("model.pkl","rb")
classifier=pickle.load(file_in)

@app.route('/')
def start():
    return "Welcome to the app"

@app.route('/predict')
def predict_note_authentication():
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
  
    return "The prediction is" + str(prediction)


if __name__=='__main__':
    app.run()


# In[ ]:




