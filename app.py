#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger


app=Flask(__name__)
Swagger(app)


file_in = open("model.pkl","rb")
classifier=pickle.load(file_in)

@app.route('/')
def start():
    return "Welcome to the app"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Authentication of Bank Notes 
    Enter values in the fields.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The result
        
    """
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
  
    return "The prediction is" + str(prediction)


if __name__=='__main__':
    app.run()


# In[ ]:




