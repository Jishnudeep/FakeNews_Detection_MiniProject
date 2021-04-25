# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:51:33 2021

@author: bjish
"""

from flask import Flask, request, jsonify, render_template
import pickle


#Initialising Flask
app = Flask(__name__)
model = pickle.load(open("final_model.sav","rb"))

#default page of web app
@app.route('/')
def home():
    return render_template('index.html')

#to use the predict button in our web-app
@app.route('/predict', methods = ['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    var = [str(x) for x in request.form.values()]
    print(var)
    prediction = model.predict(var)
    prob = model.predict_proba(var)


    #return (print("The given statement is ",prediction[0]),
      # print("The truth probability score is ",prob[0][1]))
    return render_template('predict.html', prediction_text = "The given statement is: {}".format(prediction[0]),
                           prediction_probability = "The probability score is: {0:.3f}".format(prob[0][1]))


if __name__ == "__main__":
    app.run(debug=True)