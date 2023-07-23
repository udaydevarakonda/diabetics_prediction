from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import warnings


app = Flask(__name__, static_folder='images')

model = joblib.load("model1.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        Pregnancies = request.form['Pregnancies'] 
        Glucose = request.form['Glucose'] 
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']

        arr = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                         Age]
        res = model.predict([arr])[0]
        return render_template('after.html',res=res)
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)