import pickle
from flask import Flask,request,render_template,jsonify,url_for,app
import pandas as pd
import numpy as np


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = []

    for key, value in request.form.items():

        if key == "Sex":
            data.append(1 if value == "male" else 0)
        else:
            data.append(float(value))

    final_data = np.array(data).reshape(1, -1)

    output = model.predict(final_data)[0]

    if output == 0:
        return render_template(
            'output.html',
            prediction="The person is dead"
        )

    return render_template(
        'output.html',
        prediction="The person survived"
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )