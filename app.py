from flask import Flask, render_template, request
import os
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')  
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = model.predict(final)[0]
    return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}') 
if _name_ == '_main_':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
