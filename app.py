import pandas as pd
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
from flask import Flask, request, render_template
import numpy as np
import joblib
 

# Load the trained model and necessary preprocessing tools
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl") 
 

app = Flask(__name__)



@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')
@app.route('/login')
def login():
    return render_template('login.html')
def home():
	return render_template('home.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df) 


@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/detect', methods=['POST'])
def detect():
    Id = int(request.form["Id"])
    trans_hour = int(request.form["trans_hour"])
    trans_day = int(request.form["trans_day"])
    trans_month = int(request.form["trans_month"])
    trans_year = int(request.form["trans_year"])
    category = int(request.form["category"])  # Assuming category is encoded as an integer
    upi_number = int(request.form["upi_number"])
    age = int(request.form["age"])
    trans_amount = float(request.form["trans_amount"])
    state = int(request.form["state"])  # Assuming state is encoded as an integer
    zip_code = int(request.form["zip"])

        # Create feature array
    new_input = np.array([[Id, trans_hour, trans_day, trans_month, trans_year, 
                               category, upi_number, age, trans_amount, state, zip_code]])
        
        # Apply tr8ansformations
    new_input_scaled = scaler.transform(new_input)
    new_input_selected = new_input_scaled[:, selector.support_]

        # Make prediction
    prediction = best_model.predict(new_input_selected)[0]
    if prediction== 0:
        result = "VALID TRANSACTION"
    else:
        result = "FRAUD TRANSACTION"
    return render_template('result.html', OUTPUT='{}'.format(result))

if __name__ == "__main__":
    app.run()
