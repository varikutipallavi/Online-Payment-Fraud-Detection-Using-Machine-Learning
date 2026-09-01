from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# ✅ Load model using absolute path
model_path = os.path.join(os.path.dirname(__file__), 'payments.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/pred', methods=['POST'])
def pred():
    try:
        form = request.form

        # ✅ 1. Mapping transaction type (text) to number
        type_mapping = {
            'CASH_IN': 0,
            'CASH_OUT': 1,
            'DEBIT': 2,
            'PAYMENT': 3,
            'TRANSFER': 4
        }

        # ✅ 2. Get and normalize input
        t_type = form['type'].strip().upper()
        type_encoded = type_mapping.get(t_type, -1)

        # ✅ 3. Validate type
        if type_encoded == -1:
            return render_template('result.html', prediction_text="❌ Invalid transaction")

        # ✅ 4. Prepare input array for prediction
        x = np.array([[float(form['step']), float(type_encoded), float(form['amount']),
                       float(form['oldbalanceorg']), float(form['newbalanceorig']),
                       float(form['oldbalancedest']), float(form['newbalancedest']), 0.0]])

        prediction = model.predict(x)
        result = "✅ Fraud Detected" if prediction[0] == 1 else "✅ Not Fraud"
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        return render_template('result.html', prediction_text=f"⚠️ Error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
