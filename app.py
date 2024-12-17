from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('amodel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age'))
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol = float(request.form.get('chol'))
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form.get('restecg'))
        thalach = float(request.form.get('thalach'))
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form.get('oldpeak'))
        slope = int(request.form.get('slope'))
        ca = int(request.form.get('ca'))
        thal = int(request.form.get('thal'))

        # Check if any of the values are missing
        if any(val is None for val in
               [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
            return jsonify({'error': 'Missing values'}), 400

        input_query = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        result = model.predict(input_query)[0]

        # Ensure the result is serializable by converting to a native Python type
        result = int(result)

        return jsonify({'condition level': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
