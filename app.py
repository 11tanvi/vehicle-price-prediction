from flask import Flask, request, render_template
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model components
import os
model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))

encoders = joblib.load(os.path.join(os.path.dirname(__file__), 'encoders.pkl'))
feature_names = joblib.load(os.path.join(os.path.dirname(__file__), 'feature_names.pkl'))

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form input
        input_data = {
            'make': request.form['make'],
            'model': request.form['model'],
            'engine': request.form['engine'],
            'cylinders': int(request.form['cylinders']),
            'fuel': request.form['fuel'],
            'mileage': float(request.form['mileage']),
            'transmission': request.form['transmission'],
            'trim': request.form['trim'],
            'body': request.form['body'],
            'doors': int(request.form['doors']),
            'exterior_color': request.form['exterior_color'],
            'interior_color': request.form['interior_color'],
            'drivetrain': request.form['drivetrain'],
            'car_age': int(request.form['car_age'])
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # Combine make + model
        input_df['make_model'] = input_df['make'] + ' ' + input_df['model']
        input_df.drop(columns=['make', 'model'], inplace=True)

        # Encode categorical columns
        for col in input_df.select_dtypes(include='object').columns:
            if col in encoders:
                encoder = encoders[col]
                if input_df[col].iloc[0] in encoder.classes_:
                    input_df[col] = encoder.transform(input_df[col])
                else:
                    input_df[col] = -1
            else:
                input_df[col] = -1

        #  Match column order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale and Predict
        input_scaled = scaler.transform(input_df)
        predicted_price_usd = model.predict(input_scaled)[0]

        # Convert USD to INR (approx 1 USD = 83 INR)
        predicted_price_inr = round(predicted_price_usd * 83, 2)

        # Show both USD and INR in result
       # return render_template(
           # 'index.html',
           # prediction_text=f"Estimated Price: ${round(predicted_price_usd, 2)} (~₹{predicted_price_inr})"
        # Show both USD and INR in result
        return render_template(
            'result.html',
            predicted_price=predicted_price_inr,  # For chart
            price_text=f"Estimated Price: ${round(predicted_price_usd, 2)} (~₹{predicted_price_inr})"  # For display
        )


    except Exception as e:
        return render_template('index.html', prediction_text=f'❌ Error: {str(e)}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
