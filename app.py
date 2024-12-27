from flask import Flask, request, render_template
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")  # Replace with the correct path to your model

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input from the form
        user_input = request.form.get('user_input')  # Retrieve the text input
        
        # Ensure the input is in a list format for the vectorizer
        input_features = [user_input]
        
        # Make a prediction
        prediction = model.predict(input_features)
        
        # Map the prediction to human-readable labels
        if prediction[0] == 1:
            result = "Scam"
        else:
            result = "Not a scam"
        
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
