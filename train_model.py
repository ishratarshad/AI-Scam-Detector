import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Example dataset
data = pd.DataFrame({
    'text': ['This is a scam', 'Completely legit offer', 'Scam alert', 'Buy now!', 'Amazing product!'],
    'label': [1, 0, 1, 0, 0]  # 1 = Scam, 0 = Not a scam
})

# Features and target
X = data['text']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with vectorization and classification
model = make_pipeline(CountVectorizer(), RandomForestClassifier())

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("Model saved as 'model.pkl'")
