import pickle
import pandas as pd

# Load model
model = pickle.load(open("models/model.pkl", "rb"))

# Sample input (modify based on dataset)
sample = pd.DataFrame([[7, 8, 75]], columns=["Sleep_Hours","Quality","Heart_Rate"])

# Predict
prediction = model.predict(sample)

print("Predicted Stress Level:", prediction)
