import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("house_model.pkl")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

predictions = model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()