import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("house_model.pkl")

# Load test data
X_test = pd.read_csv(r"C:\Users\acer\Desktop\project2\code\X_test.csv")
y_test = pd.read_csv(r"C:\Users\acer\Desktop\project2\code\y_test.csv").squeeze()

# Predict house prices
predictions = model.predict(X_test)

# Create scatter plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.7, label="Predicted Prices")

# Draw ideal prediction line (y = x)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2,
    label="Ideal Prediction"
)

# Labels and title
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)

plt.show()