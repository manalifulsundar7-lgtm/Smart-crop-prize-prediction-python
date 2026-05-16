import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Excel file

data = pd.read_excel(r"C:\Users\Manali Fulsundar\OneDrive\Desktop\smartcropproject1\crop_price.xlsx")

# Show dataset
print(data.head())

# Features and target
X = data[['Year','Month']]
y = data['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Show predictions
print("Predicted Prices:", predictions)

# Plot graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Crop Price Prediction")
plt.show()

# Future prediction
future_data = pd.DataFrame({'Year':[2023],'Month':[7]})
future_price = model.predict(future_data)

print("Predicted Future Price:", future_price[0])

