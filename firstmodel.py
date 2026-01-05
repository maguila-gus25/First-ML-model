# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 2: Prepare the data
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [12, 25, 32, 48, 58, 65, 72, 88, 98]
}

df = pd.DataFrame(data)

# Features and target 
x = df[['Hours_Studied']] #Input (DataFrame)
y = df['Score'] #Output (Series)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(x_test)
print("Predictions:", y_pred)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 7: Visualize results
# Convert X to 1D array for plotting 
X_1d = x['Hours_Studied'].values
y_pred_full = model.predict(x).flatten()

plt.scatter(X_1d, y, color='blue', label='Actual Scores')
plt.plot(X_1d, y_pred_full, color='red', label='Predicted Line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours Studied vs Score Prediction')
plt.legend()
plt.show()
