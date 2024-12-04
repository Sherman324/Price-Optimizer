# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
data = pd.read_csv("data/buyers.csv")

# List of columns to drop
columns_to_drop = ["PRODUCT ID", "PRODUCT", "CATEGORY", "UOM"]

# Drop the specified columns
data = data.drop(columns=columns_to_drop)

# Display the first few rows of the updated DataFrame
print(data.head())

# Split data into features (X) and target (y)
X = data.iloc[:, :-1].values  # All columns except the last one as features
y = data.iloc[:, -1].values   # The last column as the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fit the linear regression model to the training set
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predicting the results
y_pred = lr.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Save the model
joblib.dump(lr, filename='model/model.pkl')
