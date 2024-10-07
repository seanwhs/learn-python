# Housing Price Prediction Using Neural Networks

This document provides a detailed explanation of a Python script used to predict housing prices using a neural network model. The code leverages TensorFlow/Keras for model building and scikit-learn for data preprocessing and evaluation.

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from numpy import int64
from tensorflow import keras
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
```

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **matplotlib** and **seaborn**: Data visualization.
- **tensorflow** and **keras**: Machine learning framework for building neural networks.
- **scikit-learn**: Tools for splitting data, evaluating models, and feature importance.

## 2. Load and Inspect Data

```python
df = pd.read_csv('data/kc_house_data.csv')
df.head(3)
df.shape
df.info()
df.describe()
```

- **Load dataset**: Reads the housing data from a CSV file.
- **Inspect dataset**: Displays the first few rows, shape, data types, and summary statistics.

## 3. Data Preprocessing

### Convert Registration Date into Age

```python
df['reg_year'] = df['date'].str[:4]
df['reg_year'] = df['reg_year'].astype(int64)
```

- **Extract year**: Extracts the year from the date column.
- **Convert to integer**: Casts the year to integer type.

### Calculate House Age

```python
df['house_age'] = np.NaN

for i in df.index:
    if df.loc[i, 'yr_renovated'] == 0:
        df.loc[i, 'house_age'] = df.loc[i, 'reg_year'] - df.loc[i, 'yr_built']
    else:
        df.loc[i, 'house_age'] = df.loc[i, 'reg_year'] - df.loc[i, 'yr_renovated']

df.drop(['date', 'yr_built', 'yr_renovated', 'reg_year'], axis='columns', inplace=True)
```

- **Calculate age**: Computes the age of the house based on renovation and build year.
- **Remove unnecessary columns**: Drops columns related to dates.

### Remove Irrelevant Columns

```python
df.drop(['id', 'lat', 'long', 'zipcode'], axis='columns', inplace=True)
df = df[df['house_age'] != -1]
```

- **Drop columns**: Removes columns that may mislead the model or contain irrelevant data.
- **Handle invalid ages**: Filters out records with invalid `house_age`.

## 4. Data Visualization

```python
for i in df.columns:
    sns.displot(df[i])
    plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)
plt.show()

for i in df.columns:
    sns.boxplot(x=df[i])
    plt.show()
```

- **Distribution plots**: Visualizes the distribution of each feature.
- **Pair plot**: Shows relationships between features.
- **Heatmap**: Displays correlations between features.
- **Box plots**: Identifies outliers in the dataset.

## 5. Split Data into Training and Test Sets

```python
X = df.drop(['price'], axis=1)
y = df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

- **Feature and target separation**: Splits data into features (`X`) and target (`y`).
- **Train-test split**: Divides the dataset into training and test sets.

## 6. Build and Train the Model

```python
model = keras.Sequential()
model.add(layers.Dense(14, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(X_train, y_train, validation_split=0.33, batch_size=32, epochs=100)
model.summary()
```

- **Model architecture**: Constructs a neural network with several dense layers using ReLU activation functions.
- **Compile model**: Configures the model with Mean Squared Error (MSE) as the loss function and Adam optimizer.
- **Train model**: Fits the model to the training data and includes validation.

## 7. Evaluate the Model

```python
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-Squared: {r2}')
```

- **Make predictions**: Predicts prices on the test set.
- **Calculate metrics**: Computes MAE, MSE, RMSE, and R-squared values.

### Visualization of Training

```python
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['mse'], label='Train MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation MSE')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.show()
```

- **Loss, MSE, and MAE plots**: Shows training and validation metrics over epochs.

### Predictions vs Actual Prices

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()
```

- **Scatter plot**: Visualizes the correlation between actual and predicted prices.

### Residuals Analysis

```python
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.show()
```

- **Residual distribution**: Plots the distribution of residuals (errors).
- **Residuals vs. predictions**: Shows how residuals vary with predicted values.

## 8. Feature Importance

```python
results = permutation_importance(model, X_test, y_test, scoring='neg_mean_squared_error')
importance = results.importances_mean

plt.figure(figsize=(12, 6))
sns.barplot(x=X.columns, y=importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=90)
plt.show()
```

- **Permutation importance**: Assesses feature importance by evaluating model performance after permuting feature values.

## 9. Example Prediction

```python
Xnew = np.array([[2, 3, 1280, 5550, 1, 0, 0, 4, 7, 2280, 0, 1440, 5750, 60]], dtype=np.float64)
ynew = model.predict(Xnew)
print(f'Xnew={Xnew}')
print(f'Prediction = {ynew}')
```

- **New data**: Makes a prediction using a new set of feature values.