# Heart Disease Prediction Model

This document explains the Python code used for building and evaluating a heart disease prediction model using a neural network.

## Libraries Used

- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Seaborn**: For statistical data visualization.
- **TensorFlow**: For building and training the neural network model.
- **Scikit-Learn**: For data splitting, metrics, and evaluation.

## Code Explanation

### 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
```

### 2. Fetching and Exploring Data

```python
df = pd.read_csv('data/heart.csv')
df.head(3)
```
- Load the dataset and display the first 3 rows.

```python
df.shape
df.info()
df.isna().sum()
df.describe()
```
- Print dataset shape, check data types, missing values, and summary statistics.

### 3. Checking for Outliers

```python
for i in df.columns:
    sns.boxplot(x=df[i])
    plt.show()
```
- Visualize outliers using boxplots for each feature.

### 4. Data Preparation for Visualization

```python
df_copy = df.copy()
df_copy['target'] = df_copy['target'].map({0: 'Healthy', 1: 'Heart Patients'})
df_copy['sex'] = df_copy['sex'].map({0: 'Female', 1: 'Male'})
```
- Map target and sex values to descriptive labels.

### 5. Data Visualization

```python
sns.countplot(x='sex', data=df_copy, hue='target')
plt.title('Number of Males vs Females by Health Status')
plt.show()
```
- Plot count of males vs females colored by health status.

```python
sns.histplot(df_copy[df_copy.target=='Heart Patients']['age'], label='Heart Patients')
sns.histplot(df_copy[df_copy.target=='Healthy']['age'], label='Healthy')
plt.legend()
plt.show()
```
- Histogram of age distribution by health status.

```python
sns.histplot(df_copy[df_copy.target=='Heart Patients']['chol'], label="Heart Patients")
sns.histplot(df_copy[df_copy.target=='Healthy']['chol'], label="Healthy")
plt.legend()
plt.show()
```
- Histogram of cholesterol distribution by health status.

### 6. Data Preparation for Modeling

```python
X = df.drop(['target'], axis=1)
y = df.target
print(X.head(3))
print(X.shape)
print('______________')
print(y.head(3))
print(y.shape)
```
- Split dataset into features (`X`) and target (`y`).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
print(X_train.shape)
print(X_train.shape[1])
```
- Split data into training and test sets.

### 7. Building the Neural Network Model

```python
model = keras.Sequential()
model.add(layers.Dense(11, activation ='relu', input_shape=(X_train.shape[1], )))
model.add(layers.Dense(1, activation ='sigmoid'))
```
- Define a sequential model with one hidden layer (11 neurons) and an output layer with a sigmoid activation function.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.22, epochs = 300)
model.summary()
```
- Compile and train the model. Display model summary.

### 8. Plotting Training History

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('no of epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
- Plot training and validation loss over epochs.

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('no of epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
- Plot training and validation accuracy over epochs.

### 9. Evaluating the Model

```python
y_pred = model.predict(X_test).flatten()
y_pred_binary = (y_pred > 0.5).astype(int)
```
- Predict on the test data and convert probabilities to binary predictions.

```python
conf_matrix = confusion_matrix(y_test, y_pred_binary)
print("Confusion Matrix:\n", conf_matrix)
```
- Print confusion matrix.

```python
class_report = classification_report(y_test, y_pred_binary, target_names=['Healthy', 'Heart Patients'])
print("Classification Report:\n", class_report)
```
- Print classification report.

```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
```
- Calculate ROC curve and AUC.

```python
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
```
- Plot ROC curve.

### 10. Making Predictions on New Data

```python
Xnew = np.array([[78, 1, 0, 68, 280, 1, 1, 195, 0, 0.2, 1, 2, 3]], dtype=np.float64)  # Example input for an unhealthy person
# Xnew = np.array([[18, 1, 0, 130, 253, 0, 1, 144, 1, 1.4, 2, 1, 3]], dtype=np.float64)  # Example input for a healthy person
```
- Provide input features for prediction.

```python
print("Number of features:", len(Xnew[0]))  # Should match the number of input features used in training
ynew = model.predict(Xnew)
print(f'Predicted Probability: {ynew[0][0]:.4f}')
prediction = (ynew > 0.5).astype(int)
print(f'Is person a heart patient? {"Yes" if prediction[0][0] == 1 else "No"}')
```
- Predict the probability of heart disease and determine the class based on the prediction.

## Conclusion

This code demonstrates the complete process of loading a dataset, performing exploratory data analysis, building and training a neural network, evaluating the model, and making predictions on new data. Each step is crucial for developing a robust heart disease prediction model.
