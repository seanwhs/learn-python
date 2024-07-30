 Let's dive deeper into the sections of **Model Selection** and **Model Building**.

### Model Selection

In machine learning, selecting an appropriate model involves choosing the algorithm or ensemble of algorithms that best fits the data and the problem at hand. Here's how the provided code handles model selection:

#### 1. Define Model Parameters

```python
model_param = {
    'RandomForestClassifier':{
        'model':RandomForestClassifier(),
        'param':{
            'n_estimators': [10, 50, 100, 130], 
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2, 4, 1), 
            'max_features': ['auto', 'log2']
        }
    },
    'XGBClassifier':{
        'model':XGBClassifier(objective='binary:logistic'),
        'param':{
           'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]
        }
    }
}
```

- **`model_param` Dictionary**: Contains configurations for two models:
  - **RandomForestClassifier**
  - **XGBClassifier (XGBoost)**

- For each model:
  - **`model`**: Defines the actual model object (`RandomForestClassifier()` and `XGBClassifier(objective='binary:logistic')`).
  - **`param`**: Specifies a dictionary of parameters to be tested using `GridSearchCV` for hyperparameter tuning.

#### 2. Hyperparameter Tuning using GridSearchCV

```python
scores = []
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'], param_grid=mp['param'], cv=5, return_train_score=False)
    model_selection.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })
```

- **`GridSearchCV`**: This function exhaustively searches over specified parameter values for an estimator, evaluating them via cross-validation.

- **Steps in GridSearchCV Process**:
  - **`estimator=mp['model']`**: Specifies the model to be tuned.
  - **`param_grid=mp['param']`**: Sets the parameters to be tuned and their possible values.
  - **`cv=5`**: Uses 5-fold cross-validation during the tuning process.
  - **`return_train_score=False`**: Specifies not to return the training scores for each parameter setting.

- **`model_selection.fit(X, y)`**: Fits the GridSearchCV object on the entire dataset (`X` for features, `y` for target).

- **`scores.append(...)`**: Stores the results of each model's best score and best parameters in the `scores` list.

### Model Building

Once the best model (in this case, XGBoost as per the results) is selected through the above process, the next step is to build and evaluate the model using the best parameters obtained:

```python
# Instantiate XGBClassifier with best parameters
model_xgb = XGBClassifier(objective='binary:logistic', learning_rate=0.1, max_depth=20, n_estimators=200)

# Train the model on the training data
model_xgb.fit(X_train, y_train)
```

- **Instantiate the Model**: Create an instance of `XGBClassifier` with the best parameters obtained from `GridSearchCV`.

- **Training the Model**: Use the `.fit()` method to train the model on the training dataset (`X_train` for features and `y_train` for target).

### Summary

- **Model Selection**: Involves defining potential models, specifying parameter grids for hyperparameter tuning, and using `GridSearchCV` to find the best combination of parameters.
  
- **Model Building**: Once the best model and parameters are determined, the model is instantiated with those parameters and trained on the training dataset.

These steps ensure that the final model is optimized for the given dataset and problem, maximizing performance metrics such as accuracy, precision, recall, etc., as seen in the confusion matrix and other evaluation metrics in the provided code.