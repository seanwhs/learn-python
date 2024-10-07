# ROC Curve and AUC

The ROC Curve and AUC are important metrics for evaluating the performance of a classification model, especially in binary classification problems. They help in understanding how well the model distinguishes between the positive and negative classes across different threshold values.

## Code

```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
```

## ROC Curve (Receiver Operating Characteristic Curve)

- **Definition**: The ROC Curve is a graphical representation that shows the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR) at various threshold settings.
- **Interpretation**: The curve plots the TPR against the FPR for different threshold values, helping to visualize the model’s performance across different levels of classification sensitivity and specificity.

### Metrics

1. **True Positive Rate (TPR)**
   - **Definition**: Also known as Recall, the TPR is the proportion of actual positive cases that are correctly identified by the model.
   - **Formula**: \[\text{TPR} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}\]
   - **Interpretation**: TPR indicates how well the model identifies positive cases. A higher TPR means that more actual positives are correctly identified.

2. **False Positive Rate (FPR)**
   - **Definition**: The FPR is the proportion of actual negative cases that are incorrectly classified as positive by the model.
   - **Formula**: \[\text{FPR} = \frac{\text{False Positives (FP)}}{\text{False Positives (FP)} + \text{True Negatives (TN)}}\]
   - **Interpretation**: FPR shows how often the model incorrectly predicts the positive class. A lower FPR means fewer actual negatives are incorrectly classified as positives.

## AUC (Area Under the Curve)

- **Definition**: AUC stands for Area Under the Curve. It represents the area under the ROC Curve and provides a single value that summarizes the model's performance across all threshold values.
- **Interpretation**: 
  - **AUC of 1**: Indicates a perfect model that can perfectly distinguish between the positive and negative classes.
  - **AUC of 0.5**: Indicates a model with no discrimination ability, similar to random guessing.
  - **Higher AUC**: Reflects better model performance in distinguishing between the classes.

## Summary

The ROC Curve provides a visual representation of the trade-off between sensitivity (TPR) and the rate of false alarms (FPR) for different threshold values. The AUC quantifies this performance into a single number, making it easier to compare different models. A higher AUC value indicates a better model performance across various threshold settings.
