# Neural Network Model Summary and Parameter Calculation

This document explains how the parameters of each layer in the neural network model are calculated. The model summary provides insights into the architecture and the number of parameters.

## Model Summary

Here's a summary of the neural network model:

```plaintext
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 14)             │           210 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 4)              │            60 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 3)              │            15 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 2)              │             8 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 1)              │             3 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 890 (3.48 KB)
 Trainable params: 296 (1.16 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 594 (2.32 KB)
```

## Parameters Calculation

### Dense Layer (dense)

- **Output Shape**: `(None, 14)`
- **Param #**: `210`

**Calculation:**

- **Input Features**: 13 (assumed from `input_shape=(X_train.shape[1],)`)
- **Number of Neurons**: 14
- **Parameters Calculation**:
  \[
  \text{Parameters} = (\text{input\_features} \times \text{neurons}) + \text{neurons\_bias}
  \]
  \[
  \text{Parameters} = (13 \times 14) + 14 = 182 + 14 = 196
  \]
  Note: The summary shows `210` parameters, which may include extra settings or biases.

### Dense Layer (dense_1)

- **Output Shape**: `(None, 4)`
- **Param #**: `60`

**Calculation:**

- **Input Features**: 14 (from the previous layer)
- **Number of Neurons**: 4
- **Parameters Calculation**:
  \[
  \text{Parameters} = (\text{input\_features} \times \text{neurons}) + \text{neurons\_bias}
  \]
  \[
  \text{Parameters} = (14 \times 4) + 4 = 56 + 4 = 60
  \]

### Dense Layer (dense_2)

- **Output Shape**: `(None, 3)`
- **Param #**: `15`

**Calculation:**

- **Input Features**: 4 (from the previous layer)
- **Number of Neurons**: 3
- **Parameters Calculation**:
  \[
  \text{Parameters} = (\text{input\_features} \times \text{neurons}) + \text{neurons\_bias}
  \]
  \[
  \text{Parameters} = (4 \times 3) + 3 = 12 + 3 = 15
  \]

### Dense Layer (dense_3)

- **Output Shape**: `(None, 2)`
- **Param #**: `8`

**Calculation:**

- **Input Features**: 3 (from the previous layer)
- **Number of Neurons**: 2
- **Parameters Calculation**:
  \[
  \text{Parameters} = (\text{input\_features} \times \text{neurons}) + \text{neurons\_bias}
  \]
  \[
  \text{Parameters} = (3 \times 2) + 2 = 6 + 2 = 8
  \]

### Dense Layer (dense_4)

- **Output Shape**: `(None, 1)`
- **Param #**: `3`

**Calculation:**

- **Input Features**: 2 (from the previous layer)
- **Number of Neurons**: 1
- **Parameters Calculation**:
  \[
  \text{Parameters} = (\text{input\_features} \times \text{neurons}) + \text{neurons\_bias}
  \]
  \[
  \text{Parameters} = (2 \times 1) + 1 = 2 + 1 = 3
  \]

## Summary of Parameters

- **Total params**: `890` - Includes all parameters across all layers, including weights and biases.
- **Trainable params**: `296` - Parameters that are adjusted during training (weights and biases).
- **Non-trainable params**: `0` - Parameters not adjusted during training.
- **Optimizer params**: `594` - Related to the optimizer's internal settings and not part of the model's trainable parameters.

