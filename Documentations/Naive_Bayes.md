# Naive Bayes Classification - Probabilistic Approach

**Task 3: Naive Bayes Classification (10 Marks)**

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Gaussian Naive Bayes](#gaussian-naive-bayes)
4. [Implementation](#implementation)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results Interpretation](#results-interpretation)
7. [Strengths and Limitations](#strengths-and-limitations)

---

## Overview

Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem. It's called "naive" because it assumes feature independence, which rarely holds in practice but often works surprisingly well. The algorithm is fast, simple, and effective for many real-world applications.

### Why Use Naive Bayes?

✓ **Fast training and prediction** - One of the fastest algorithms  
✓ **Works well with small datasets** - Needs less training data  
✓ **Handles high dimensions** - Doesn't suffer from curse of dimensionality  
✓ **Probabilistic output** - Provides confidence scores  
✓ **Simple and interpretable** - Easy to understand and explain  

---

## Theoretical Foundation

### Bayes' Theorem

The foundation of Naive Bayes is **Bayes' Theorem:**

$$P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}$$

**In classification context:**

$$P(\text{Class}|\text{Features}) = \frac{P(\text{Features}|\text{Class}) \cdot P(\text{Class})}{P(\text{Features})}$$

**Breaking down the terms:**

1. **$P(Y|X)$** - **Posterior Probability**
   - Probability of class Y given features X
   - What we want to calculate
   - The answer to: "Given these features, what's the probability of this class?"

2. **$P(X|Y)$** - **Likelihood**
   - Probability of observing features X given class Y
   - Learned from training data
   - The answer to: "If this is class Y, how likely are these feature values?"

3. **$P(Y)$** - **Prior Probability**
   - Probability of class Y before seeing any features
   - Simply the proportion of each class in training data
   - Example: If 60% of training data is Class 1, then P(Class 1) = 0.6

4. **$P(X)$** - **Evidence**
   - Probability of observing features X
   - Same for all classes (normalizing constant)
   - Often ignored because we only compare probabilities

### The "Naive" Assumption

**Feature Independence Assumption:**

$$P(X|Y) = P(x_1, x_2, ..., x_n|Y) = P(x_1|Y) \cdot P(x_2|Y) \cdot ... \cdot P(x_n|Y)$$

**What this means:**
- Features are assumed to be conditionally independent given the class
- Knowing one feature's value doesn't tell us anything about other features
- **Rarely true in practice**, but algorithm works well anyway!

**Example:**
```
Classifying emails as spam/not spam:
- Feature 1: Contains "free"
- Feature 2: Contains "money"

Naive assumption: Presence of "free" doesn't affect probability of "money"
Reality: These words often appear together in spam!

Despite violation, Naive Bayes works well for spam filtering!
```

### Classification Decision

For each class, calculate:

$$\text{score}(\text{class}) = P(\text{class}) \cdot \prod_{i=1}^{n} P(x_i|\text{class})$$

**Predict:** Class with highest score

$$\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{n} P(x_i|y)$$

---

## Gaussian Naive Bayes

### When to Use Gaussian NB

Use **Gaussian Naive Bayes** when features are **continuous** (numerical) and approximately follow a **normal (Gaussian) distribution**.

### Mathematical Model

For continuous features, assume each feature follows a Gaussian distribution:

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

Where:
- $\mu_y$ = Mean of feature $x_i$ for class $y$
- $\sigma_y^2$ = Variance of feature $x_i$ for class $y$

**Training Process:**
1. For each class and each feature:
   - Calculate mean ($\mu$)
   - Calculate standard deviation ($\sigma$)
2. Store these parameters
3. Use them for prediction

**Prediction Process:**
1. For each class:
   - Calculate prior probability P(class)
   - For each feature:
     - Calculate likelihood using Gaussian formula
   - Multiply prior by all likelihoods
2. Choose class with highest probability

### Example Calculation

**Given:**
- Training data for Class 0: feature1 values [1, 2, 3]
- Training data for Class 1: feature1 values [7, 8, 9]

**Training:**
- Class 0: $\mu_0 = 2$, $\sigma_0 = 1$
- Class 1: $\mu_1 = 8$, $\sigma_1 = 1$
- Prior: P(Class 0) = 0.5, P(Class 1) = 0.5

**Predicting new point with feature1 = 2.5:**

```
P(Class 0 | 2.5) ∝ P(Class 0) × P(2.5 | Class 0)
                 = 0.5 × Gaussian(2.5, μ=2, σ=1)
                 = 0.5 × 0.35 = 0.175

P(Class 1 | 2.5) ∝ P(Class 1) × P(2.5 | Class 1)
                 = 0.5 × Gaussian(2.5, μ=8, σ=1)
                 = 0.5 × 0.00001 ≈ 0

Prediction: Class 0 (higher probability)
```

---

## Implementation

### Loading Preprocessed Data

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

def load_preprocessed_data():
    """Load the data we saved in preprocessing step"""
    
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test
```

### Training the Model

```python
def train_naive_bayes(X_train, y_train):
    """Train Gaussian Naive Bayes classifier"""
    
    # Initialize the classifier
    nb_classifier = GaussianNB()
    
    # Train the model
    nb_classifier.fit(X_train, y_train)
    
    print("✓ Gaussian Naive Bayes model trained!")
    
    # Display model information
    print(f"\nNumber of classes: {len(nb_classifier.classes_)}")
    print(f"Classes: {nb_classifier.classes_}")
    print(f"Number of features: {nb_classifier.theta_.shape[1]}")
    
    return nb_classifier
```

**What happens during training:**
1. Calculate class priors: P(Class 0), P(Class 1)
2. For each class and feature:
   - Calculate mean (theta_)
   - Calculate variance (sigma_)
3. Store parameters in model

### Making Predictions

```python
def predict_and_evaluate(classifier, X_test, y_test):
    """Make predictions on test set"""
    
    # Get predictions
    y_pred = classifier.predict(X_test)
    
    # Get probability estimates
    y_proba = classifier.predict_proba(X_test)
    
    print(f"Predictions made for {len(y_pred)} test samples")
    
    return y_pred, y_proba
```

**Prediction process:**
1. For each test sample:
   - Calculate P(Class 0 | features)
   - Calculate P(Class 1 | features)
   - Choose class with higher probability

---

## Evaluation Metrics

### Why Multiple Metrics?

**Accuracy alone is insufficient!** Especially with imbalanced datasets.

Example:
```
Dataset: 95% Class 0, 5% Class 1
Dummy classifier that always predicts Class 0:
Accuracy = 95% (seems great!)
But: Never correctly identifies Class 1 (useless!)
```

### Confusion Matrix

**Structure:**
```
                Predicted
                0      1
Actual    0    TN     FP
          1    FN     TP
```

**Definitions:**
- **TN (True Negative):** Correctly predicted Class 0
- **FP (False Positive):** Incorrectly predicted Class 1 (Type I error)
- **FN (False Negative):** Incorrectly predicted Class 0 (Type II error)
- **TP (True Positive):** Correctly predicted Class 1

### Metric Formulas

**1. Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Meaning:** Overall correctness
- **Range:** 0 to 1 (0% to 100%)
- **Use when:** Classes are balanced
- **Limitation:** Misleading with imbalanced classes

**2. Precision**

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Meaning:** Of all positive predictions, how many were correct?
- **Question:** "When model says positive, how often is it right?"
- **High precision:** Few false alarms
- **Important when:** False positives are costly

**Example:** Medical test
- High precision: When test says "disease," patient probably has it
- Low precision: Many false alarms, unnecessary treatments

**3. Recall (Sensitivity)**

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **Meaning:** Of all actual positives, how many did we find?
- **Question:** "Of all positive cases, how many did we catch?"
- **High recall:** Few missed positives
- **Important when:** False negatives are costly

**Example:** Cancer screening
- High recall: Catch most cancer cases (few missed)
- Low recall: Miss many cancer cases (dangerous!)

**4. F1-Score**

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Meaning:** Harmonic mean of precision and recall
- **Balances** both metrics
- **Use when:** Need balance between precision and recall
- **Range:** 0 to 1

### Metric Tradeoff

**Precision vs Recall Tradeoff:**

```
High Threshold → More confident predictions
├─ Higher Precision (fewer false positives)
└─ Lower Recall (more false negatives)

Low Threshold → More liberal predictions
├─ Lower Precision (more false positives)
└─ Higher Recall (fewer false negatives)
```

**Which to optimize?**

| Scenario | Optimize | Why |
|----------|----------|-----|
| Spam Filter | Precision | Don't want false positives (legitimate emails marked spam) |
| Disease Screening | Recall | Don't want false negatives (miss sick patients) |
| Fraud Detection | Recall | Must catch fraudulent transactions |
| Recommendation | Balance (F1) | Don't want too many bad recommendations or miss good ones |

### Implementation

```python
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, 
                            classification_report, confusion_matrix)

def calculate_metrics(y_test, y_pred):
    """Calculate all evaluation metrics"""
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print("EVALUATION METRICS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Detailed report
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, 
                               target_names=['Class 0', 'Class 1']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX")
    print(cm)
    
    return accuracy, precision, recall, f1, cm
```

---

## Results Interpretation

### What Makes Good Performance?

**General Guidelines:**

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Accuracy | <70% | 70-80% | 80-90% | >90% |
| Precision | <70% | 70-80% | 80-90% | >90% |
| Recall | <70% | 70-80% | 80-90% | >90% |
| F1-Score | <70% | 70-80% | 80-90% | >90% |

**Note:** These are general guidelines. Acceptable performance depends on:
- Domain (medical diagnosis requires higher accuracy)
- Baseline (random guessing = 50% for binary)
- Business requirements

### Interpretation Template

```python
def interpret_results(accuracy, precision, recall, f1, cm):
    """Provide interpretation of results"""
    
    interpretation = f"""
    NAIVE BAYES PERFORMANCE ANALYSIS:
    
    1. OVERALL ACCURACY: {accuracy*100:.2f}%
       - The model correctly classifies {accuracy*100:.2f}% of all instances
       - Performance level: {'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Moderate'}
    
    2. PRECISION: {precision*100:.2f}%
       - When model predicts positive, it's correct {precision*100:.2f}% of time
       - False positive rate: {(1-precision)*100:.2f}%
       - {'Low false alarms' if precision > 0.85 else 'Moderate false alarms'}
    
    3. RECALL: {recall*100:.2f}%
       - Model catches {recall*100:.2f}% of all positive cases
       - Miss rate: {(1-recall)*100:.2f}%
       - {'Low miss rate' if recall > 0.85 else 'Moderate miss rate'}
    
    4. F1-SCORE: {f1*100:.2f}%
       - Balanced performance measure
       - {'Well-balanced' if abs(precision-recall) < 0.1 else 'Some imbalance'}
    
    5. CONFUSION MATRIX:
       True Negatives: {cm[0][0]} - Correctly identified Class 0
       False Positives: {cm[0][1]} - Wrongly predicted Class 1
       False Negatives: {cm[1][0]} - Wrongly predicted Class 0
       True Positives: {cm[1][1]} - Correctly identified Class 1
    """
    
    print(interpretation)
```

### Model Insights

**Why Naive Bayes Performs Well/Poorly:**

**Performs Well When:**
- Features approximately independent
- Features follow Gaussian distributions
- Classes are well-separated in feature space
- Dataset is small to medium size

**Performs Poorly When:**
- Strong feature correlations exist
- Features have complex distributions
- Classes heavily overlap
- Non-linear decision boundaries needed

---

## Strengths and Limitations

### Strengths

✓ **Speed**
- Very fast training: O(n × d)
- Very fast prediction: O(d)
- where n = samples, d = features

✓ **Simplicity**
- Easy to understand
- Easy to implement
- Easy to interpret

✓ **Data Efficiency**
- Works with small datasets
- Needs less training data than discriminative models

✓ **High Dimensions**
- Doesn't suffer from curse of dimensionality
- Works well with many features

✓ **Probabilistic**
- Provides probability estimates
- Can quantify confidence
- Useful for decision-making

✓ **Robust**
- Handles irrelevant features well
- Not prone to overfitting
- Naturally handles missing data

### Limitations

✗ **Independence Assumption**
- Features rarely independent in practice
- Can't capture feature interactions
- May underperform when assumption violated

✗ **Distribution Assumption**
- Assumes Gaussian distributions
- Reality often differs
- May need feature transformation

✗ **Zero Probability Problem**
- If feature value never seen in training for a class
- Probability becomes zero (multiplicative effect)
- Solution: Laplace smoothing

✗ **Decision Boundaries**
- Only linear (in probability space)
- Can't capture complex boundaries
- Struggles with non-linear patterns

✗ **Correlated Features**
- Violates independence assumption
- Over-weights correlated features
- May need feature selection

---

## Complete Implementation

```python
"""
Task 3: Naive Bayes Classification
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train model
    nb_classifier = train_naive_bayes(X_train, y_train)
    
    # Make predictions
    y_pred = nb_classifier.predict(X_test)
    
    # Evaluate
    accuracy, precision, recall, f1, cm = calculate_metrics(y_test, y_pred)
    
    # Interpret
    interpret_results(accuracy, precision, recall, f1, cm)
    
    # Save model
    import pickle
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)
    
    print("\n✓ Naive Bayes classification complete!")

if __name__ == "__main__":
    main()
```

---

## Key Takeaways

✓ **Naive Bayes is probabilistic** - Based on Bayes' Theorem  
✓ **"Naive" means independence** - Assumes features are independent  
✓ **Gaussian NB for continuous features** - Assumes normal distributions  
✓ **Fast and efficient** - Excellent for real-time applications  
✓ **Multiple metrics needed** - Accuracy alone is insufficient  
✓ **Precision vs Recall tradeoff** - Balance depends on application  
✓ **Works surprisingly well** - Despite strong assumptions  

---

**File Reference:** `3_naive_bayes.py`  
**Previous:** [Preprocessing.md](Preprocessing.md)  
**Next:** [KNN_Classification.md](KNN_Classification.md)
