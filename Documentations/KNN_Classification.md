# K-Nearest Neighbors (KNN) Classification - Instance-Based Learning

**Task 4: K-Nearest Neighbors Classification (10 Marks)**

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Algorithm Mechanics](#algorithm-mechanics)
4. [Implementation](#implementation)
5. [K-Value Analysis](#k-value-analysis)
6. [Performance Evaluation](#performance-evaluation)
7. [Comparison with Naive Bayes](#comparison-with-naive-bayes)

---

## Overview

K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm that makes predictions based on the similarity to training examples. Unlike Naive Bayes, KNN makes no assumptions about the underlying data distribution, making it highly flexible but computationally expensive.

### Key Characteristics

✓ **Non-parametric** - No assumptions about data distribution  
✓ **Instance-based** - Stores all training data  
✓ **Lazy learning** - No training phase, all work at prediction  
✓ **Distance-based** - Uses similarity metrics  
✓ **Flexible** - Can capture complex decision boundaries  

---

## Theoretical Foundation

### Core Concept

**The Intuition:**
> "Similar instances belong to similar classes"

KNN operates on a simple principle: If you want to classify a new data point, look at its **k nearest neighbors** in the training set and assign the majority class.

### Mathematical Formulation

**For a new point $x$:**

1. **Calculate distances** to all training points:
   $$d(x, x_i) = \text{distance}(x, x_i)$$

2. **Find k nearest neighbors:**
   $$N_k(x) = \{k \text{ points with smallest distances}\}$$

3. **Majority vote:**
   $$\hat{y} = \arg\max_{c} \sum_{x_i \in N_k(x)} \mathbb{1}(y_i = c)$$

Where $\mathbb{1}$ is the indicator function (1 if true, 0 if false).

### Distance Metrics

KNN can use various distance metrics. The most common is **Euclidean distance**.

#### Euclidean Distance

**Formula (2D):**
$$d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2}$$

**General formula (n dimensions):**
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

**Example:**
```
Point A: [3, 4]
Point B: [6, 8]

Distance = √[(3-6)² + (4-8)²]
        = √[(-3)² + (-4)²]
        = √[9 + 16]
        = √25
        = 5
```

#### Why Euclidean Distance?

✓ **Intuitive** - Straight-line distance  
✓ **Works well** for continuous features  
✓ **Standard choice** for most applications  
✓ **Geometrically meaningful** in feature space  

**Other distance metrics:**
- **Manhattan Distance:** Sum of absolute differences (good for grid-like data)
- **Minkowski Distance:** Generalization of Euclidean and Manhattan
- **Cosine Similarity:** Good for text and high-dimensional data
- **Hamming Distance:** For categorical features

### Why Feature Scaling is MANDATORY

**Critical Point:** KNN is **extremely sensitive** to feature scales!

**Example demonstrating the problem:**

```python
# Without Normalization
Feature 1 (Age): ranges 0-100
Feature 2 (Salary): ranges 0-100,000

Person A: [25, 50000]
Person B: [26, 50000]  # Similar age, same salary
Person C: [25, 55000]  # Same age, different salary

Distance(A, B) = √[(25-26)² + (50000-50000)²] = √1 = 1
Distance(A, C) = √[(25-25)² + (50000-55000)²] = √25,000,000 = 5000

KNN thinks B is 5000x closer to A than C is!
But age difference is actually meaningful too!
```

**With Standardization:**
```python
# After Standardization (mean=0, std=1)
Both features contribute proportionally!

Distance(A, B) ≈ small (considers age difference)
Distance(A, C) ≈ moderate (considers salary difference)

Fair comparison! ✓
```

**Why this matters:**
- Large-scale features dominate distance calculations
- Small-scale features become irrelevant
- Model effectively ignores some features
- **Performance degrades catastrophically**

**Solution:** **Always standardize features before KNN!**

---

## Algorithm Mechanics

### Training Phase

**Spoiler:** There isn't one!

```python
def train(X_train, y_train):
    # Store the data
    self.X_train = X_train
    self.y_train = y_train
    # That's it! No computation!
```

**Why no training?**
- KNN is a **lazy learner**
- Simply stores training data
- All computation happens during prediction
- No parameters to learn

**Advantages:**
✓ Instant "training"
✓ Can add new data without retraining
✓ Always uses latest data

**Disadvantages:**
✗ Must store entire dataset (memory intensive)
✗ Slow predictions (must compute all distances)
✗ Doesn't scale well to large datasets

### Prediction Phase

**For each test point:**

```python
def predict(x_test, X_train, y_train, k):
    # Step 1: Calculate distances to all training points
    distances = []
    for x_train in X_train:
        dist = euclidean_distance(x_test, x_train)
        distances.append(dist)
    
    # Step 2: Find k nearest neighbors (smallest distances)
    k_indices = argsort(distances)[:k]
    k_nearest_labels = y_train[k_indices]
    
    # Step 3: Majority vote
    predicted_class = most_common(k_nearest_labels)
    
    return predicted_class
```

**Computational Complexity:**
- **Training:** O(1) - just storage
- **Prediction:** O(n × d) per sample
  - n = number of training samples
  - d = number of features
- **Total for m predictions:** O(m × n × d)

**For large datasets:** Very slow!

### Visual Example

```
Training Data:           Test Point:
● Class 0                ★ Unknown
■ Class 1

Feature 2
    ↑
    |  ●    ■
    |    ●  ■  ★
    |  ●  ●    ■
    |    ●  ■
    └──────────→ Feature 1

k=3: Find 3 nearest neighbors to ★
Neighbors: ■ ■ ●
Votes: Class 1: 2, Class 0: 1
Prediction: Class 1 (majority)
```

---

## Implementation

### Loading Preprocessed Data

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def load_preprocessed_data():
    """Load standardized data from preprocessing step"""
    
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test
```

**Note:** Data must be standardized! This was done in preprocessing.

### Training KNN (k=3)

```python
def train_knn(X_train, y_train, k=3):
    """Initialize and 'train' KNN classifier"""
    
    # Initialize KNN with k neighbors
    knn_classifier = KNeighborsClassifier(
        n_neighbors=k,          # Number of neighbors
        metric='euclidean',     # Distance metric
        weights='uniform'       # All neighbors weighted equally
    )
    
    # 'Train' (just stores the data)
    knn_classifier.fit(X_train, y_train)
    
    print(f"✓ KNN model (k={k}) initialized!")
    print(f"  Distance metric: Euclidean")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    
    return knn_classifier
```

**Parameters:**
- **n_neighbors (k):** Number of neighbors to consider
- **metric:** Distance function (euclidean, manhattan, etc.)
- **weights:** How to weight neighbors
  - 'uniform': All neighbors equally important
  - 'distance': Closer neighbors more important

### Making Predictions

```python
def predict_and_evaluate(classifier, X_test, y_test, k_value):
    """Make predictions and calculate metrics"""
    
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\nKNN (k={k_value}) RESULTS:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    return y_pred, accuracy, precision, recall, f1
```

---

## K-Value Analysis

### What is k?

**k** is the number of nearest neighbors to consider when making a prediction.

**k=1:** Look at only the single nearest neighbor  
**k=3:** Look at three nearest neighbors, majority vote  
**k=5:** Look at five nearest neighbors, majority vote  

### Bias-Variance Tradeoff

The choice of k involves a fundamental tradeoff in machine learning.

#### Small k (e.g., k=1, k=3)

**Characteristics:**
- High variance, low bias
- Very sensitive to individual data points
- Complex, irregular decision boundaries
- Captures local patterns well

**Advantages:**
✓ Flexible decision boundaries
✓ Can capture fine-grained patterns
✓ Good for data with clear local structure

**Disadvantages:**
✗ Sensitive to noise and outliers
✗ Prone to overfitting
✗ May capture random fluctuations
✗ Less stable predictions

**Visual:**
```
Decision boundary with k=1:
Extremely jagged, follows every point
```

#### Large k (e.g., k=7, k=11, k=21)

**Characteristics:**
- Low variance, high bias
- Less sensitive to individual points
- Smooth, regular decision boundaries
- Averages over larger neighborhoods

**Advantages:**
✓ More stable predictions
✓ Robust to noise and outliers
✓ Better generalization
✓ Smoother decision boundaries

**Disadvantages:**
✗ May be too simple
✗ Prone to underfitting
✗ May miss local patterns
✗ Decision boundary too smooth

**Visual:**
```
Decision boundary with k=21:
Very smooth, ignores minor variations
```

#### Optimal k

**Finding the sweet spot:**
```
k too small → Overfitting
k too large → Underfitting
k optimal  → Best generalization
```

### Comparing k=3, k=5, k=7

```python
def compare_k_values(X_train, X_test, y_train, y_test, 
                     k_values=[3, 5, 7]):
    """Compare performance across different k values"""
    
    results = []
    
    for k in k_values:
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        
        # Evaluate
        y_pred = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'k': k,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        print(f"\nk={k}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(results_df.to_string(index=False))
    
    return results_df
```

### Visualization

```python
def plot_k_comparison(results_df):
    """Visualize performance across k values"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = idx // 2, idx % 2
        
        axes[row, col].plot(results_df['k'], results_df[metric],
                           marker='o', linewidth=2, markersize=10)
        axes[row, col].set_xlabel('k value', fontweight='bold')
        axes[row, col].set_ylabel(title, fontweight='bold')
        axes[row, col].set_title(f'{title} vs k')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xticks(results_df['k'])
        
        # Annotate values
        for k, val in zip(results_df['k'], results_df[metric]):
            axes[row, col].annotate(f'{val:.3f}',
                                   xy=(k, val),
                                   xytext=(0, 10),
                                   textcoords='offset points',
                                   ha='center')
    
    plt.tight_layout()
    plt.savefig('knn_k_comparison.png', dpi=300)
    plt.show()
```

### Impact of Increasing k

**Observed Patterns:**

1. **Accuracy may increase then decrease:**
   - Small k: Overfitting, lower accuracy
   - Optimal k: Best accuracy
   - Large k: Underfitting, lower accuracy

2. **Stability increases:**
   - Predictions become more stable
   - Less affected by individual points
   - More consistent results

3. **Decision boundary smooths:**
   - Transitions from jagged to smooth
   - Local details lost
   - Global patterns emphasized

**Dataset-Specific Observations:**

```
If accuracy increases with k:
→ Dataset has noisy training data
→ Smaller k was overfitting
→ Larger k provides better generalization

If accuracy decreases with k:
→ Local patterns are important
→ Smaller k captures necessary detail
→ Larger k over-smooths

If accuracy peaks at middle k:
→ Sweet spot between bias and variance
→ Optimal balance for this dataset
```

### Choosing Optimal k

**Rule of Thumb:**
$$k_{optimal} \approx \sqrt{n}$$

Where n = number of training samples.

**Better Approach: Cross-Validation**

```python
from sklearn.model_selection import cross_val_score

# Test multiple k values
k_range = range(1, 21, 2)  # Odd numbers only
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, 
                            cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find best k
best_k = k_range[np.argmax(cv_scores)]
print(f"Optimal k: {best_k}")
```

**Why odd k for binary classification?**
- Avoids ties in voting
- k=3: Can't have 1.5 votes per class
- k=4: Could have 2-2 tie!

---

## Performance Evaluation

### Evaluation Metrics (Same as Naive Bayes)

KNN uses the same evaluation framework:
- **Accuracy:** Overall correctness
- **Precision:** Correctness of positive predictions
- **Recall:** Coverage of actual positives
- **F1-Score:** Harmonic mean of precision and recall

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(cm, k_value):
    """Visualize confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix - KNN (k={k_value})', 
             fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.savefig(f'confusion_matrix_knn_k{k_value}.png', dpi=300)
    plt.show()
```

---

## Comparison with Naive Bayes

### Key Differences

| Aspect | Naive Bayes | KNN |
|--------|-------------|-----|
| **Type** | Probabilistic | Instance-based |
| **Assumptions** | Feature independence | None |
| **Training** | Fast (learns parameters) | None (stores data) |
| **Prediction** | Very fast | Slow (compute distances) |
| **Memory** | Low (stores parameters) | High (stores all data) |
| **Scaling** | Not required | **MANDATORY** |
| **Decision Boundary** | Linear (probabilistic) | Flexible (non-linear) |
| **Interpretability** | High | Moderate |
| **Overfitting Risk** | Low | Moderate (small k) |

### When to Use Each

**Use Naive Bayes when:**
- Speed is critical
- Dataset is small
- Features are approximately independent
- Probabilistic interpretation needed
- Real-time predictions required

**Use KNN when:**
- Decision boundaries are complex
- Features are scaled/normalized
- No assumptions about data distribution
- Local patterns are important
- Can afford computational cost

---

## Complete Implementation

```python
"""
Task 4: K-Nearest Neighbors Classification
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train KNN with k=3
    knn_k3 = train_knn(X_train, y_train, k=3)
    
    # Evaluate k=3
    y_pred, acc, prec, rec, f1 = predict_and_evaluate(
        knn_k3, X_test, y_test, k_value=3
    )
    
    # Compare k values
    results_df = compare_k_values(
        X_train, X_test, y_train, y_test, 
        k_values=[3, 5, 7]
    )
    
    # Visualize comparison
    plot_k_comparison(results_df)
    
    # Save best model
    best_k = results_df.loc[results_df['accuracy'].idxmax(), 'k']
    best_knn = KNeighborsClassifier(n_neighbors=int(best_k))
    best_knn.fit(X_train, y_train)
    
    import pickle
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(best_knn, f)
    
    print(f"\n✓ Best KNN model (k={best_k}) saved!")

if __name__ == "__main__":
    main()
```

---

## Key Takeaways

✓ **KNN is instance-based** - Uses training examples directly  
✓ **No training phase** - Lazy learning algorithm  
✓ **k controls complexity** - Small k: complex, Large k: simple  
✓ **Feature scaling is MANDATORY** - Critical for distance calculations  
✓ **Computationally expensive** - Slow predictions on large datasets  
✓ **Flexible decision boundaries** - Can capture non-linear patterns  
✓ **Bias-variance tradeoff** - Choose k carefully  
✓ **Use cross-validation** - Find optimal k systematically  

---

**File Reference:** `4_knn_classification.py`  
**Previous:** [Naive_Bayes.md](Naive_Bayes.md)  
**Next:** [Comparison_Analysis.md](Comparison_Analysis.md)
