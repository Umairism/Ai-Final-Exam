"""
Task 3: Naive Bayes Classification (10 Marks)
- Train a Gaussian Naive Bayes classifier
- Predict class labels for test set
- Evaluate using Accuracy, Precision, Recall, F1-score
- Interpret the results
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

def load_preprocessed_data():
    """Load the preprocessed data"""
    
    print("=" * 60)
    print("LOADING PREPROCESSED DATA")
    print("=" * 60)
    
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"\n✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Testing set: {X_test.shape[0]} samples")
    print(f"✓ Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def train_naive_bayes(X_train, y_train):
    """Train Gaussian Naive Bayes classifier"""
    
    print("\n" + "=" * 60)
    print("TRAINING GAUSSIAN NAIVE BAYES CLASSIFIER")
    print("=" * 60)
    
    # Initialize the Gaussian Naive Bayes classifier
    nb_classifier = GaussianNB()
    
    # Train the classifier
    print("\nTraining the model...")
    nb_classifier.fit(X_train, y_train)
    
    print("✓ Gaussian Naive Bayes model trained successfully!")
    
    # Display model parameters
    print("\nModel Information:")
    print("-" * 60)
    print(f"Number of classes: {len(nb_classifier.classes_)}")
    print(f"Classes: {nb_classifier.classes_}")
    print(f"Number of features: {nb_classifier.theta_.shape[1]}")
    
    return nb_classifier

def predict_and_evaluate(classifier, X_test, y_test, model_name="Naive Bayes"):
    """Make predictions and evaluate the classifier"""
    
    print("\n" + "=" * 60)
    print(f"PREDICTING AND EVALUATING {model_name.upper()}")
    print("=" * 60)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Display evaluation metrics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Display classification report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['Class 0', 'Class 1']))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name)
    
    return y_pred, accuracy, precision, recall, f1

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved as '{filename}'")
    plt.show()

def interpret_results(accuracy, precision, recall, f1, cm):
    """Interpret the Naive Bayes results"""
    
    print("\n" + "=" * 60)
    print("INTERPRETATION OF RESULTS")
    print("=" * 60)
    
    interpretation = f"""
    NAIVE BAYES PERFORMANCE ANALYSIS:
    
    1. OVERALL ACCURACY: {accuracy*100:.2f}%
       - The model correctly classifies {accuracy*100:.2f}% of all instances
       - {'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Moderate' if accuracy > 0.7 else 'Needs improvement'} performance
    
    2. PRECISION: {precision*100:.2f}%
       - Out of all instances predicted as positive (Class 1), {precision*100:.2f}% are truly positive
       - {'Low false positive rate' if precision > 0.85 else 'Moderate false positive rate' if precision > 0.7 else 'High false positive rate'}
       - Important when the cost of false positives is high
    
    3. RECALL (SENSITIVITY): {recall*100:.2f}%
       - Out of all actual positive instances, the model correctly identifies {recall*100:.2f}%
       - {'Low false negative rate' if recall > 0.85 else 'Moderate false negative rate' if recall > 0.7 else 'High false negative rate'}
       - Important when the cost of missing positive cases is high
    
    4. F1-SCORE: {f1*100:.2f}%
       - Harmonic mean of precision and recall
       - Provides a balanced measure of model performance
       - {'Well-balanced' if abs(precision - recall) < 0.1 else 'Some imbalance between precision and recall'}
    
    5. CONFUSION MATRIX ANALYSIS:
       - True Negatives (TN): {cm[0][0]} - Correctly predicted Class 0
       - False Positives (FP): {cm[0][1]} - Incorrectly predicted as Class 1
       - False Negatives (FN): {cm[1][0]} - Incorrectly predicted as Class 0
       - True Positives (TP): {cm[1][1]} - Correctly predicted Class 1
    
    6. NAIVE BAYES CHARACTERISTICS:
       ✓ Fast training and prediction
       ✓ Works well with small to medium datasets
       ✓ Assumes feature independence (may not hold in practice)
       ✓ Probabilistic approach provides confidence scores
       ✓ Less sensitive to feature scaling than distance-based methods
       
    7. KEY INSIGHTS:
       - {'The model shows strong performance across all metrics' if min(precision, recall) > 0.8 else 'The model shows moderate performance'}
       - {'Precision and recall are well-balanced' if abs(precision - recall) < 0.1 else f'{"Precision" if precision > recall else "Recall"} is notably {"higher" if precision > recall else "lower"}'}
       - Gaussian Naive Bayes assumes features follow normal distribution
       - Independence assumption may affect performance if features are correlated
    """
    
    print(interpretation)

def save_model(classifier):
    """Save the trained model"""
    
    import pickle
    
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print("\n✓ Naive Bayes model saved as 'naive_bayes_model.pkl'")

def main():
    """Main execution function"""
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train Naive Bayes classifier
    nb_classifier = train_naive_bayes(X_train, y_train)
    
    # Predict and evaluate
    y_pred, accuracy, precision, recall, f1 = predict_and_evaluate(
        nb_classifier, X_test, y_test, "Naive Bayes"
    )
    
    # Get confusion matrix for interpretation
    cm = confusion_matrix(y_test, y_pred)
    
    # Interpret results
    interpret_results(accuracy, precision, recall, f1, cm)
    
    # Save model
    save_model(nb_classifier)
    
    # Save results for comparison
    results = {
        'model': 'Naive Bayes',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    np.save('naive_bayes_results.npy', results)
    
    print("\n" + "=" * 60)
    print("NAIVE BAYES CLASSIFICATION COMPLETE!")
    print("=" * 60)
    print("\nNext Step: Proceed to KNN classification (4_knn_classification.py)")

if __name__ == "__main__":
    main()
