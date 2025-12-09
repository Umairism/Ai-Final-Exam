"""
Task 4: K-Nearest Neighbors (KNN) Classification (10 Marks)
- Train KNN model with k=3 using Euclidean distance
- Evaluate using same metrics as Naive Bayes
- Compare KNN performance for k=3, 5, 7
- Discuss how increasing k affects performance
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

def train_knn(X_train, y_train, k=3):
    """Train KNN classifier with specified k value"""
    
    print("\n" + "=" * 60)
    print(f"TRAINING KNN CLASSIFIER (k={k})")
    print("=" * 60)
    
    # Initialize KNN classifier with Euclidean distance (default metric)
    knn_classifier = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',  # Euclidean distance
        weights='uniform'    # All neighbors weighted equally
    )
    
    # Train the classifier
    print(f"\nTraining KNN model with k={k}...")
    knn_classifier.fit(X_train, y_train)
    
    print(f"✓ KNN model (k={k}) trained successfully!")
    
    # Display model parameters
    print("\nModel Information:")
    print("-" * 60)
    print(f"Number of neighbors (k): {k}")
    print(f"Distance metric: Euclidean")
    print(f"Weight function: Uniform")
    print(f"Number of training samples: {X_train.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    return knn_classifier

def predict_and_evaluate(classifier, X_test, y_test, k_value):
    """Make predictions and evaluate the classifier"""
    
    print("\n" + "=" * 60)
    print(f"PREDICTING AND EVALUATING KNN (k={k_value})")
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
    print(f"EVALUATION METRICS (k={k_value})")
    print("=" * 60)
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Display classification report
    print("\n" + "=" * 60)
    print(f"DETAILED CLASSIFICATION REPORT (k={k_value})")
    print("=" * 60)
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['Class 0', 'Class 1']))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, accuracy, precision, recall, f1, cm

def plot_confusion_matrix(cm, k_value):
    """Plot confusion matrix for KNN"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - KNN (k={k_value})', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    filename = f'confusion_matrix_knn_k{k_value}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved as '{filename}'")
    plt.show()

def compare_k_values(X_train, X_test, y_train, y_test, k_values=[3, 5, 7]):
    """Compare KNN performance for different k values"""
    
    print("\n" + "=" * 60)
    print("COMPARING KNN PERFORMANCE FOR DIFFERENT K VALUES")
    print("=" * 60)
    
    results = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing with k={k}")
        print('='*60)
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        results.append({
            'k': k,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        print(f"\nResults for k={k}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("\n", results_df.to_string(index=False))
    
    # Plot comparison
    plot_k_comparison(results_df)
    
    return results_df

def plot_k_comparison(results_df):
    """Plot comparison of metrics across different k values"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].plot(results_df['k'], results_df[metric], 
                           marker='o', linewidth=2, markersize=10, 
                           color=color, label=title)
        axes[row, col].set_xlabel('k value', fontsize=12, fontweight='bold')
        axes[row, col].set_ylabel(title, fontsize=12, fontweight='bold')
        axes[row, col].set_title(f'{title} vs k', fontsize=13, fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xticks(results_df['k'])
        
        # Add value labels on points
        for k, val in zip(results_df['k'], results_df[metric]):
            axes[row, col].annotate(f'{val:.3f}', 
                                   xy=(k, val), 
                                   textcoords="offset points",
                                   xytext=(0,10), 
                                   ha='center',
                                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig('knn_k_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ K-value comparison plot saved as 'knn_k_comparison.png'")
    plt.show()

def discuss_k_impact(results_df):
    """Discuss how increasing k affects performance"""
    
    print("\n" + "=" * 60)
    print("DISCUSSION: IMPACT OF INCREASING K")
    print("=" * 60)
    
    # Calculate trends
    k_values = results_df['k'].tolist()
    accuracies = results_df['accuracy'].tolist()
    
    discussion = f"""
    HOW INCREASING K AFFECTS KNN PERFORMANCE:
    
    1. BIAS-VARIANCE TRADEOFF:
       
       Small k (e.g., k=1, 3):
       ✓ Low bias - Model is more sensitive to local patterns
       ✗ High variance - More susceptible to noise and outliers
       ✗ Decision boundaries are irregular and complex
       ✗ Risk of overfitting
       
       Large k (e.g., k=7, 9, 11):
       ✗ High bias - Model averages over more neighbors
       ✓ Low variance - More stable and less affected by noise
       ✓ Decision boundaries are smoother
       ✓ Risk of underfitting if k is too large
    
    2. OBSERVED RESULTS IN THIS DATASET:
       
       k=3: Accuracy = {accuracies[0]:.4f}
       k=5: Accuracy = {accuracies[1]:.4f}
       k=7: Accuracy = {accuracies[2]:.4f}
       
       {'Accuracy increases with k' if accuracies[2] > accuracies[0] else 'Accuracy decreases with k' if accuracies[2] < accuracies[0] else 'Accuracy remains stable'}
       {'- Suggests the dataset benefits from smoother decision boundaries' if accuracies[2] > accuracies[0] else '- Suggests local patterns are important'}
    
    3. COMPUTATIONAL CONSIDERATIONS:
       - Larger k requires more distance calculations
       - Training time: O(1) - just stores data
       - Prediction time: O(n*d) where n=training samples, d=features
       - Increasing k doesn't significantly affect computation
    
    4. DECISION BOUNDARY SMOOTHNESS:
       - Small k: Complex, jagged boundaries (can capture local patterns)
       - Large k: Smooth, generalized boundaries (better for noisy data)
       - Optimal k depends on data complexity and noise level
    
    5. CLASS IMBALANCE CONSIDERATION:
       - If classes are imbalanced, larger k may bias toward majority class
       - Can use weighted voting to address this
    
    6. RULE OF THUMB:
       - Often k = √n (where n is training samples) is a good starting point
       - For this dataset: √{len(accuracies)} ≈ {int(np.sqrt(len(accuracies)))}
       - Should use cross-validation to find optimal k
       - k should be odd to avoid ties in binary classification
    
    7. RECOMMENDATION FOR THIS DATASET:
       Best performing k: {results_df.loc[results_df['accuracy'].idxmax(), 'k']:.0f}
       (Accuracy: {results_df['accuracy'].max():.4f})
       
       - This k value provides the best balance for this specific dataset
       - {'Consider testing larger k values if overfitting is suspected' if k_values[0] == results_df.loc[results_df['accuracy'].idxmax(), 'k'] else 'This k value achieves good generalization'}
    """
    
    print(discussion)

def compare_with_naive_bayes(knn_results_df):
    """Compare KNN with Naive Bayes"""
    
    print("\n" + "=" * 60)
    print("COMPARISON: KNN vs NAIVE BAYES")
    print("=" * 60)
    
    # Load Naive Bayes results
    try:
        nb_results = np.load('naive_bayes_results.npy', allow_pickle=True).item()
        
        # Get best KNN results
        best_knn_idx = knn_results_df['accuracy'].idxmax()
        best_knn = knn_results_df.iloc[best_knn_idx]
        
        print("\nPerformance Comparison:")
        print("-" * 60)
        print(f"{'Metric':<15} {'Naive Bayes':<15} {'KNN (k={best_knn["k"]:.0f})':<15} {'Winner':<15}")
        print("-" * 60)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            nb_val = nb_results[metric]
            knn_val = best_knn[metric]
            winner = 'Naive Bayes' if nb_val > knn_val else 'KNN' if knn_val > nb_val else 'Tie'
            print(f"{metric.capitalize():<15} {nb_val:<15.4f} {knn_val:<15.4f} {winner:<15}")
        
        discussion = f"""
        
        WHICH CLASSIFIER PERFORMS BETTER AND WHY:
        
        1. PERFORMANCE COMPARISON:
           - Naive Bayes Accuracy: {nb_results['accuracy']:.4f}
           - Best KNN Accuracy: {best_knn['accuracy']:.4f}
           - Winner: {'Naive Bayes' if nb_results['accuracy'] > best_knn['accuracy'] else 'KNN' if best_knn['accuracy'] > nb_results['accuracy'] else 'Tie - Both perform similarly'}
        
        2. WHY THE DIFFERENCE:
           
           If Naive Bayes performs better:
           - Features may follow Gaussian distributions
           - Feature independence assumption may hold reasonably well
           - Dataset may have clear probabilistic separability
           - Less sensitive to feature scaling issues
           
           If KNN performs better:
           - Local patterns and neighborhoods are important
           - Decision boundaries may be non-linear
           - Features may not follow Gaussian distributions
           - Feature interactions matter (NB assumes independence)
        
        3. ALGORITHMIC DIFFERENCES:
           
           Naive Bayes:
           ✓ Probabilistic model based on Bayes' theorem
           ✓ Assumes feature independence
           ✓ Fast training and prediction
           ✓ Works well with small datasets
           ✓ Provides probability estimates
           ✗ Independence assumption often violated
           
           KNN:
           ✓ Non-parametric, instance-based learning
           ✓ No assumptions about data distribution
           ✓ Can capture complex decision boundaries
           ✓ Simple and intuitive
           ✗ Slow prediction (needs to compute distances)
           ✗ Memory intensive (stores all training data)
           ✗ Sensitive to feature scaling
           ✗ Curse of dimensionality
        
        4. SUITABILITY FOR THIS DATASET:
           - Dataset has {4} features (moderate dimensionality)
           - Binary classification task
           - {'KNN benefits from normalized features' if best_knn['accuracy'] > nb_results['accuracy'] else 'Probabilistic approach works well'}
           - {'Local patterns are important' if best_knn['accuracy'] > nb_results['accuracy'] else 'Global distributions are sufficient'}
        """
        
        print(discussion)
        
    except FileNotFoundError:
        print("\n⚠ Naive Bayes results not found. Please run 3_naive_bayes.py first.")

def save_knn_results(best_k, X_train, y_train):
    """Save the best KNN model"""
    
    import pickle
    
    print("\n" + "=" * 60)
    print("SAVING BEST KNN MODEL")
    print("=" * 60)
    
    # Train best model
    best_knn = KNeighborsClassifier(n_neighbors=int(best_k), metric='euclidean')
    best_knn.fit(X_train, y_train)
    
    # Save model
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(best_knn, f)
    
    print(f"\n✓ Best KNN model (k={best_k}) saved as 'knn_model.pkl'")

def main():
    """Main execution function"""
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train KNN with k=3
    knn_k3 = train_knn(X_train, y_train, k=3)
    
    # Predict and evaluate k=3
    y_pred_k3, acc_k3, prec_k3, rec_k3, f1_k3, cm_k3 = predict_and_evaluate(
        knn_k3, X_test, y_test, k_value=3
    )
    
    # Plot confusion matrix for k=3
    plot_confusion_matrix(cm_k3, k_value=3)
    
    # Compare different k values (3, 5, 7)
    results_df = compare_k_values(X_train, X_test, y_train, y_test, k_values=[3, 5, 7])
    
    # Discuss impact of k
    discuss_k_impact(results_df)
    
    # Compare with Naive Bayes
    compare_with_naive_bayes(results_df)
    
    # Save best model
    best_k = results_df.loc[results_df['accuracy'].idxmax(), 'k']
    save_knn_results(best_k, X_train, y_train)
    
    # Save comparison results
    results_df.to_csv('knn_k_comparison_results.csv', index=False)
    print("\n✓ K-value comparison results saved as 'knn_k_comparison_results.csv'")
    
    print("\n" + "=" * 60)
    print("KNN CLASSIFICATION COMPLETE!")
    print("=" * 60)
    print("\nNext Step: Proceed to final comparison and analysis (5_comparison_analysis.py)")

if __name__ == "__main__":
    main()
