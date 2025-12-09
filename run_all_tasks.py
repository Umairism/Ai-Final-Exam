"""
Master Script - Run All Tasks Sequentially
This script executes all 5 tasks in order:
1. Data Understanding
2. Preprocessing
3. Naive Bayes Classification
4. KNN Classification
5. Comparison & Critical Analysis
"""

import subprocess
import sys
import time

def print_header(task_name, task_number):
    """Print a formatted header for each task"""
    print("\n" + "=" * 80)
    print(f"TASK {task_number}: {task_name}")
    print("=" * 80 + "\n")

def run_script(script_name, task_name, task_number):
    """Run a Python script and handle errors"""
    print_header(task_name, task_number)
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Task {task_number} completed successfully in {elapsed_time:.2f} seconds!")
        print("-" * 80)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}:")
        print(f"  {str(e)}")
        print("\nPlease fix the error and try again.")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        print("  Make sure all task files are in the current directory.")
        return False

def main():
    """Main execution function"""
    
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "NAIVE BAYES & KNN CLASSIFICATION ANALYSIS" + " " * 22 + "║")
    print("║" + " " * 25 + "Master Execution Script" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Define tasks
    tasks = [
        ("1_data_understanding.py", "Data Understanding", 1),
        ("2_preprocessing.py", "Preprocessing", 2),
        ("3_naive_bayes.py", "Naive Bayes Classification", 3),
        ("4_knn_classification.py", "KNN Classification", 4),
        ("5_comparison_analysis.py", "Comparison & Critical Analysis", 5)
    ]
    
    overall_start_time = time.time()
    successful_tasks = 0
    
    # Run each task sequentially
    for script_name, task_name, task_number in tasks:
        success = run_script(script_name, task_name, task_number)
        
        if success:
            successful_tasks += 1
            time.sleep(1)  # Brief pause between tasks
        else:
            print(f"\nExecution stopped at Task {task_number}")
            break
    
    # Print final summary
    overall_elapsed_time = time.time() - overall_start_time
    
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal tasks completed: {successful_tasks}/{len(tasks)}")
    print(f"Total execution time: {overall_elapsed_time:.2f} seconds ({overall_elapsed_time/60:.2f} minutes)")
    
    if successful_tasks == len(tasks):
        print("\n✓ ALL TASKS COMPLETED SUCCESSFULLY!")
        print("\nGenerated Files:")
        print("  • Visualizations:")
        print("    - scatter_plot_feature1_vs_feature2.png")
        print("    - pairplot_all_features.png")
        print("    - normalization_comparison.png")
        print("    - confusion_matrix_naive_bayes.png")
        print("    - confusion_matrix_knn_k3.png")
        print("    - knn_k_comparison.png")
        print("    - comprehensive_comparison.png")
        print("\n  • Models:")
        print("    - naive_bayes_model.pkl")
        print("    - knn_model.pkl")
        print("    - scaler.pkl")
        print("\n  • Data:")
        print("    - X_train.npy, X_test.npy")
        print("    - y_train.npy, y_test.npy")
        print("\n  • Reports:")
        print("    - technical_report.txt")
        print("    - algorithm_comparison_table.csv")
        print("    - final_model_comparison.csv")
        print("    - knn_k_comparison_results.csv")
    else:
        print("\n⚠ Some tasks failed. Please check the error messages above.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
