"""
Quick Start Guide
================================================================================
This script helps you get started with the Naive Bayes & KNN Classification project.
Run this first to ensure all dependencies are installed and the environment is ready.
================================================================================
"""

import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    print("=" * 80)
    print("CHECKING PYTHON VERSION")
    print("=" * 80)
    
    version = sys.version_info
    print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 7:
        print("✓ Python version is compatible (Python 3.7+)")
        return True
    else:
        print("✗ Python 3.7 or higher is required")
        return False

def install_dependencies():
    """Install required packages"""
    print("\n" + "=" * 80)
    print("INSTALLING DEPENDENCIES")
    print("=" * 80)
    
    print("\nInstalling packages from requirements.txt...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
        ])
        print("✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Error installing dependencies")
        print("\nTry installing manually:")
        print("  pip install pandas numpy scikit-learn matplotlib seaborn")
        return False

def check_imports():
    """Verify all required packages can be imported"""
    print("\n" + "=" * 80)
    print("VERIFYING PACKAGE IMPORTS")
    print("=" * 80)
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    all_imports_ok = True
    
    for module, package_name in packages.items():
        try:
            __import__(module)
            print(f"✓ {package_name:20s} imported successfully")
        except ImportError:
            print(f"✗ {package_name:20s} import failed")
            all_imports_ok = False
    
    return all_imports_ok

def check_dataset():
    """Check if the dataset file exists"""
    print("\n" + "=" * 80)
    print("CHECKING DATASET")
    print("=" * 80)
    
    import os
    
    dataset_file = "complex_binary_dataset.csv"
    
    if os.path.exists(dataset_file):
        file_size = os.path.getsize(dataset_file)
        print(f"\n✓ Dataset found: {dataset_file}")
        print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        return True
    else:
        print(f"\n✗ Dataset not found: {dataset_file}")
        print("  Please ensure the dataset file is in the current directory")
        return False

def check_script_files():
    """Check if all task scripts exist"""
    print("\n" + "=" * 80)
    print("CHECKING SCRIPT FILES")
    print("=" * 80)
    
    import os
    
    scripts = [
        "1_data_understanding.py",
        "2_preprocessing.py",
        "3_naive_bayes.py",
        "4_knn_classification.py",
        "5_comparison_analysis.py",
        "run_all_tasks.py"
    ]
    
    all_scripts_exist = True
    
    print()
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"✗ {script} - MISSING")
            all_scripts_exist = False
    
    return all_scripts_exist

def display_next_steps():
    """Display instructions for running the project"""
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    print("""
Ready to run the project! You have two options:

OPTION 1: Run all tasks at once (Recommended)
----------------------------------------------
    python run_all_tasks.py

This will execute all 5 tasks sequentially and generate all outputs.


OPTION 2: Run tasks individually
---------------------------------
    python 1_data_understanding.py
    python 2_preprocessing.py
    python 3_naive_bayes.py
    python 4_knn_classification.py
    python 5_comparison_analysis.py

Note: Tasks must be run in order!


WHAT TO EXPECT:
---------------
• Total execution time: ~1-3 minutes (depends on your system)
• Multiple visualizations will be generated (.png files)
• Models will be trained and saved (.pkl files)
• Comprehensive reports will be created (.txt, .csv files)
• All outputs will be saved in the current directory


OUTPUTS:
--------
Visualizations (7 PNG files):
  - scatter_plot_feature1_vs_feature2.png
  - pairplot_all_features.png
  - normalization_comparison.png
  - confusion_matrix_naive_bayes.png
  - confusion_matrix_knn_k3.png
  - knn_k_comparison.png
  - comprehensive_comparison.png

Models (3 PKL files):
  - naive_bayes_model.pkl
  - knn_model.pkl
  - scaler.pkl

Data (4 NPY files):
  - X_train.npy, X_test.npy
  - y_train.npy, y_test.npy

Reports (4 files):
  - technical_report.txt
  - algorithm_comparison_table.csv
  - final_model_comparison.csv
  - knn_k_comparison_results.csv


For more information, see README.md
""")

def main():
    """Main execution function"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "QUICK START - ENVIRONMENT CHECK" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", install_dependencies()),
        ("Package Imports", check_imports()),
        ("Dataset File", check_dataset()),
        ("Script Files", check_script_files())
    ]
    
    # Summary
    print("\n" + "=" * 80)
    print("ENVIRONMENT CHECK SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"\n{check_name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("\n✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        print("Your environment is ready to run the project.")
        display_next_steps()
    else:
        print("\n⚠ SOME CHECKS FAILED")
        print("Please resolve the issues above before running the project.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
