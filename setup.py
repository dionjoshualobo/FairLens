"""
Setup and Configuration Script for AI Governance Tool
"""

import subprocess
import sys
import os

def install_packages():
    """Install required packages"""
    packages = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.3.0',
        'shap>=0.42.0',
        'fairlearn>=0.8.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        'streamlit>=1.28.0',
        'imbalanced-learn>=0.10.0',
        'lime>=0.2.0',
        'scipy>=1.9.0',
        'openpyxl>=3.1.0',
        'python-dateutil>=2.8.0'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    print("\n🎉 Package installation completed!")

def create_sample_config():
    """Create sample configuration file"""
    config_content = """
# AI Governance Tool Configuration

# Model Configuration
DEFAULT_MODEL = "RandomForest"
MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    },
    "LogisticRegression": {
        "random_state": 42,
        "max_iter": 1000
    },
    "GradientBoosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42
    }
}

# Privacy Configuration
PRIVACY_THRESHOLDS = {
    "high_risk_threshold": 3,
    "k_anonymity_min": 3,
    "uniqueness_threshold": 0.9
}

# Bias Configuration
BIAS_THRESHOLDS = {
    "demographic_parity_threshold": 0.1,
    "equalized_odds_threshold": 0.1,
    "equal_opportunity_threshold": 0.8
}

# Explainability Configuration
EXPLAINABILITY_CONFIG = {
    "use_shap": True,
    "use_lime": True,
    "max_samples_shap": 100,
    "max_samples_lime": 50
}

# Compliance Frameworks
COMPLIANCE_FRAMEWORKS = ["GDPR", "CCPA", "HIPAA"]
DEFAULT_FRAMEWORK = "GDPR"
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration file created: config.py")

def create_run_script():
    """Create a simple run script"""
    run_script = """#!/usr/bin/env python
\"\"\"
Quick start script for AI Governance Tool
\"\"\"

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    # Run the Streamlit app
    os.system("streamlit run main.py")
"""
    
    with open('run.py', 'w') as f:
        f.write(run_script)
    
    print("✅ Run script created: run.py")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Governance Tool...")
    print("=" * 50)
    
    # Install packages
    install_packages()
    
    print("\n⚙️ Creating configuration files...")
    create_sample_config()
    create_run_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("1. python run.py")
    print("   OR")
    print("2. streamlit run main.py")
    print("\nThe application will open in your web browser.")

if __name__ == "__main__":
    main()
