"""
Demo Data Loader and Validator
"""

import pandas as pd
import numpy as np

def load_and_validate_loan_data():
    """Load and validate the loan dataset"""
    try:
        # Try to load the data
        data = pd.read_csv('Datasets/loan_data.csv')
        
        print(f"✅ Data loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Basic validation
        print("\n📊 Data Overview:")
        print(f"- Total records: {len(data):,}")
        print(f"- Features: {len(data.columns)}")
        print(f"- Missing values: {data.isnull().sum().sum()}")
        
        # Check for potential sensitive features
        sensitive_indicators = ['gender', 'age', 'race', 'ethnicity', 'religion', 'income']
        found_sensitive = []
        
        for col in data.columns:
            for indicator in sensitive_indicators:
                if indicator.lower() in col.lower():
                    found_sensitive.append(col)
                    break
        
        print(f"\n🎯 Potential sensitive features found: {found_sensitive}")
        
        # Check target variable
        potential_targets = ['status', 'default', 'approved', 'outcome', 'result']
        target_candidates = []
        
        for col in data.columns:
            for target in potential_targets:
                if target.lower() in col.lower():
                    target_candidates.append(col)
                    break
        
        print(f"🎯 Potential target variables: {target_candidates}")
        
        # Data quality check
        print(f"\n🔍 Data Quality:")
        for col in data.columns:
            missing_pct = (data[col].isnull().sum() / len(data)) * 100
            unique_vals = data[col].nunique()
            print(f"- {col}: {missing_pct:.1f}% missing, {unique_vals} unique values")
        
        return data
        
    except FileNotFoundError:
        print("❌ Could not find loan_data.csv in Datasets folder")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def create_sample_analysis():
    """Create a sample analysis workflow"""
    print("🚀 Starting sample analysis workflow...")
    
    data = load_and_validate_loan_data()
    if data is None:
        return
    
    # Privacy analysis preview
    print("\n🔒 Privacy Analysis Preview:")
    
    # Look for potential PII
    pii_patterns = {
        'email': r'@',
        'phone': r'\d{3}[-.]?\d{3}[-.]?\d{4}',
        'ssn': r'\d{3}-?\d{2}-?\d{4}'
    }
    
    for col in data.columns:
        if data[col].dtype == 'object':
            sample_values = data[col].dropna().head(5).tolist()
            print(f"- {col}: {sample_values}")
    
    # Bias analysis preview
    print("\n⚖️ Bias Analysis Preview:")
    
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical features: {categorical_cols}")
    
    for col in categorical_cols[:3]:  # Show first 3
        value_counts = data[col].value_counts()
        print(f"- {col}: {dict(value_counts)}")
    
    # Model readiness check
    print("\n🤖 Model Readiness Check:")
    
    # Check for potential target
    binary_cols = []
    for col in data.columns:
        if data[col].nunique() == 2:
            binary_cols.append(col)
    
    print(f"Binary columns (potential targets): {binary_cols}")
    
    # Feature summary
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    print("\n✅ Sample analysis completed!")
    print("Run 'streamlit run main.py' to start the full analysis tool.")

if __name__ == "__main__":
    create_sample_analysis()
