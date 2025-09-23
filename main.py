"""
AI Model Governance and Fairness Analysis Tool
Comprehensive analysis for bias, privacy, and governance compliance
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML and Fairness Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

# Fairness and Bias Detection
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    st.warning("Fairlearn not installed. Some fairness metrics will be unavailable.")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
    st.sidebar.success("✅ SHAP available for explainability")
except ImportError:
    SHAP_AVAILABLE = False
    st.sidebar.warning("⚠️ SHAP not installed")
    st.sidebar.code("pip install shap", language="bash")

# Privacy and Data Analysis
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from bias_detection import AdvancedBiasDetector, FairnessMitigator, calculate_intersectional_bias
    from privacy_analysis import PrivacyAnalyzer
    from explainability import ModelExplainer
    CUSTOM_MODULES_AVAILABLE = True
except ImportError:
    CUSTOM_MODULES_AVAILABLE = False
    st.info("Custom modules not found. Using basic analysis only.")

class ModelGovernanceAnalyzer:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.sensitive_features = []
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self, file_path):
        """Load and preprocess the dataset"""
        try:
            self.data = pd.read_csv(file_path)
            
            # Basic data quality checks
            original_shape = self.data.shape
            
            # Remove completely empty rows
            self.data = self.data.dropna(how='all')
            
            # Remove duplicate rows
            initial_len = len(self.data)
            self.data = self.data.drop_duplicates()
            duplicates_removed = initial_len - len(self.data)
            
            st.success(f"Data loaded successfully! Shape: {self.data.shape}")
            
            if duplicates_removed > 0:
                st.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Data quality summary
            st.subheader("📋 Data Quality Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(self.data))
            with col2:
                st.metric("Features", len(self.data.columns))
            with col3:
                missing_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
                st.metric("Missing Data %", f"{missing_pct:.1f}%")
            
            # Check for potential issues
            warnings = []
            
            # Check for columns with too many missing values
            for col in self.data.columns:
                missing_pct = (self.data[col].isnull().sum() / len(self.data)) * 100
                if missing_pct > 50:
                    warnings.append(f"Column '{col}' has {missing_pct:.1f}% missing values")
            
            # Check for columns with only one unique value
            for col in self.data.columns:
                if self.data[col].nunique() == 1:
                    warnings.append(f"Column '{col}' has only one unique value")
            
            if warnings:
                st.warning("⚠️ Data Quality Issues:")
                for warning in warnings:
                    st.write(f"- {warning}")
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def handle_missing_values(self):
        """Comprehensive missing value detection and handling"""
        st.header("🔧 Missing Value Analysis & Handling")
        
        if self.data is None:
            st.warning("Please load data first.")
            return False
        
        # Calculate missing value statistics
        missing_stats = []
        total_rows = len(self.data)
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                missing_stats.append({
                    'Column': col,
                    'Missing_Count': missing_count,
                    'Missing_Percentage': missing_pct,
                    'Data_Type': str(self.data[col].dtype),
                    'Non_Missing_Count': total_rows - missing_count,
                    'Unique_Values': self.data[col].nunique()
                })
        
        if not missing_stats:
            st.success("✅ No missing values found in the dataset!")
            return True
        
        # Display missing value summary
        st.subheader("📊 Missing Value Summary")
        missing_df = pd.DataFrame(missing_stats).sort_values('Missing_Percentage', ascending=False)
        
        # Color coding based on severity
        def color_missing_pct(val):
            if val >= 70:
                return 'background-color: #ffebee'  # Light red
            elif val >= 50:
                return 'background-color: #fff3e0'  # Light orange
            elif val >= 20:
                return 'background-color: #f3e5f5'  # Light purple
            else:
                return 'background-color: #e8f5e8'  # Light green
        
        styled_df = missing_df.style.applymap(color_missing_pct, subset=['Missing_Percentage'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Visual representation
        fig = px.bar(missing_df, x='Column', y='Missing_Percentage',
                     title='Missing Values Percentage by Column',
                     color='Missing_Percentage',
                     color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Critical columns analysis
        st.subheader("⚠️ Critical Missing Value Issues")
        
        critical_cols = missing_df[missing_df['Missing_Percentage'] >= 70]['Column'].tolist()
        high_missing_cols = missing_df[missing_df['Missing_Percentage'] >= 50]['Column'].tolist()
        moderate_missing_cols = missing_df[missing_df['Missing_Percentage'] >= 20]['Column'].tolist()
        
        if critical_cols:
            st.error(f"🚨 **Critical** (≥70% missing): {', '.join(critical_cols)}")
            st.write("**Recommendation**: Consider removing these columns as they have too little data to be useful.")
        
        if high_missing_cols:
            st.warning(f"⚠️ **High** (50-70% missing): {', '.join(high_missing_cols)}")
            st.write("**Recommendation**: Carefully evaluate if these columns are essential. Consider removal or advanced imputation.")
        
        if moderate_missing_cols:
            st.info(f"📝 **Moderate** (20-50% missing): {', '.join(moderate_missing_cols)}")
            st.write("**Recommendation**: Can be handled with imputation strategies.")
        
        # User decision interface
        st.subheader("🛠️ Missing Value Handling Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Automatic Recommendations:**")
            auto_drop_cols = missing_df[missing_df['Missing_Percentage'] >= 70]['Column'].tolist()
            if auto_drop_cols:
                st.write(f"🗑️ **Drop columns** (≥70% missing): {', '.join(auto_drop_cols)}")
            
            numerical_cols = missing_df[missing_df['Data_Type'].str.contains('int|float')]['Column'].tolist()
            categorical_cols = missing_df[~missing_df['Data_Type'].str.contains('int|float')]['Column'].tolist()
            
            if numerical_cols:
                st.write(f"🔢 **Numerical columns** for imputation: {', '.join(numerical_cols)}")
            if categorical_cols:
                st.write(f"📝 **Categorical columns** for imputation: {', '.join(categorical_cols)}")
        
        with col2:
            st.write("**Your Choices:**")
            
            # Column dropping strategy
            if critical_cols or high_missing_cols:
                drop_strategy = st.selectbox(
                    "Handle high-missing columns:",
                    [
                        "Auto-drop columns with ≥70% missing",
                        "Drop columns with ≥50% missing", 
                        "Keep all columns and impute",
                        "Let me choose manually"
                    ]
                )
            else:
                drop_strategy = "No high-missing columns to drop"
                st.info("No columns with >50% missing values")
            
            # Imputation strategy
            imputation_strategy = st.selectbox(
                "Imputation method:",
                [
                    "Smart imputation (mean/median for numerical, mode for categorical)",
                    "Mean/Mode imputation only",
                    "Median/Mode imputation only", 
                    "KNN imputation (advanced)",
                    "Forward fill",
                    "Backward fill",
                    "Drop rows with missing values"
                ]
            )
        
        # Manual column selection if needed
        if drop_strategy == "Let me choose manually":
            st.subheader("🎯 Manual Column Selection")
            st.write("**Select columns to DROP (remove from dataset):**")
            
            cols_to_drop = st.multiselect(
                "Columns to remove:",
                missing_df['Column'].tolist(),
                default=auto_drop_cols,
                help="Select columns you want to remove from the dataset"
            )
            
            if cols_to_drop:
                st.warning(f"⚠️ You selected to drop: {', '.join(cols_to_drop)}")
                
                # Check if target variable suggestions would be affected
                remaining_cols = [col for col in self.data.columns if col not in cols_to_drop]
                potential_targets = []
                for col in remaining_cols:
                    if col in self.data.columns:
                        unique_vals = self.data[col].nunique()
                        if 2 <= unique_vals <= 10:
                            potential_targets.append(col)
                
                if potential_targets:
                    st.info(f"✅ After dropping, potential target variables remain: {', '.join(potential_targets[:3])}")
                else:
                    st.error("❌ Warning: No suitable target variables will remain after dropping these columns!")
        
        # Apply the strategy
        if st.button("🚀 Apply Missing Value Strategy", type="primary"):
            return self._apply_missing_value_strategy(
                missing_df, drop_strategy, imputation_strategy, 
                cols_to_drop if 'cols_to_drop' in locals() else []
            )
        
        return False
    
    def _apply_missing_value_strategy(self, missing_df, drop_strategy, imputation_strategy, manual_cols_to_drop):
        """Apply the selected missing value handling strategy"""
        
        original_shape = self.data.shape
        cols_dropped = []
        
        try:
            # Step 1: Handle column dropping
            if drop_strategy == "Auto-drop columns with ≥70% missing":
                cols_to_drop = missing_df[missing_df['Missing_Percentage'] >= 70]['Column'].tolist()
            elif drop_strategy == "Drop columns with ≥50% missing":
                cols_to_drop = missing_df[missing_df['Missing_Percentage'] >= 50]['Column'].tolist()
            elif drop_strategy == "Let me choose manually":
                cols_to_drop = manual_cols_to_drop
            else:
                cols_to_drop = []
            
            if cols_to_drop:
                self.data = self.data.drop(columns=cols_to_drop)
                cols_dropped = cols_to_drop
                st.success(f"✅ Dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}")
            
            # Step 2: Handle imputation for remaining missing values
            remaining_missing = self.data.isnull().sum()
            cols_with_missing = remaining_missing[remaining_missing > 0].index.tolist()
            
            if not cols_with_missing:
                st.success("✅ No missing values remaining after column drops!")
                self._display_cleaning_summary(original_shape, cols_dropped, 0)
                return True
            
            if imputation_strategy == "Drop rows with missing values":
                rows_before = len(self.data)
                self.data = self.data.dropna()
                rows_dropped = rows_before - len(self.data)
                st.success(f"✅ Dropped {rows_dropped} rows with missing values")
                
            else:
                # Separate numerical and categorical columns
                numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
                
                numerical_missing = [col for col in numerical_cols if col in cols_with_missing]
                categorical_missing = [col for col in categorical_cols if col in cols_with_missing]
                
                # Apply imputation
                if imputation_strategy == "Smart imputation (mean/median for numerical, mode for categorical)":
                    # Numerical: use median (more robust to outliers)
                    if numerical_missing:
                        imputer_num = SimpleImputer(strategy='median')
                        self.data[numerical_missing] = imputer_num.fit_transform(self.data[numerical_missing])
                        st.success(f"✅ Imputed numerical columns with median: {', '.join(numerical_missing)}")
                    
                    # Categorical: use most frequent
                    if categorical_missing:
                        imputer_cat = SimpleImputer(strategy='most_frequent')
                        self.data[categorical_missing] = imputer_cat.fit_transform(self.data[categorical_missing])
                        st.success(f"✅ Imputed categorical columns with mode: {', '.join(categorical_missing)}")
                
                elif imputation_strategy == "Mean/Mode imputation only":
                    if numerical_missing:
                        imputer_num = SimpleImputer(strategy='mean')
                        self.data[numerical_missing] = imputer_num.fit_transform(self.data[numerical_missing])
                        st.success(f"✅ Imputed numerical columns with mean: {', '.join(numerical_missing)}")
                    
                    if categorical_missing:
                        imputer_cat = SimpleImputer(strategy='most_frequent')
                        self.data[categorical_missing] = imputer_cat.fit_transform(self.data[categorical_missing])
                        st.success(f"✅ Imputed categorical columns with mode: {', '.join(categorical_missing)}")
                
                elif imputation_strategy == "Median/Mode imputation only":
                    if numerical_missing:
                        imputer_num = SimpleImputer(strategy='median')
                        self.data[numerical_missing] = imputer_num.fit_transform(self.data[numerical_missing])
                        st.success(f"✅ Imputed numerical columns with median: {', '.join(numerical_missing)}")
                    
                    if categorical_missing:
                        imputer_cat = SimpleImputer(strategy='most_frequent')
                        self.data[categorical_missing] = imputer_cat.fit_transform(self.data[categorical_missing])
                        st.success(f"✅ Imputed categorical columns with mode: {', '.join(categorical_missing)}")
                
                elif imputation_strategy == "KNN imputation (advanced)":
                    if numerical_missing:
                        # KNN imputation for numerical columns
                        imputer_knn = KNNImputer(n_neighbors=5)
                        self.data[numerical_missing] = imputer_knn.fit_transform(self.data[numerical_missing])
                        st.success(f"✅ Applied KNN imputation to numerical columns: {', '.join(numerical_missing)}")
                    
                    if categorical_missing:
                        # Use most frequent for categorical
                        imputer_cat = SimpleImputer(strategy='most_frequent')
                        self.data[categorical_missing] = imputer_cat.fit_transform(self.data[categorical_missing])
                        st.success(f"✅ Imputed categorical columns with mode: {', '.join(categorical_missing)}")
                
                elif imputation_strategy == "Forward fill":
                    for col in cols_with_missing:
                        self.data[col] = self.data[col].ffill()
                    st.success(f"✅ Applied forward fill to: {', '.join(cols_with_missing)}")
                
                elif imputation_strategy == "Backward fill":
                    for col in cols_with_missing:
                        self.data[col] = self.data[col].bfill()
                    st.success(f"✅ Applied backward fill to: {', '.join(cols_with_missing)}")
            
            # Final verification
            final_missing = self.data.isnull().sum().sum()
            self._display_cleaning_summary(original_shape, cols_dropped, final_missing)
            
            if final_missing == 0:
                st.success("🎉 **Perfect!** No missing values remain in the dataset!")
                return True
            else:
                st.warning(f"⚠️ {final_missing} missing values still remain. You may want to apply additional cleaning.")
                return True
                
        except Exception as e:
            st.error(f"❌ Error applying missing value strategy: {str(e)}")
            return False
    
    def _display_cleaning_summary(self, original_shape, cols_dropped, final_missing):
        """Display summary of data cleaning results"""
        st.subheader("📋 Data Cleaning Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Shape", f"{original_shape[0]}×{original_shape[1]}")
        with col2:
            st.metric("Final Shape", f"{self.data.shape[0]}×{self.data.shape[1]}")
        with col3:
            st.metric("Columns Dropped", len(cols_dropped))
        with col4:
            st.metric("Missing Values Left", final_missing)
        
        if cols_dropped:
            st.info(f"🗑️ **Dropped columns**: {', '.join(cols_dropped)}")
        
        # Show remaining data quality
        st.write("**Final Data Quality:**")
        quality_info = pd.DataFrame({
            'Metric': ['Total Records', 'Total Features', 'Missing Values', 'Completeness %'],
            'Value': [
                len(self.data),
                len(self.data.columns),
                final_missing,
                f"{((1 - final_missing / (len(self.data) * len(self.data.columns))) * 100):.1f}%"
            ]
        })
        st.dataframe(quality_info, use_container_width=True)
    
    def analyze_target_variable(self, target_col):
        """Analyze and prepare target variable for analysis"""
        st.subheader("🎯 Target Variable Analysis")
        
        target_values = self.data[target_col].value_counts().sort_index()
        st.write("**Target Variable Distribution:**")
        
        # Display distribution table
        target_df = pd.DataFrame({
            'Value': target_values.index,
            'Count': target_values.values,
            'Percentage': (target_values.values / len(self.data) * 100).round(2)
        })
        st.dataframe(target_df)
        
        # Visualization
        fig = px.bar(target_df, x='Value', y='Count', 
                     title=f"Distribution of {target_col}",
                     text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig)
        
        # Analysis and recommendations
        unique_values = len(target_values)
        
        if unique_values == 2:
            st.success("✅ Binary classification problem detected")
            
            # Check for class imbalance
            min_class_pct = target_df['Percentage'].min()
            if min_class_pct < 10:
                st.warning(f"⚠️ Class imbalance detected: smallest class is {min_class_pct:.1f}%")
                st.info("Consider using stratified sampling or resampling techniques")
            
        elif unique_values > 10:
            st.warning(f"⚠️ Multi-class problem with {unique_values} classes")
            st.info("Fairness analysis works best with binary or few-class problems")
            
        else:
            st.info(f"ℹ️ Multi-class classification with {unique_values} classes")
        
        return target_df
    
    def data_overview(self):
        """Provide comprehensive data overview"""
        st.header("📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(self.data))
        with col2:
            st.metric("Features", len(self.data.columns))
        with col3:
            st.metric("Missing Values", self.data.isnull().sum().sum())
        
        # Data types and basic info
        st.subheader("Data Types and Basic Information")
        info_df = pd.DataFrame({
            'Feature': self.data.columns,
            'Data Type': [str(dtype) for dtype in self.data.dtypes],  # Convert to string to fix Arrow serialization
            'Non-Null Count': self.data.count(),
            'Null Count': self.data.isnull().sum(),
            'Unique Values': [self.data[col].nunique() for col in self.data.columns]
        })
        st.dataframe(info_df)
        
        # Sample data with target guidance
        st.subheader("Sample Data")
        st.dataframe(self.data.head(10))
        
        # Target variable guidance
        st.subheader("🎯 Target Variable Guidance")
        st.info("**What makes a good target variable for classification:**")
        st.write("✅ **Categorical variables** with 2-10 unique values")
        st.write("✅ **Business outcomes** like approval/denial, success/failure, high/low risk")
        st.write("❌ **Avoid**: Continuous numbers (age, income), IDs, dates")
        
        # Show potential targets
        potential_targets = []
        for col in self.data.columns:
            unique_vals = self.data[col].nunique()
            if 2 <= unique_vals <= 10:
                sample_values = self.data[col].value_counts().head(3)
                potential_targets.append({
                    'Column': col,
                    'Unique Values': unique_vals,
                    'Sample Values': ', '.join([f"{k}({v})" for k, v in sample_values.items()])
                })
        
        if potential_targets:
            st.write("**🎯 Recommended Target Variables:**")
            targets_df = pd.DataFrame(potential_targets)
            st.dataframe(targets_df)
        else:
            st.warning("No obvious target variables found. Look for columns representing outcomes or decisions.")
        
        # Special guidance for loan datasets
        if any('loan' in col.lower() for col in self.data.columns):
            st.info("📋 **For Loan Datasets**: Look for columns like 'loan_status', 'default', 'approved', 'outcome'")
            loan_related = [col for col in self.data.columns if any(keyword in col.lower() 
                          for keyword in ['status', 'default', 'approved', 'outcome', 'result'])]
            if loan_related:
                st.write(f"**Found loan-related columns**: {', '.join(loan_related)}")
        
        return info_df
    
    def privacy_analysis(self):
        """Comprehensive privacy analysis"""
        st.header("🔒 Privacy Analysis")
        
        # Page description
        st.info("📋 **About Privacy Analysis:** This step identifies sensitive data in your dataset that could pose privacy risks or compliance issues. We analyze each feature for personally identifiable information (PII), assess re-identification risks, and provide recommendations for data protection based on privacy regulations like GDPR and CCPA.")
        
        # Identify sensitive features
        sensitive_patterns = {
            'age': ['age', 'birth', 'dob'],
            'gender': ['gender', 'sex'],
            'race': ['race', 'ethnicity', 'ethnic'],
            'religion': ['religion', 'religious'],
            'location': ['address', 'zip', 'postal', 'location', 'lat', 'lon'],
            'personal_id': ['id', 'ssn', 'social', 'passport', 'license'],
            'financial': ['income', 'salary', 'wage', 'credit', 'score'],
            'health': ['health', 'medical', 'disease', 'condition'],
            'contact': ['email', 'phone', 'mobile', 'contact']
        }
        
        identified_sensitive = {}
        for category, patterns in sensitive_patterns.items():
            matches = []
            for col in self.data.columns:
                if any(pattern.lower() in col.lower() for pattern in patterns):
                    matches.append(col)
            if matches:
                identified_sensitive[category] = matches
        
        st.subheader("🎯 Identified Sensitive Features")
        for category, features in identified_sensitive.items():
            st.write(f"**{category.title()}**: {', '.join(features)}")
        
        # Privacy risk assessment
        st.subheader("⚠️ Privacy Risk Assessment")
        
        risk_scores = {}
        for col in self.data.columns:
            risk_score = 0
            
            # High uniqueness = higher risk
            uniqueness = self.data[col].nunique() / len(self.data)
            if uniqueness > 0.9:
                risk_score += 3
            elif uniqueness > 0.5:
                risk_score += 2
            elif uniqueness > 0.1:
                risk_score += 1
            
            # Check if it's in sensitive categories
            for category, features in identified_sensitive.items():
                if col in features:
                    risk_score += 2
            
            # Check for potential identifiers
            if any(keyword in col.lower() for keyword in ['id', 'key', 'unique']):
                risk_score += 3
            
            risk_scores[col] = min(risk_score, 5)  # Cap at 5
        
        risk_df = pd.DataFrame(list(risk_scores.items()), columns=['Feature', 'Privacy Risk'])
        risk_df['Risk Level'] = risk_df['Privacy Risk'].map({
            0: 'Very Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High', 5: 'Critical'
        })
        
        st.dataframe(risk_df.sort_values('Privacy Risk', ascending=False))
        
        # Compliance recommendations
        st.subheader("📋 Compliance Recommendations")
        
        high_risk_features = risk_df[risk_df['Privacy Risk'] >= 3]['Feature'].tolist()
        
        recommendations = []
        if high_risk_features:
            recommendations.append("🔴 **High Priority**: Consider anonymizing or removing high-risk features: " + ", ".join(high_risk_features))
        
        if 'gender' in str(identified_sensitive.get('gender', [])).lower():
            recommendations.append("⚖️ **Fairness**: Gender detected - ensure compliance with anti-discrimination laws")
        
        if 'age' in str(identified_sensitive.get('age', [])).lower():
            recommendations.append("👥 **Age Discrimination**: Age-related features detected - verify ADEA compliance")
        
        recommendations.extend([
            "📊 **Data Minimization**: Only collect and use data necessary for the business purpose",
            "🔐 **Encryption**: Ensure sensitive data is encrypted at rest and in transit",
            "📝 **Consent**: Verify proper consent mechanisms are in place",
            "🕒 **Retention**: Implement data retention policies",
            "🌍 **Geographic**: Consider GDPR, CCPA, and other regional privacy laws"
        ])
        
        for rec in recommendations:
            st.write(rec)
        
        return risk_df, identified_sensitive
    
    def bias_analysis(self):
        """Comprehensive bias detection and analysis"""
        st.header("⚖️ Bias Analysis")
        
        # Page description
        st.info("⚖️ **About Bias Analysis:** This step helps identify potential discrimination in your data by analyzing relationships between sensitive demographic features (like gender, race, age) and outcomes. We examine statistical disparities that could indicate unfair treatment and prepare your data for fairness-aware model training.")
        
        # Identify categorical features that could be sensitive
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        st.subheader("🎯 Select Sensitive Features for Bias Analysis")
        
        # Let user select sensitive features
        # Use session state to remember previous selections
        if 'selected_sensitive' not in st.session_state:
            # Initialize with smart defaults only on first visit
            st.session_state.selected_sensitive = [col for col in categorical_cols if any(keyword in col.lower() 
                    for keyword in ['gender', 'race', 'ethnicity', 'age', 'education'])]
        
        selected_sensitive = st.multiselect(
            "Select features to analyze for bias:",
            categorical_cols,
            default=st.session_state.selected_sensitive,
            key="bias_sensitive_multiselect"
        )
        
        # Update session state when selection changes
        if selected_sensitive != st.session_state.selected_sensitive:
            st.session_state.selected_sensitive = selected_sensitive
        
        # Show current selections for user feedback
        if selected_sensitive:
            st.success(f"📊 **Selected sensitive features**: {', '.join(selected_sensitive)}")
        
        if not selected_sensitive:
            st.warning("Please select at least one sensitive feature for bias analysis.")
            return None, None
        
        # Target variable selection with smart suggestions
        st.subheader("🎯 Target Variable Selection")
        
        # Suggest appropriate target variables
        suggested_targets = []
        for col in self.data.columns:
            unique_vals = self.data[col].nunique()
            if 2 <= unique_vals <= 10:  # Good range for classification
                suggested_targets.append(col)
        
        # Look for common target patterns
        target_patterns = ['status', 'default', 'approved', 'outcome', 'result', 'class', 'label', 'target']
        likely_targets = []
        for col in self.data.columns:
            if any(pattern.lower() in col.lower() for pattern in target_patterns):
                likely_targets.append(col)
        
        # Display suggestions
        if suggested_targets:
            st.info(f"💡 **Suggested target variables** (2-10 unique values): {', '.join(suggested_targets[:5])}")
        if likely_targets:
            st.info(f"🎯 **Likely target variables** (by name): {', '.join(likely_targets)}")
        
        # Show data types to help user choose
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Categorical Features:**")
            for col in categorical_cols[:10]:  # Show first 10
                unique_count = self.data[col].nunique()
                st.write(f"- {col} ({unique_count} unique)")
        
        with col2:
            st.write("**Numerical Features:**")
            for col in numerical_cols[:10]:  # Show first 10
                unique_count = self.data[col].nunique()
                st.write(f"- {col} ({unique_count} unique)")
        
        # Use session state to remember target variable selection
        if 'target_col' not in st.session_state:
            st.session_state.target_col = self.data.columns[0]  # Default to first column
        
        # Find the index of the previously selected target variable
        try:
            current_index = self.data.columns.tolist().index(st.session_state.target_col)
        except (ValueError, AttributeError):
            current_index = 0  # Fallback to first column if previous selection not found
        
        target_col = st.selectbox(
            "Select target variable:", 
            self.data.columns.tolist(),
            index=current_index,
            key="bias_target_selectbox"
        )
        
        # Update session state when target variable changes
        if target_col != st.session_state.target_col:
            st.session_state.target_col = target_col
        
        # Show current target selection for user feedback
        if target_col:
            st.success(f"🎯 **Selected target variable**: {target_col}")
        
        if target_col:
            # Analyze target variable
            target_analysis = self.analyze_target_variable(target_col)
            
            # Enhanced target variable validation
            unique_targets = self.data[target_col].nunique()
            
            if unique_targets < 2:
                st.error(f"❌ Target variable '{target_col}' has only {unique_targets} unique value(s). Need at least 2 for classification.")
                return None, None
            elif unique_targets > 20:
                st.error(f"❌ Target variable '{target_col}' has {unique_targets} unique values. This appears to be a continuous variable or identifier, not suitable for classification.")
                st.info("💡 **Suggestion**: Choose a categorical variable like loan_status, approval, outcome, etc.")
                
                # Show alternative suggestions
                better_alternatives = [col for col in self.data.columns if 2 <= self.data[col].nunique() <= 10]
                if better_alternatives:
                    st.write(f"**Better alternatives**: {', '.join(better_alternatives[:5])}")
                return None, None
            elif unique_targets > 10:
                st.warning(f"⚠️ Target variable '{target_col}' has {unique_targets} unique values. This is a complex multi-class problem.")
                st.info("Fairness analysis works best with binary (2 classes) or simple multi-class (3-5 classes) problems.")
                
                # Ask user if they want to continue
                continue_anyway = st.checkbox("Continue with this target variable anyway (not recommended)")
                if not continue_anyway:
                    return None, None
            
            # Check if target is numeric with many unique values (likely continuous)
            if self.data[target_col].dtype in [np.int64, np.float64] and unique_targets > 10:
                st.error(f"❌ '{target_col}' appears to be a continuous numerical variable (like age, income, etc.).")
                st.info("💡 **For classification**, you need a categorical target variable like:")
                st.write("- loan_status (approved/denied)")
                st.write("- default (yes/no)")
                st.write("- risk_level (low/medium/high)")
                return None, None
            
            # Statistical bias analysis
            st.subheader("📊 Statistical Bias Analysis")
            
            for sensitive_feature in selected_sensitive:
                if sensitive_feature not in self.data.columns:
                    st.warning(f"⚠️ Feature '{sensitive_feature}' not found in data")
                    continue
                    
                st.write(f"**Analysis for: {sensitive_feature}**")
                
                # Cross-tabulation
                try:
                    crosstab = pd.crosstab(self.data[sensitive_feature], self.data[target_col], normalize='index')
                    
                    fig = px.bar(crosstab, title=f"Target Distribution by {sensitive_feature}")
                    st.plotly_chart(fig)
                    
                    # Statistical test
                    try:
                        from scipy.stats import chi2_contingency
                        chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(self.data[sensitive_feature], self.data[target_col]))
                        
                        st.write(f"Chi-square test p-value: {p_value:.4f}")
                        if p_value < 0.05:
                            st.warning(f"⚠️ Significant association detected between {sensitive_feature} and {target_col}")
                        else:
                            st.success(f"✅ No significant bias detected for {sensitive_feature}")
                    except Exception as e:
                        st.info(f"Could not perform statistical test: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error analyzing {sensitive_feature}: {str(e)}")
        
        # Ensure we always return a tuple
        if not target_col:
            st.warning("⚠️ No target variable selected.")
            return None, None
            
        return selected_sensitive, target_col
    
    def train_models(self, target_col, sensitive_features):
        """Train multiple models for comparison"""
        st.subheader("🤖 Model Training")
        
        # Page description
        st.info("🤖 **About Model Training:** This step trains multiple machine learning models on your data to compare their performance and fairness characteristics. We use Random Forest, Logistic Regression, and Gradient Boosting algorithms, then select the best performer for explainability and fairness analysis.")
        
        # Check for missing values before training
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            st.error(f"❌ **Missing values detected** ({missing_values} total)")
            st.write("**Please handle missing values first before training models.**")
            
            # Show which columns have missing values
            cols_with_missing = self.data.columns[self.data.isnull().any()].tolist()
            missing_info = pd.DataFrame({
                'Column': cols_with_missing,
                'Missing_Count': [self.data[col].isnull().sum() for col in cols_with_missing],
                'Missing_Percentage': [f"{(self.data[col].isnull().sum() / len(self.data)) * 100:.1f}%" 
                                     for col in cols_with_missing]
            })
            st.dataframe(missing_info)
            
            st.info("💡 **Go to Step 1.5: Handle Missing Values** to clean your data first.")
            return None, None
        
        # Prepare data
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        # Check for missing values in target
        if y.isnull().any():
            st.error(f"❌ Target variable '{target_col}' has missing values. Please handle this first.")
            return None, None
        
        # Display target variable distribution
        target_counts = y.value_counts()
        st.write("**Target Variable Distribution:**")
        st.write(target_counts)
        
        # Check for stratification feasibility
        min_class_count = target_counts.min()
        can_stratify = min_class_count >= 2
        
        if not can_stratify:
            st.warning(f"⚠️ Smallest class has only {min_class_count} sample(s). Cannot use stratified sampling.")
            st.info("Consider collecting more data for underrepresented classes or using different sampling strategy.")
        
        # Encode categorical variables
        label_encoders = {}
        X_encoded = X.copy()
        
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Split data with or without stratification
        if can_stratify:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42, stratify=y
            )
            st.success("✅ Used stratified sampling to maintain class distribution")
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            st.info("ℹ️ Used random sampling (no stratification)")
        
        # Display split information
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training Set:**")
            st.write(f"- Size: {len(self.y_train)}")
            st.write(f"- Distribution: {dict(self.y_train.value_counts())}")
        
        with col2:
            st.write("**Test Set:**")
            st.write(f"- Size: {len(self.y_test)}")
            st.write(f"- Distribution: {dict(self.y_test.value_counts())}")
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        model_results = {}
        
        for name, model in models.items():
            try:
                st.write(f"Training {name}...")
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
                # Calculate accuracy with error handling
                try:
                    accuracy = accuracy_score(self.y_test, y_pred)
                except Exception as acc_error:
                    st.warning(f"Could not calculate standard accuracy for {name}: {str(acc_error)}")
                    # Calculate simple accuracy manually
                    accuracy = np.mean(self.y_test == y_pred)
                
                model_results[name] = {'model': model, 'accuracy': accuracy, 'predictions': y_pred}
                st.write(f"✅ {name} trained successfully (Accuracy: {accuracy:.4f})")
                
                # Additional model diagnostics
                unique_predictions = len(np.unique(y_pred))
                st.write(f"   - Predictions: {unique_predictions} unique values")
                
                if unique_predictions == 1:
                    st.warning(f"   ⚠️ {name} is predicting only one class! This suggests a problem with the data or target variable.")
                
            except Exception as e:
                st.error(f"❌ Error training {name}: {str(e)}")
                continue
        
        if not model_results:
            st.error("❌ No models were trained successfully. Please check your data.")
            return None, None
        
        # Display results
        results_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Accuracy': [results['accuracy'] for results in model_results.values()]
        }).sort_values('Accuracy', ascending=False)
        
        st.dataframe(results_df)
        
        # Select best model
        best_model_name = results_df.iloc[0]['Model']
        self.model = model_results[best_model_name]['model']
        
        st.success(f"🏆 Best model: {best_model_name} (Accuracy: {model_results[best_model_name]['accuracy']:.4f})")
        
        # Educational section on ML algorithms
        st.subheader("📚 Understanding Machine Learning Algorithms")
        
        with st.expander("🤖 About the Algorithms Used"):
            st.write("**🌳 Random Forest:**")
            st.write("• Combines multiple decision trees to make predictions")
            st.write("• Good balance of accuracy and interpretability")
            st.write("• Less prone to overfitting than single decision trees")
            st.write("• Feature importance based on how much each feature improves tree splits")
            
            st.write("**📊 Logistic Regression:**")
            st.write("• Uses statistical relationships to predict probabilities")
            st.write("• Highly interpretable - coefficients show direct feature impact")
            st.write("• Works well for binary classification (yes/no, approve/deny)")
            st.write("• Assumes linear relationship between features and log-odds of outcome")
            
            st.write("**🚀 Gradient Boosting:**")
            st.write("• Builds models sequentially, each correcting previous model's errors")
            st.write("• Often achieves high accuracy but can be complex to interpret")
            st.write("• Risk of overfitting if not properly tuned")
            st.write("• Feature importance based on how often features are used for splitting")
        
        return model_results, label_encoders
    
    def fairness_metrics(self, sensitive_features, model_results, label_encoders):
        """Calculate fairness metrics"""
        st.subheader("📏 Fairness Metrics")
        
        # Check target variable format
        unique_targets = sorted(self.y_test.unique())
        st.write(f"**Target variable values**: {unique_targets}")
        
        # Determine positive label for binary classification
        if len(unique_targets) == 2:
            if set(unique_targets) == {0, 1}:
                pos_label = 1
            elif set(unique_targets) == {-1, 1}:
                pos_label = 1
            else:
                # Convert to binary format
                pos_label = unique_targets[1]  # Use the second value as positive
                st.info(f"Converting target values to binary format. Positive label: {pos_label}")
        else:
            st.warning(f"⚠️ Target variable has {len(unique_targets)} classes. Fairness metrics work best with binary classification.")
            pos_label = unique_targets[-1]  # Use last value as positive
        
        if not FAIRLEARN_AVAILABLE:
            st.warning("⚠️ Fairlearn not installed. Using basic fairness analysis.")
            
            # Basic fairness analysis without fairlearn
            for sensitive_feature in sensitive_features:
                st.write(f"**Basic Fairness Analysis for: {sensitive_feature}**")
                
                # Get sensitive feature for test set
                sensitive_test = self.X_test[sensitive_feature]
                unique_groups = sensitive_test.unique()
                
                for model_name, results in model_results.items():
                    y_pred = results['predictions']
                    
                    st.write(f"**{model_name}**:")
                    
                    # Calculate basic metrics per group
                    group_metrics = {}
                    for group in unique_groups:
                        group_mask = sensitive_test == group
                        if group_mask.sum() > 0:
                            group_accuracy = accuracy_score(
                                self.y_test[group_mask], 
                                y_pred[group_mask]
                            )
                            
                            # Positive prediction rate
                            group_positive_rate = np.mean(y_pred[group_mask] == pos_label)
                            
                            # True positive rate (sensitivity)
                            group_y_true = self.y_test[group_mask]
                            group_y_pred = y_pred[group_mask]
                            
                            if (group_y_true == pos_label).sum() > 0:
                                tpr = np.mean(group_y_pred[group_y_true == pos_label] == pos_label)
                            else:
                                tpr = 0
                            
                            group_metrics[group] = {
                                'accuracy': group_accuracy,
                                'positive_rate': group_positive_rate,
                                'tpr': tpr,
                                'count': group_mask.sum()
                            }
                            
                            st.write(f"  - Group {group} (n={group_mask.sum()}): Accuracy = {group_accuracy:.3f}, Positive Rate = {group_positive_rate:.3f}, TPR = {tpr:.3f}")
                    
                    # Calculate basic fairness metrics
                    if len(group_metrics) >= 2:
                        positive_rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
                        tprs = [metrics['tpr'] for metrics in group_metrics.values()]
                        
                        demographic_parity_diff = max(positive_rates) - min(positive_rates)
                        equalized_odds_diff = max(tprs) - min(tprs)
                        
                        st.write(f"  📊 **Fairness Summary**:")
                        st.write(f"  - Demographic Parity Difference: {demographic_parity_diff:.4f}")
                        st.write(f"  - Equalized Odds Difference: {equalized_odds_diff:.4f}")
                        
                        if demographic_parity_diff > 0.1:
                            st.warning(f"  ⚠️ High demographic parity difference")
                        if equalized_odds_diff > 0.1:
                            st.warning(f"  ⚠️ High equalized odds difference")
            return
        
        # Fairlearn-based analysis with proper label handling
        for sensitive_feature in sensitive_features:
            st.write(f"**Fairness Analysis for: {sensitive_feature}**")
            
            # Get sensitive feature for test set
            sensitive_test = self.X_test[sensitive_feature]
            
            for model_name, results in model_results.items():
                y_pred = results['predictions']
                
                try:
                    # Convert target and predictions to binary if needed
                    y_true_binary = (self.y_test == pos_label).astype(int)
                    y_pred_binary = (y_pred == pos_label).astype(int)
                    
                    # Calculate demographic parity
                    dp_diff = demographic_parity_difference(
                        y_true_binary, y_pred_binary, sensitive_features=sensitive_test
                    )
                    
                    # Calculate equalized odds
                    eo_diff = equalized_odds_difference(
                        y_true_binary, y_pred_binary, sensitive_features=sensitive_test
                    )
                    
                    st.write(f"**{model_name}**:")
                    st.write(f"- Demographic Parity Difference: {dp_diff:.4f}")
                    st.write(f"- Equalized Odds Difference: {eo_diff:.4f}")
                    
                    # Interpretation
                    if abs(dp_diff) > 0.1:
                        st.warning(f"⚠️ High demographic parity difference for {model_name}")
                    else:
                        st.success(f"✅ Acceptable demographic parity for {model_name}")
                        
                    if abs(eo_diff) > 0.1:
                        st.warning(f"⚠️ High equalized odds difference for {model_name}")
                    else:
                        st.success(f"✅ Acceptable equalized odds for {model_name}")
                        
                except Exception as e:
                    st.error(f"Error calculating fairness metrics for {model_name}: {str(e)}")
                    
                    # Fallback to basic analysis
                    st.info("Falling back to basic group-wise analysis...")
                    unique_groups = sensitive_test.unique()
                    
                    for group in unique_groups:
                        group_mask = sensitive_test == group
                        if group_mask.sum() > 0:
                            group_accuracy = accuracy_score(
                                self.y_test[group_mask], 
                                y_pred[group_mask]
                            )
                            group_positive_rate = np.mean(y_pred[group_mask] == pos_label)
                            st.write(f"  - Group {group}: Accuracy = {group_accuracy:.3f}, Positive Rate = {group_positive_rate:.3f}")
                    
                    continue
    
        # Educational section on fairness metrics
        st.subheader("📚 Understanding Fairness Metrics")
        
        with st.expander("⚖️ Fairness Metrics Explained"):
            st.write("**📊 Demographic Parity:**")
            st.write("• **Acceptable**: Different groups receive positive outcomes at similar rates")
            st.write("• **Unacceptable**: One group gets approved 80% of time, another only 40%")
            st.write("• Example: Equal loan approval rates across racial groups")
            st.write("• May conflict with merit-based decisions if groups have different qualifications")
            
            st.write("**⚖️ Equalized Odds:**")
            st.write("• **Acceptable**: Model has similar accuracy for all groups")
            st.write("• **Unacceptable**: Model correctly identifies 90% of qualified applicants from one group but only 60% from another")
            st.write("• Focuses on equal treatment of truly qualified individuals")
            st.write("• Generally preferred over demographic parity for merit-based decisions")
            
            st.write("**🎯 Which Metric to Use:**")
            st.write("• **Use Demographic Parity** when equal representation is the goal")
            st.write("• **Use Equalized Odds** when equal treatment based on merit is the goal")
            st.write("• Consider business context and legal requirements when choosing")
    
    def explainability_analysis(self):
        """SHAP-based model explainability"""
        st.header("🔍 Model Explainability")
        
        # Page description
        st.info("🔍 **About Explainability Analysis:** This step reveals how your AI model makes decisions by identifying which features are most important for predictions. Understanding model behavior is crucial for trust, regulatory compliance, and detecting potential bias in decision-making patterns.")
        
        if self.model is None:
            st.warning("Please train a model first.")
            return
        
        st.subheader("🎯 Model Interpretability Analysis")
        
        # Display model information
        model_name = type(self.model).__name__
        st.write(f"**Analyzing Model**: {model_name}")
        
        # Check target variable for SHAP compatibility
        unique_targets = sorted(self.y_test.unique())
        is_binary = len(unique_targets) == 2
        st.write(f"**Target Classes**: {unique_targets} ({'Binary' if is_binary else 'Multi-class'})")
        
        # Try multiple explanation approaches
        explanation_success = False
        
        if SHAP_AVAILABLE:
            st.subheader("🔬 SHAP Analysis")
            
            try:
                # Convert target to binary format for SHAP if needed
                if not (set(unique_targets) == {0, 1} or set(unique_targets) == {-1, 1}):
                    st.info(f"Converting target values to binary format for SHAP analysis...")
                    # Create binary version
                    pos_label = unique_targets[-1]  # Use last value as positive
                    y_test_binary = (self.y_test == pos_label).astype(int)
                    st.write(f"Positive label: {pos_label} → 1, Others → 0")
                else:
                    y_test_binary = self.y_test
                
                # Ensure data is in proper format (convert to numpy arrays)
                # Fix: Properly handle DataFrame to numpy conversion for SHAP
                if hasattr(self.X_train, 'values'):
                    X_train_np = self.X_train.values.astype(np.float32)
                else:
                    X_train_np = np.array(self.X_train, dtype=np.float32)
                    
                if hasattr(self.X_test, 'values'):
                    X_test_np = self.X_test.values.astype(np.float32)
                else:
                    X_test_np = np.array(self.X_test, dtype=np.float32)
                
                # Verify shapes and data types
                st.write(f"**Data for SHAP**: Training shape: {X_train_np.shape}, Test shape: {X_test_np.shape}")
                st.write(f"**Data types**: Training: {X_train_np.dtype}, Test: {X_test_np.dtype}")
                
                # Get feature names before conversion
                if hasattr(self.X_test, 'columns'):
                    feature_names = self.X_test.columns.tolist()
                else:
                    feature_names = [f'feature_{i}' for i in range(X_test_np.shape[1])]
                
                st.write(f"**Features for SHAP analysis**: {len(feature_names)} features")
                st.write(f"**Feature names**: {feature_names[:5]}..." if len(feature_names) > 5 else f"**Feature names**: {feature_names}")
                
                # Choose appropriate SHAP explainer based on model type
                if hasattr(self.model, 'estimators_') and 'Forest' in model_name:
                    # Random Forest
                    st.write("Using TreeExplainer for Random Forest...")
                    explainer = shap.TreeExplainer(self.model)
                    
                    # Use smaller sample for efficiency
                    sample_size = min(30, len(X_test_np))
                    X_sample = X_test_np[:sample_size]
                    
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Handle multi-output case
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values_to_plot = shap_values[1]  # Use positive class
                        st.info("Using SHAP values for positive class (class 1)")
                    else:
                        shap_values_to_plot = shap_values
                    
                elif 'Gradient' in model_name:
                    # Gradient Boosting
                    st.write("Using TreeExplainer for Gradient Boosting...")
                    explainer = shap.TreeExplainer(self.model)
                    
                    sample_size = min(30, len(X_test_np))
                    X_sample = X_test_np[:sample_size]
                    
                    shap_values = explainer.shap_values(X_sample)
                    
                    # For binary classification, GradientBoosting might return single array
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values_to_plot = shap_values[1]  # Positive class
                        st.info("Using SHAP values for positive class")
                    else:
                        shap_values_to_plot = shap_values
                    
                elif 'Logistic' in model_name:
                    # Logistic Regression
                    st.write("Using LinearExplainer for Logistic Regression...")
                    
                    sample_size = min(30, len(X_test_np))
                    X_sample = X_test_np[:sample_size]
                    X_background = X_train_np[:100]  # Background for linear explainer
                    
                    explainer = shap.LinearExplainer(self.model, X_background)
                    shap_values = explainer.shap_values(X_sample)
                    shap_values_to_plot = shap_values
                    
                else:
                    # Fallback to KernelExplainer (slower but works for any model)
                    st.write("Using KernelExplainer (this may take a moment)...")
                    
                    # Use very small samples for KernelExplainer
                    sample_size = min(20, len(X_test_np))
                    background_size = min(50, len(X_train_np))
                    
                    X_sample = X_test_np[:sample_size]
                    X_background = X_train_np[:background_size]
                    
                    explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                    shap_values = explainer.shap_values(X_sample)
                    
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values_to_plot = shap_values[1]  # Positive class
                    else:
                        shap_values_to_plot = shap_values
                
                # Calculate feature importance from SHAP values
                if shap_values_to_plot is not None:
                    # Ensure shap_values is 2D array
                    if len(shap_values_to_plot.shape) == 1:
                        shap_values_to_plot = shap_values_to_plot.reshape(1, -1)
                    
                    # Calculate mean absolute SHAP values for feature importance
                    feature_importance_shap = np.abs(shap_values_to_plot).mean(0)
                    
                    # Ensure lengths match
                    if len(feature_importance_shap) != len(feature_names):
                        min_len = min(len(feature_importance_shap), len(feature_names))
                        feature_importance_shap = feature_importance_shap[:min_len]
                        feature_names = feature_names[:min_len]
                    
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP_Importance': feature_importance_shap
                    }).sort_values('SHAP_Importance', ascending=False)
                    
                    st.subheader("📊 SHAP Feature Importance")
                    
                    # Enhanced analysis of feature importance
                    st.write("**🎯 Key Insights from Feature Importance:**")
                    
                    # Top 3 most important features
                    top_3_features = feature_importance_df.head(3)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🥇 Most Important Feature",
                            value=top_3_features.iloc[0]['Feature'],
                            delta=f"SHAP: {top_3_features.iloc[0]['SHAP_Importance']:.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="🥈 Second Most Important",
                            value=top_3_features.iloc[1]['Feature'],
                            delta=f"SHAP: {top_3_features.iloc[1]['SHAP_Importance']:.4f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="🥉 Third Most Important",
                            value=top_3_features.iloc[2]['Feature'],
                            delta=f"SHAP: {top_3_features.iloc[2]['SHAP_Importance']:.4f}"
                        )
                    
                    # Feature importance distribution analysis
                    total_importance = feature_importance_df['SHAP_Importance'].sum()
                    top_5_contribution = feature_importance_df.head(5)['SHAP_Importance'].sum() / total_importance * 100
                    
                    st.write(f"**📈 Feature Concentration Analysis:**")
                    st.write(f"• Top 5 features contribute **{top_5_contribution:.1f}%** of total model decisions")
                    
                    if top_5_contribution > 80:
                        st.warning("⚠️ **High Feature Concentration**: Model relies heavily on few features - consider feature engineering or regularization")
                    elif top_5_contribution < 40:
                        st.info("📊 **Distributed Importance**: Model uses many features relatively equally - good for robustness")
                    else:
                        st.success("✅ **Balanced Feature Usage**: Healthy distribution of feature importance")
                    
                    # Bias risk assessment based on feature importance
                    st.write("**⚖️ Fairness Risk Assessment:**")
                    sensitive_in_top_10 = []
                    top_10_features = feature_importance_df.head(10)['Feature'].tolist()
                    
                    # Check if sensitive features appear in top important features
                    for feature in top_10_features:
                        feature_lower = feature.lower()
                        if any(sensitive_word in feature_lower for sensitive_word in 
                               ['gender', 'sex', 'race', 'ethnicity', 'age', 'religion', 'marital']):
                            sensitive_in_top_10.append(feature)
                    
                    if sensitive_in_top_10:
                        st.error(f"🚨 **Bias Risk Detected**: Sensitive features {sensitive_in_top_10} are highly important for predictions")
                        st.write("**Recommendation**: Implement fairness constraints or consider feature removal/transformation")
                    else:
                        st.success("✅ **Low Bias Risk**: No obviously sensitive features in top 10 most important")
                    
                    # Plot top features
                    top_features = feature_importance_df.head(10)
                    fig = px.bar(top_features, x='SHAP_Importance', y='Feature', 
                                orientation='h', title="Top 10 Feature Importance (SHAP)")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)
                    
                    st.dataframe(feature_importance_df.head(15))
                    
                    # Add SHAP summary plot if possible
                    try:
                        st.subheader("📈 SHAP Summary Plot")
                        st.write("*Note: This shows how each feature impacts the model's predictions*")
                        
                        # Create SHAP summary plot
                        fig_shap, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values_to_plot, X_sample, 
                                        feature_names=feature_names, 
                                        plot_type="dot", show=False, ax=ax)
                        st.pyplot(fig_shap)
                        plt.close()
                        
                    except Exception as plot_error:
                        st.info(f"Could not generate SHAP summary plot: {str(plot_error)}")
                    
                    explanation_success = True
                    st.success("✅ SHAP analysis completed successfully!")
                
            except Exception as e:
                st.error(f"SHAP analysis failed: {str(e)}")
                st.info("This could be due to:")
                st.write("- Complex model architecture")
                st.write("- Data format incompatibility") 
                st.write("- Memory limitations with large datasets")
                st.info("Falling back to alternative explanation methods...")
        else:
            st.warning("⚠️ SHAP not installed. Install with: `pip install shap`")
        
        # Fallback to model's built-in feature importance
        if not explanation_success:
            st.subheader("📈 Model-Based Feature Importance")
            
            if hasattr(self.model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': self.X_test.columns,
                    'Importance': self.model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                top_features = importance_df.head(10)
                fig = px.bar(top_features, x='Importance', y='Feature',
                           orientation='h', title="Top 10 Feature Importance (Model Built-in)")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)
                
                st.dataframe(importance_df.head(15))
                explanation_success = True
                
            elif hasattr(self.model, 'coef_'):
                # For linear models
                coef_importance = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
                
                importance_df = pd.DataFrame({
                    'Feature': self.X_test.columns,
                    'Coefficient_Magnitude': coef_importance
                }).sort_values('Coefficient_Magnitude', ascending=False)
                
                # Plot coefficient importance
                top_features = importance_df.head(10)
                fig = px.bar(top_features, x='Coefficient_Magnitude', y='Feature',
                           orientation='h', title="Top 10 Feature Importance (Coefficients)")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)
                
                st.dataframe(importance_df.head(15))
                explanation_success = True
        
        # Permutation importance as additional fallback
        if not explanation_success:
            st.subheader("🔄 Permutation Importance")
            try:
                from sklearn.inspection import permutation_importance
                
                perm_importance = permutation_importance(
                    self.model, self.X_test, self.y_test, 
                    n_repeats=5, random_state=42
                )
                
                importance_df = pd.DataFrame({
                    'Feature': self.X_test.columns,
                    'Permutation_Importance': perm_importance.importances_mean,
                    'Std': perm_importance.importances_std
                }).sort_values('Permutation_Importance', ascending=False)
                
                # Plot permutation importance
                top_features = importance_df.head(10)
                fig = px.bar(top_features, x='Permutation_Importance', y='Feature',
                           orientation='h', title="Top 10 Feature Importance (Permutation)",
                           error_x='Std')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)
                
                st.dataframe(importance_df.head(15))
                explanation_success = True
                
            except Exception as e:
                st.error(f"Permutation importance failed: {str(e)}")
        
        # Model complexity analysis
        st.subheader("🔧 Model Complexity Analysis")
        
        complexity_info = {}
        
        if hasattr(self.model, 'n_estimators'):
            complexity_info['Number of Estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            complexity_info['Max Depth'] = self.model.max_depth
        
        if hasattr(self.model, 'n_features_in_'):
            complexity_info['Number of Features'] = self.model.n_features_in_
        
        # Display complexity metrics
        if complexity_info:
            col1, col2, col3 = st.columns(3)
            items = list(complexity_info.items())
            
            for i, (key, value) in enumerate(items):
                if i % 3 == 0 and i // 3 < 1:
                    col1.metric(key, value)
                elif i % 3 == 1 and i // 3 < 1:
                    col2.metric(key, value)
                elif i % 3 == 2 and i // 3 < 1:
                    col3.metric(key, value)
        
        # Interpretability recommendations
        st.subheader("💡 Interpretability Recommendations")
        
        recommendations = []
        
        if 'Gradient' in model_name:
            recommendations.append("🌳 Consider using Random Forest for better explainability")
            recommendations.append("📊 Gradient Boosting models can be complex - monitor for overfitting")
        
        if 'Forest' in model_name:
            recommendations.append("🌲 Random Forest provides good balance of accuracy and interpretability")
            recommendations.append("🔍 Consider analyzing individual tree decisions for specific predictions")
        
        if 'Linear' in model_name:
            recommendations.append("📈 Linear models are highly interpretable")
            recommendations.append("⚖️ Coefficients directly show feature impact on predictions")
        
        recommendations.extend([
            "📝 Document key features and their business relevance",
            "🔄 Regularly validate model explanations with domain experts",
            "📊 Consider LIME for local explanations of individual predictions"
        ])
        
        for rec in recommendations:
            st.write(rec)
        
        if not explanation_success:
            st.error("❌ Unable to generate feature importance analysis. Please check your model and data.")
        else:
            st.success("✅ Model explainability analysis completed!")
        
        # Educational section on key concepts
        st.subheader("📚 Understanding Key Concepts")
        
        # Create expandable sections for each concept
        with st.expander("🤖 Machine Learning Algorithms Explained"):
            st.write("**🌳 Random Forest:**")
            st.write("• Combines multiple decision trees to make predictions")
            st.write("• Good balance of accuracy and interpretability")
            st.write("• Less prone to overfitting than single decision trees")
            st.write("• Feature importance based on how much each feature improves tree splits")
            
            st.write("**📊 Logistic Regression:**")
            st.write("• Uses statistical relationships to predict probabilities")
            st.write("• Highly interpretable - coefficients show direct feature impact")
            st.write("• Works well for binary classification (yes/no, approve/deny)")
            st.write("• Assumes linear relationship between features and log-odds of outcome")
            
            st.write("**🚀 Gradient Boosting:**")
            st.write("• Builds models sequentially, each correcting previous model's errors")
            st.write("• Often achieves high accuracy but can be complex to interpret")
            st.write("• Risk of overfitting if not properly tuned")
            st.write("• Feature importance based on how often features are used for splitting")
        
        with st.expander("⚖️ Fairness Metrics Explained"):
            st.write("**📊 Demographic Parity:**")
            st.write("• **Acceptable**: Different groups receive positive outcomes at similar rates")
            st.write("• **Unacceptable**: One group gets approved 80% of time, another only 40%")
            st.write("• Example: Equal loan approval rates across racial groups")
            st.write("• May conflict with merit-based decisions if groups have different qualifications")
            
            st.write("**⚖️ Equalized Odds:**")
            st.write("• **Acceptable**: Model has similar accuracy for all groups")
            st.write("• **Unacceptable**: Model correctly identifies 90% of qualified applicants from one group but only 60% from another")
            st.write("• Focuses on equal treatment of truly qualified individuals")
            st.write("• Generally preferred over demographic parity for merit-based decisions")
            
            st.write("**🎯 Which Metric to Use:**")
            st.write("• **Use Demographic Parity** when equal representation is the goal")
            st.write("• **Use Equalized Odds** when equal treatment based on merit is the goal")
            st.write("• Consider business context and legal requirements when choosing")
    
    def generate_governance_report(self, risk_df, sensitive_features, identified_sensitive):
        """Generate comprehensive governance report"""
        st.header("📋 Governance Report")
        
        # Page description
        st.info("📋 **About Governance Report:** This comprehensive report summarizes all findings from privacy, bias, and explainability analyses. It provides executive-level insights, actionable recommendations, and regulatory compliance guidance to help you deploy AI responsibly and meet industry standards.")
        
        report_data = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_records': len(self.data),
                'total_features': len(self.data.columns),
                'missing_values': self.data.isnull().sum().sum()
            },
            'privacy_assessment': {
                'high_risk_features': risk_df[risk_df['Privacy Risk'] >= 3]['Feature'].tolist(),
                'identified_sensitive_categories': list(identified_sensitive.keys())
            },
            'bias_assessment': {
                'analyzed_sensitive_features': sensitive_features,
                'recommendations': []
            }
        }
        
        # Display summary
        st.subheader("Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔍 High-Risk Features Analysis**")
            
            # Show actionable privacy risks
            high_risk_features = risk_df[risk_df['Privacy Risk'] >= 3]
            critical_risk_features = risk_df[risk_df['Privacy Risk'] >= 4]
            
            if len(critical_risk_features) > 0:
                st.error(f"🚨 **{len(critical_risk_features)} CRITICAL risk features** need immediate attention")
                
                # Explain what makes features critical
                with st.expander("❓ Why are these features Critical Risk?", expanded=True):
                    st.warning("**Critical Risk Features** are identified because they:")
                    st.write("🔍 **High Uniqueness**: >90% unique values (likely personal identifiers)")
                    st.write("🏷️ **Sensitive Patterns**: Names contain 'id', 'ssn', 'email', 'phone', 'passport'")
                    st.write("🌍 **Location Data**: Addresses, coordinates, postal codes")
                    st.write("💳 **Financial IDs**: Account numbers, credit card info")
                    st.write("⚠️ **Re-identification Risk**: Could identify individuals when combined")
                
                st.write("**🚨 Critical Risk Features Found:**")
                for _, row in critical_risk_features.head(3).iterrows():
                    st.write(f"• **{row['Feature']}** (Risk Level: {row['Risk Level']})")
            elif len(high_risk_features) > 0:
                st.warning(f"⚠️ **{len(high_risk_features)} high-risk features** need review")
                for _, row in high_risk_features.head(3).iterrows():
                    st.write(f"• **{row['Feature']}** (Risk Level: {row['Risk Level']})")
            else:
                st.success("✅ **No critical privacy risks** detected in current features")
                
            # Show dataset-specific insights
            total_features = len(risk_df)
            low_risk_count = len(risk_df[risk_df['Privacy Risk'] <= 2])
            st.info(f"📊 **Privacy Summary**: {low_risk_count}/{total_features} features are low-risk")
        
        with col2:
            st.write("**🎯 Domain-Specific Insights**")
            
            # Show domain-specific analysis
            column_names = [col.lower() for col in self.data.columns]
            domain_insights = []
            
            if any('loan' in col or 'credit' in col for col in column_names):
                domain_insights.append("🏦 **Financial Services** data detected")
                domain_insights.append("→ FCRA compliance required")
                
            if any('person' in col or 'gender' in col for col in column_names):
                domain_insights.append("👤 **Personal Demographics** present")
                domain_insights.append("→ EEOC bias monitoring needed")
                
            if any('income' in col or 'salary' in col for col in column_names):
                domain_insights.append("💰 **Financial Information** identified")
                domain_insights.append("→ Consider differential privacy")
                
            if any('age' in col for col in column_names):
                domain_insights.append("📅 **Age Information** found")
                domain_insights.append("→ ADEA compliance monitoring")
            
            if domain_insights:
                for insight in domain_insights:
                    st.write(insight)
            else:
                st.write("📋 **General dataset** - no specific domain patterns detected")
                
            # Dataset size context
            total_records = len(self.data)
            if total_records < 1000:
                st.write(f"⚠️ **Small dataset** ({total_records:,} records)")
                st.write("→ Limited statistical power for bias detection")
            elif total_records > 100000:
                st.write(f"📈 **Large dataset** ({total_records:,} records)")  
                st.write("→ Consider sampling for efficient analysis")
        
        # Show actionable privacy risk visualization
        st.subheader("📊 Privacy Risk Analysis")
        
        # Create a more useful visualization
        high_risk_features = risk_df[risk_df['Privacy Risk'] >= 2].sort_values('Privacy Risk', ascending=True)
        
        if len(high_risk_features) > 0:
            # Bar chart showing specific risky features
            fig = px.bar(
                high_risk_features.tail(10),  # Show top 10 risky features
                x='Privacy Risk', 
                y='Feature',
                color='Risk Level',
                orientation='h',
                title="🔍 Features Requiring Privacy Attention",
                color_discrete_map={
                    'Critical': '#ff4444',
                    'Very High': '#ff8800', 
                    'High': '#ffaa00',
                    'Medium': '#ffdd00',
                    'Low': '#88dd00',
                    'Very Low': '#44dd44'
                }
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table of risky features
            st.write("**🎯 Priority Features for Review:**")
            priority_features = high_risk_features[['Feature', 'Risk Level', 'Privacy Risk']].tail(5)
            st.dataframe(priority_features, use_container_width=True)
        else:
            st.success("✅ **Excellent Privacy Profile** - No features require special attention")
            st.info("All features have low privacy risk scores")
        
        # Generate specialized recommendations based on actual analysis
        recommendations = self._generate_specialized_recommendations(
            risk_df, sensitive_features, identified_sensitive, report_data
        )
        
        # Recommendations
        st.subheader("🎯 Data-Specific Key Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Export functionality
        st.subheader("📤 Export Report")
        
        if st.button("Generate Detailed Report"):
            # Get bias metrics if available
            bias_metrics_text = ""
            if hasattr(self, 'model') and self.model is not None and hasattr(st.session_state, 'model_results'):
                model_results = st.session_state.model_results
                best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
                best_accuracy = model_results[best_model_name]['accuracy']
                
                bias_metrics_text = f"""
## Model Performance & Bias Analysis
### Best Model: {best_model_name}
- Accuracy: {best_accuracy:.4f}
- Analyzed Sensitive Features: {', '.join(sensitive_features)}

### Bias Analysis Results:
- Target Variable: {st.session_state.target_col if hasattr(st.session_state, 'target_col') else 'Not specified'}
- Sensitive Features Analyzed: {len(sensitive_features)} features
- Fairness Metrics: {"Calculated with Fairlearn" if globals().get('FAIRLEARN_AVAILABLE', False) else "Basic analysis performed"}

### Bias Risk Assessment:
{chr(10).join([f"- {feature}: Requires fairness monitoring for equal treatment" for feature in sensitive_features])}
"""
            else:
                bias_metrics_text = """
## Model Performance & Bias Analysis
- Model Training: Not completed
- Bias Analysis: Sensitive features identified but model training required for quantitative bias assessment
"""

            # Get critical risk features with explanations
            critical_risk_features = risk_df[risk_df['Privacy Risk'] >= 4]['Feature'].tolist()
            critical_risk_text = ""
            if critical_risk_features:
                critical_risk_text = f"""
### ⚠️ CRITICAL PRIVACY RISKS IDENTIFIED:
{chr(10).join([f"- {feature}: Requires immediate anonymization or removal" for feature in critical_risk_features])}

**Why Critical:** These features likely contain personally identifiable information (PII) with >90% unique values or sensitive naming patterns (ID, SSN, email, phone, etc.)
"""
            
            # Create comprehensive report
            report_text = f"""
# AI Model Governance Report
Generated on: {report_data['report_date']}

## Dataset Overview
- Total Records: {report_data['dataset_info']['total_records']:,}
- Total Features: {report_data['dataset_info']['total_features']}
- Missing Values: {report_data['dataset_info']['missing_values']}

## Privacy Assessment
### High-Risk Features:
{chr(10).join(f"- {feature}" for feature in report_data['privacy_assessment']['high_risk_features'])}

{critical_risk_text}

### Sensitive Data Categories Identified:
{chr(10).join(f"- {category.title()}: {len(identified_sensitive[category])} features" for category in report_data['privacy_assessment']['identified_sensitive_categories'])}
{bias_metrics_text}
## Data-Specific Recommendations
{chr(10).join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])}

## Compliance Checklist
✅ Data inventory completed
✅ Privacy risk assessment conducted  
✅ Sensitive features identified
{"✅ Bias analysis performed" if sensitive_features else "⚠️ Bias analysis pending"}
{"✅ Model explainability provided" if hasattr(self, 'model') and self.model is not None else "⚠️ Model training required"}

## Next Steps
1. Address critical privacy risks immediately
2. {"Complete model training and bias assessment" if not hasattr(self, 'model') else "Implement fairness constraints"}
3. Set up continuous monitoring
4. Document model cards for compliance
            """
            
            st.download_button(
                label="Download Report",
                data=report_text,
                file_name=f"governance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def _generate_specialized_recommendations(self, risk_df, sensitive_features, identified_sensitive, report_data):
        """Generate specialized recommendations based on actual data analysis"""
        recommendations = []
        
        # Privacy-specific recommendations
        high_risk_features = risk_df[risk_df['Privacy Risk'] >= 3]['Feature'].tolist()
        critical_risk_features = risk_df[risk_df['Privacy Risk'] >= 4]['Feature'].tolist()
        
        if critical_risk_features:
            recommendations.append(
                f"🚨 **CRITICAL PRIVACY RISK**: Immediately review columns {', '.join(critical_risk_features)} - "
                f"these likely contain personally identifiable information that should be anonymized or removed"
            )
        elif high_risk_features:
            recommendations.append(
                f"🔴 **High Privacy Priority**: Assess columns {', '.join(high_risk_features)} for anonymization - "
                f"these have high uniqueness or contain sensitive patterns"
            )
        else:
            recommendations.append("✅ **Privacy Assessment**: No critical privacy risks detected in current features")
        
        # Sensitive feature recommendations
        if 'personal_id' in identified_sensitive:
            id_features = identified_sensitive['personal_id']
            recommendations.append(
                f"🔐 **Identity Protection**: Remove or hash identifier columns {', '.join(id_features)} "
                f"before model deployment to prevent re-identification"
            )
        
        if 'location' in identified_sensitive:
            location_features = identified_sensitive['location']
            recommendations.append(
                f"🌍 **Location Privacy**: Consider geographic aggregation for {', '.join(location_features)} "
                f"to reduce location-based discrimination while preserving utility"
            )
        
        if 'financial' in identified_sensitive:
            financial_features = identified_sensitive['financial']
            recommendations.append(
                f"💰 **Financial Data**: Apply differential privacy or binning to {', '.join(financial_features)} "
                f"to protect sensitive financial information"
            )
        
        # Bias-specific recommendations
        if sensitive_features:
            if 'gender' in str(sensitive_features).lower() or 'gender' in identified_sensitive:
                recommendations.append(
                    "⚖️ **Gender Bias Mitigation**: Implement fairness constraints (demographic parity or equalized odds) "
                    "to ensure equal treatment across gender groups - required for EEOC compliance"
                )
            
            if 'race' in str(sensitive_features).lower() or 'race' in identified_sensitive:
                recommendations.append(
                    "🤝 **Racial Fairness**: Apply bias testing and mitigation techniques - "
                    "consider using fairlearn's postprocessing methods for equal opportunity"
                )
            
            if 'age' in str(sensitive_features).lower() or 'age' in identified_sensitive:
                recommendations.append(
                    "👥 **Age Discrimination**: Ensure compliance with Age Discrimination in Employment Act (ADEA) - "
                    "monitor for disparate impact on age groups 40+"
                )
        else:
            recommendations.append(
                "📊 **Bias Analysis**: Consider analyzing additional demographic features for fairness assessment"
            )
        
        # Dataset size and quality recommendations
        total_records = report_data['dataset_info']['total_records']
        missing_pct = (report_data['dataset_info']['missing_values'] / 
                      (total_records * report_data['dataset_info']['total_features'])) * 100
        
        if total_records < 1000:
            recommendations.append(
                f"⚠️ **Sample Size**: Dataset has only {total_records:,} records - "
                "consider collecting more data for robust model performance and bias detection"
            )
        elif total_records > 100000:
            recommendations.append(
                f"📈 **Large Dataset**: With {total_records:,} records, implement sampling strategies "
                "for efficient bias monitoring and consider distributed fairness assessment"
            )
        
        if missing_pct > 10:
            recommendations.append(
                f"🔧 **Data Quality**: {missing_pct:.1f}% missing data detected - "
                "implement robust missing value strategies and document impact on fairness metrics"
            )
        
        # Domain-specific recommendations
        column_names = [col.lower() for col in self.data.columns]
        
        if any('loan' in col or 'credit' in col for col in column_names):
            recommendations.append(
                "🏦 **Financial Services**: Implement FCRA compliance measures, document model decisions "
                "for adverse action notices, and regularly audit for redlining patterns"
            )
        
        if any('medical' in col or 'health' in col for col in column_names):
            recommendations.append(
                "🏥 **Healthcare Data**: Ensure HIPAA compliance, implement health equity monitoring, "
                "and consider social determinants of health in fairness assessment"
            )
        
        if any('employ' in col or 'hire' in col or 'job' in col for col in column_names):
            recommendations.append(
                "💼 **Employment Decisions**: Follow EEOC guidelines, implement 4/5ths rule testing, "
                "and document job-relatedness of all features used in hiring models"
            )
        
        # Model performance and monitoring recommendations
        if hasattr(self, 'model') and self.model is not None:
            model_name = type(self.model).__name__
            
            if 'Forest' in model_name:
                recommendations.append(
                    "🌳 **Random Forest Monitoring**: Set up feature importance drift detection "
                    "and monitor for changes in tree structure that might indicate bias shifts"
                )
            elif 'Gradient' in model_name:
                recommendations.append(
                    "📊 **Gradient Boosting Oversight**: Implement early stopping to prevent overfitting "
                    "on sensitive attributes and monitor boosting iterations for bias amplification"
                )
            elif 'Linear' in model_name or 'Logistic' in model_name:
                recommendations.append(
                    "📈 **Linear Model Transparency**: Leverage coefficient interpretability for bias explanation "
                    "and implement regular coefficient stability monitoring"
                )
        
        # Regulatory and compliance recommendations
        recommendations.append(
            "📋 **Documentation**: Maintain model cards documenting training data, performance metrics, "
            "limitations, and fairness evaluations for regulatory compliance"
        )
        
        recommendations.append(
            "🔄 **Continuous Monitoring**: Implement automated bias monitoring in production "
            "with alerts for fairness metric degradation and regular retraining schedules"
        )
        
        # Training and organizational recommendations
        if len(identified_sensitive) > 3:
            recommendations.append(
                "🎓 **Team Training**: Conduct specialized bias training given the high number of sensitive attributes - "
                "ensure team understands intersectional bias and multiple protected class interactions"
            )
        else:
            recommendations.append(
                "📚 **Basic Training**: Ensure team completes responsible AI training covering "
                "bias detection, fairness metrics, and ethical considerations"
            )
        
        return recommendations

def main():
    st.set_page_config(
        page_title="AI Model Governance & Fairness Analyzer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ AI Model Governance & Fairness Analyzer")
    st.markdown("Comprehensive analysis for bias, privacy, and governance compliance")
    
    # Check for optional dependencies and show installation guide
    if not SHAP_AVAILABLE:
        with st.expander("🔧 Optional Dependencies Setup", expanded=True):
            st.error("**⚠️ SHAP not installed** - Advanced explainability features are unavailable")
            
            st.write("**To enable SHAP analysis:**")
            st.code("pip install shap", language="bash")
            
            st.info("**What SHAP provides:**")
            st.write("✅ Detailed feature importance analysis")
            st.write("✅ Individual prediction explanations") 
            st.write("✅ Model behavior visualization")
            st.write("✅ Trustworthy AI insights")
            
            st.warning("**Note**: After installing SHAP, restart your Streamlit application.")
    
    analyzer = ModelGovernanceAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("📋 Analysis Steps")
    steps = [
        "1. Load Data",
        "1.5 Handle Missing Values",
        "2. Data Overview", 
        "3. Privacy Analysis",
        "4. Bias Analysis",
        "5. Model Training",
        "6. Explainability",
        "7. Governance Report"
    ]
    
    selected_step = st.sidebar.radio("Select Analysis Step:", steps)
    
    # File upload
    if "1. Load Data" in selected_step:
        st.header("📁 Data Loading")
        
        st.subheader("Choose Data Source")
        
        # Radio button for data source selection
        data_source = st.radio(
            "Select your data source:",
            ["Use Sample Loan Data (45K records)", "Use Large Loan Default Data (148K records)", "Upload Your Own Dataset"],
            help="Choose from sample datasets or upload your own CSV file"
        )
        
        if data_source == "Use Sample Loan Data (45K records)":
            default_path = "Datasets/loan_data.csv"
            st.info("📊 **Sample Loan Data**: Clean, structured dataset with person demographics, loan details, and approval status. Perfect for getting started!")
            
            # Show dataset preview info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", "45K")
            with col2:
                st.metric("Features", "14")
            with col3:
                st.metric("Type", "Clean & Simple")
            
            # Show sample features
            st.write("**Key Features Include:**")
            sample_features = [
                "👤 **Demographics**: person_age, person_gender, person_education",
                "💰 **Financial**: person_income, credit_score, loan_amnt", 
                "🎯 **Target**: loan_status (0=Denied, 1=Approved)",
                "🏠 **Other**: person_home_ownership, loan_intent, loan_int_rate"
            ]
            for feature in sample_features:
                st.write(feature)
                
            if st.button("Load Sample Loan Data", type="primary"):
                if analyzer.load_data(default_path):
                    st.session_state.data_loaded = True
                    st.session_state.analyzer = analyzer
                    
        elif data_source == "Use Large Loan Default Data (148K records)":
            large_path = "Datasets/Loan_Default.csv"
            st.info("📈 **Large Loan Default Data**: Comprehensive dataset with more complex features and larger sample size. Great for advanced analysis!")
            
            # Show dataset preview info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Records", "148K")
            with col2:
                st.metric("Features", "34")
            with col3:
                st.metric("Type", "Complex & Large")
            
            # Show sample features
            st.write("**Key Features Include:**")
            complex_features = [
                "👤 **Demographics**: Gender, age, Region",
                "💰 **Financial**: income, Credit_Score, loan_amount, rate_of_interest",
                "🎯 **Target**: Status (0=Default, 1=No Default)", 
                "🏠 **Property**: property_value, construction_type, occupancy_type",
                "📊 **Advanced**: LTV, dtir1, Interest_rate_spread, total_units"
            ]
            for feature in complex_features:
                st.write(feature)
                
            st.warning("⚠️ **Note**: This dataset is larger and more complex. Processing may take longer but provides richer analysis opportunities.")
                
            if st.button("Load Large Loan Default Data", type="primary"):
                if analyzer.load_data(large_path):
                    st.session_state.data_loaded = True
                    st.session_state.analyzer = analyzer
                    
        else:  # Upload Your Own Dataset
            st.info("📤 **Upload Your Own Data**: Upload a CSV file with your own dataset for custom analysis.")
            
            st.write("**Requirements for your dataset:**")
            st.write("✅ CSV format with headers")
            st.write("✅ At least 100 rows recommended")
            st.write("✅ Include categorical target variable (for bias analysis)")
            st.write("✅ Include demographic features (for fairness analysis)")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv",
                help="Upload your CSV file. Make sure it has headers and includes both demographic features and a target variable."
            )
            
            if uploaded_file is not None:
                if analyzer.load_data(uploaded_file):
                    st.session_state.data_loaded = True
                    st.session_state.analyzer = analyzer
    
    # Check if data is loaded
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        analyzer = st.session_state.analyzer
        
        if "1.5 Handle Missing Values" in selected_step:
            # Check if data has missing values
            missing_count = analyzer.data.isnull().sum().sum()
            if missing_count == 0:
                st.success("✅ No missing values found in your dataset!")
                st.info("You can skip this step and proceed to the next analysis.")
            else:
                st.info(f"Found {missing_count} missing values in the dataset. Let's handle them!")
                
                # Handle missing values
                if analyzer.handle_missing_values():
                    st.session_state.data_cleaned = True
                    st.session_state.analyzer = analyzer
                    
                    # Show guidance for next steps
                    st.success("🎉 Missing values handled successfully!")
                    st.info("👆 You can now proceed to **Step 2: Data Overview** or any other analysis step.")
        
        elif "2. Data Overview" in selected_step:
            info_df = analyzer.data_overview()
        
        elif "3. Privacy Analysis" in selected_step:
            risk_df, identified_sensitive = analyzer.privacy_analysis()
            st.session_state.risk_df = risk_df
            st.session_state.identified_sensitive = identified_sensitive
        
        elif "4. Bias Analysis" in selected_step:
            try:
                selected_sensitive, target_col = analyzer.bias_analysis()
                if selected_sensitive and target_col:
                    st.session_state.selected_sensitive = selected_sensitive
                    st.session_state.target_col = target_col
                    st.success("✅ Bias analysis completed successfully!")
                else:
                    st.info("ℹ️ Please complete the sensitive feature and target variable selection to proceed.")
            except Exception as e:
                st.error(f"❌ Error in bias analysis: {str(e)}")
                st.info("Please check your data and try again.")
        
        elif "5. Model Training" in selected_step:
            if hasattr(st.session_state, 'selected_sensitive') and hasattr(st.session_state, 'target_col'):
                # Check if model is already trained for the current target and sensitive features
                cache_key = f"{st.session_state.target_col}_{str(st.session_state.selected_sensitive)}"
                if (
                    hasattr(st.session_state, 'model_results') and 
                    hasattr(st.session_state, 'label_encoders') and 
                    hasattr(st.session_state, 'model_cache_key') and 
                    st.session_state.model_cache_key == cache_key
                ):
                    st.info("Model already trained for current selection. Skipping retraining.")
                    model_results = st.session_state.model_results
                    label_encoders = st.session_state.label_encoders
                else:
                    try:
                        model_results, label_encoders = analyzer.train_models(
                            st.session_state.target_col, 
                            st.session_state.selected_sensitive
                        )
                        if model_results is not None and label_encoders is not None:
                            st.session_state.model_results = model_results
                            st.session_state.label_encoders = label_encoders
                            st.session_state.model_cache_key = cache_key
                        else:
                            st.error("❌ Model training failed. Please check your data and try again.")
                            return
                    except Exception as e:
                        st.error(f"❌ Error in model training: {str(e)}")
                        st.info("Please check your data quality and try again.")
                        return
                # Calculate fairness metrics
                if FAIRLEARN_AVAILABLE:
                    analyzer.fairness_metrics(
                        st.session_state.selected_sensitive,
                        model_results,
                        label_encoders
                    )
                else:
                    st.warning("⚠️ Fairlearn not available. Install with: pip install fairlearn")
            else:
                st.warning("Please complete bias analysis first.")
        
        elif "6. Explainability" in selected_step:
            analyzer.explainability_analysis()
        
        elif "7. Governance Report" in selected_step:
            if (hasattr(st.session_state, 'risk_df') and 
                hasattr(st.session_state, 'selected_sensitive') and
                hasattr(st.session_state, 'identified_sensitive')):
                analyzer.generate_governance_report(
                    st.session_state.risk_df,
                    st.session_state.selected_sensitive,
                    st.session_state.identified_sensitive
                )
            else:
                st.warning("Please complete previous analysis steps first.")
    
    else:
        st.info("👆 Please load your data first using the sidebar.")

if __name__ == "__main__":
    main()
