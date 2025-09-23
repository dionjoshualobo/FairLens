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
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not installed. Advanced explainability features will be unavailable.")

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
            'Data Type': self.data.dtypes,
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
        
        # Identify categorical features that could be sensitive
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        st.subheader("🎯 Select Sensitive Features for Bias Analysis")
        
        # Let user select sensitive features
        selected_sensitive = st.multiselect(
            "Select features to analyze for bias:",
            categorical_cols,
            default=[col for col in categorical_cols if any(keyword in col.lower() 
                    for keyword in ['gender', 'race', 'ethnicity', 'age', 'education'])]
        )
        
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
        
        target_col = st.selectbox("Select target variable:", self.data.columns.tolist())
        
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
        
        # Prepare data
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
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
    
    def explainability_analysis(self):
        """SHAP-based model explainability"""
        st.header("🔍 Model Explainability")
        
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
                
                # Choose appropriate SHAP explainer based on model type
                if hasattr(self.model, 'estimators_') and 'Forest' in model_name:
                    # Random Forest
                    st.write("Using TreeExplainer for Random Forest...")
                    explainer = shap.TreeExplainer(self.model)
                    
                    # Use smaller sample for efficiency and avoid dimension issues
                    sample_size = min(50, len(self.X_test))
                    X_sample = self.X_test.iloc[:sample_size] if hasattr(self.X_test, 'iloc') else self.X_test[:sample_size]
                    
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Handle multi-output case
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values_to_plot = shap_values[1]  # Use positive class
                        st.info("Using SHAP values for positive class")
                    else:
                        shap_values_to_plot = shap_values
                    
                elif 'Gradient' in model_name and is_binary:
                    # Gradient Boosting - only for binary
                    st.write("Using TreeExplainer for Gradient Boosting...")
                    explainer = shap.TreeExplainer(self.model)
                    
                    sample_size = min(50, len(self.X_test))
                    X_sample = self.X_test.iloc[:sample_size] if hasattr(self.X_test, 'iloc') else self.X_test[:sample_size]
                    
                    shap_values = explainer.shap_values(X_sample)
                    shap_values_to_plot = shap_values
                    
                elif hasattr(self.model, 'predict_proba'):
                    # Use KernelExplainer for other models
                    st.write("Using KernelExplainer (this may take a moment)...")
                    
                    # Ensure we have proper DataFrame/array format
                    X_background = self.X_train.iloc[:30] if hasattr(self.X_train, 'iloc') else self.X_train[:30]
                    X_sample = self.X_test.iloc[:30] if hasattr(self.X_test, 'iloc') else self.X_test[:30]
                    
                    explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
                    shap_values = explainer.shap_values(X_sample)
                    
                    if isinstance(shap_values, list) and len(shap_values) > 1:
                        shap_values_to_plot = shap_values[1]  # Positive class
                    else:
                        shap_values_to_plot = shap_values
                else:
                    # Linear explainer for linear models
                    st.write("Using LinearExplainer...")
                    
                    X_background = self.X_train.iloc[:50] if hasattr(self.X_train, 'iloc') else self.X_train[:50]
                    X_sample = self.X_test.iloc[:50] if hasattr(self.X_test, 'iloc') else self.X_test[:50]
                    
                    explainer = shap.LinearExplainer(self.model, X_background)
                    shap_values = explainer.shap_values(X_sample)
                    shap_values_to_plot = shap_values
                
                # Calculate feature importance from SHAP values
                if shap_values_to_plot is not None:
                    # Ensure shap_values is 2D array
                    if len(shap_values_to_plot.shape) == 1:
                        shap_values_to_plot = shap_values_to_plot.reshape(1, -1)
                    
                    feature_importance_shap = np.abs(shap_values_to_plot).mean(0)
                    
                    # Get feature names
                    if hasattr(self.X_test, 'columns'):
                        feature_names = self.X_test.columns.tolist()
                    else:
                        feature_names = [f'feature_{i}' for i in range(len(feature_importance_shap))]
                    
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
                    
                    # Plot top features
                    top_features = feature_importance_df.head(10)
                    fig = px.bar(top_features, x='SHAP_Importance', y='Feature', 
                                orientation='h', title="Top 10 Feature Importance (SHAP)")
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig)
                    
                    st.dataframe(feature_importance_df.head(15))
                    
                    explanation_success = True
                    st.success("✅ SHAP analysis completed successfully!")
                
            except Exception as e:
                st.warning(f"SHAP analysis failed: {str(e)}")
                st.info("Falling back to alternative explanation methods...")
        
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
    
    def generate_governance_report(self, risk_df, sensitive_features, identified_sensitive):
        """Generate comprehensive governance report"""
        st.header("📋 Governance Report")
        
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
            st.write("**Privacy Risk Summary**")
            risk_counts = risk_df['Risk Level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title="Privacy Risk Distribution")
            st.plotly_chart(fig)
        
        with col2:
            st.write("**Compliance Checklist**")
            checklist = [
                "✅ Data inventory completed",
                "✅ Sensitive features identified",
                "✅ Privacy risk assessment conducted",
                "✅ Bias analysis performed",
                "✅ Model explainability provided"
            ]
            for item in checklist:
                st.write(item)
        
        # Recommendations
        st.subheader("🎯 Key Recommendations")
        
        recommendations = [
            "1. **High Priority**: Address high-risk privacy features through anonymization or removal",
            "2. **Bias Mitigation**: Implement fairness constraints in model training",
            "3. **Monitoring**: Set up continuous monitoring for model bias and drift",
            "4. **Documentation**: Maintain comprehensive model documentation",
            "5. **Training**: Ensure team is trained on responsible AI practices"
        ]
        
        for rec in recommendations:
            st.write(rec)
        
        # Export functionality
        st.subheader("📤 Export Report")
        
        if st.button("Generate Detailed Report"):
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

### Sensitive Data Categories Identified:
{chr(10).join(f"- {category}" for category in report_data['privacy_assessment']['identified_sensitive_categories'])}

## Bias Analysis
### Analyzed Features:
{chr(10).join(f"- {feature}" for feature in sensitive_features)}

## Recommendations
{chr(10).join(recommendations)}
            """
            
            st.download_button(
                label="Download Report",
                data=report_text,
                file_name=f"governance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def main():
    st.set_page_config(
        page_title="AI Model Governance & Fairness Analyzer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ AI Model Governance & Fairness Analyzer")
    st.markdown("Comprehensive analysis for bias, privacy, and governance compliance")
    
    analyzer = ModelGovernanceAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("📋 Analysis Steps")
    steps = [
        "1. Load Data",
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
        
        # Default to loan_data.csv
        default_path = "Datasets/loan_data.csv"
        
        use_default = st.checkbox("Use default loan_data.csv", value=True)
        
        if use_default:
            if analyzer.load_data(default_path):
                st.session_state.data_loaded = True
                st.session_state.analyzer = analyzer
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                if analyzer.load_data(uploaded_file):
                    st.session_state.data_loaded = True
                    st.session_state.analyzer = analyzer
    
    # Check if data is loaded
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        analyzer = st.session_state.analyzer
        
        if "2. Data Overview" in selected_step:
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
                try:
                    model_results, label_encoders = analyzer.train_models(
                        st.session_state.target_col, 
                        st.session_state.selected_sensitive
                    )
                    
                    if model_results is not None and label_encoders is not None:
                        st.session_state.model_results = model_results
                        st.session_state.label_encoders = label_encoders
                        
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
                        st.error("❌ Model training failed. Please check your data and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error in model training: {str(e)}")
                    st.info("Please check your data quality and try again.")
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
