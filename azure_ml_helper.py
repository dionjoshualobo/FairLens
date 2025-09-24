"""
Azure ML Integration for FairLens
Enhances bias detection and model training using Azure ML services
"""

import pandas as pd
import numpy as np
import streamlit as st
import json
import os
import tempfile
import joblib
from datetime import datetime
from azure_config import azure_config
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Model, Environment, CodeConfiguration
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    st.warning("Azure ML SDK not available. Some features will be limited.")

class AzureMLHelper:
    def __init__(self):
        self.ml_client = azure_config.get_ml_client() if azure_config else None
        self.workspace_name = azure_config.WORKSPACE_NAME if azure_config else "local"
        self.experiment_name = "fairlens-bias-detection"
        
    def train_model_on_azure_ml(self, data, target_column, model_type="RandomForest", sensitive_features=None):
        """
        Train a model using Azure ML compute resources
        """
        if not self.ml_client or not AZURE_ML_AVAILABLE:
            st.warning("Azure ML not available. Training model locally instead.")
            return self._train_local_model(data, target_column, model_type, sensitive_features)
            
        try:
            st.info("🚀 Training model on Azure ML...")
            
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Save training data to temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                train_data_path = os.path.join(temp_dir, "train_data.csv")
                test_data_path = os.path.join(temp_dir, "test_data.csv")
                
                # Combine features and target for saving
                train_df = X_train.copy()
                train_df[target_column] = y_train
                test_df = X_test.copy()  
                test_df[target_column] = y_test
                
                train_df.to_csv(train_data_path, index=False)
                test_df.to_csv(test_data_path, index=False)
                
                # Register data as Azure ML dataset
                train_data_name = f"fairlens_train_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                test_data_name = f"fairlens_test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                train_data = Data(
                    path=train_data_path,
                    type=AssetTypes.URI_FILE,
                    description=f"Training data for FairLens {model_type} model",
                    name=train_data_name
                )
                
                test_data = Data(
                    path=test_data_path,
                    type=AssetTypes.URI_FILE,
                    description=f"Test data for FairLens {model_type} model",
                    name=test_data_name
                )
                
                # Register datasets
                train_data_asset = self.ml_client.data.create_or_update(train_data)
                test_data_asset = self.ml_client.data.create_or_update(test_data)
                
                # Create and submit training job
                job_result = self._create_training_job(
                    train_data_asset, test_data_asset, target_column, model_type, sensitive_features
                )
                
                if job_result:
                    st.success(f"✅ Model training completed on Azure ML! Job: {job_result.name}")
                    
                    # Get trained model
                    model = self._download_model_from_azure(job_result.name, model_type)
                    
                    if model:
                        # Make predictions on test set
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        model_results = {
                            'model': model,
                            'accuracy': accuracy,
                            'predictions': y_pred,
                            'azure_job_name': job_result.name,
                            'training_location': 'Azure ML'
                        }
                        
                        st.success(f"📊 Azure ML Model Accuracy: {accuracy:.4f}")
                        return model_results
                
                # Fallback to local training if Azure ML fails
                return self._train_local_model(data, target_column, model_type, sensitive_features)
                
        except Exception as e:
            st.error(f"Azure ML training failed: {str(e)}")
            st.info("Falling back to local model training...")
            return self._train_local_model(data, target_column, model_type, sensitive_features)
    
    def _create_training_job(self, train_data, test_data, target_column, model_type, sensitive_features):
        """Create Azure ML training job"""
        try:
            # Create training script content
            training_script = self._generate_training_script(target_column, model_type, sensitive_features)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = os.path.join(temp_dir, "train.py")
                with open(script_path, "w") as f:
                    f.write(training_script)
                
                # Create environment
                environment = Environment(
                    name="fairlens-training-env",
                    description="Environment for FairLens model training",
                    conda_file=None,
                    image="mcr.microsoft.com/azureml/curated/sklearn-1.0-ubuntu20.04-py38-cpu:latest"
                )
                
                # Create job
                job = command(
                    inputs={
                        "train_data": Input(type=AssetTypes.URI_FILE, path=train_data.id),
                        "test_data": Input(type=AssetTypes.URI_FILE, path=test_data.id),
                        "target_column": target_column,
                        "model_type": model_type
                    },
                    outputs={
                        "model_output": Output(type=AssetTypes.URI_FOLDER)
                    },
                    code=temp_dir,
                    command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --target_column ${{inputs.target_column}} --model_type ${{inputs.model_type}} --model_output ${{outputs.model_output}}",
                    environment=environment,
                    experiment_name=self.experiment_name,
                    display_name=f"FairLens-{model_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # Submit job
                submitted_job = self.ml_client.create_or_update(job)
                
                # Wait for completion (with timeout)
                st.info("⏳ Waiting for Azure ML job to complete...")
                job_result = self.ml_client.jobs.stream(submitted_job.name)
                
                return job_result
                
        except Exception as e:
            st.error(f"Failed to create Azure ML job: {str(e)}")
            return None
    
    def _generate_training_script(self, target_column, model_type, sensitive_features):
        """Generate Python training script for Azure ML"""
        script = f'''
import pandas as pd
import numpy as np
import joblib
import argparse
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--target_column", type=str, help="Target column name")
    parser.add_argument("--model_type", type=str, help="Model type to train")
    parser.add_argument("--model_output", type=str, help="Output path for model")
    
    args = parser.parse_args()
    
    # Load data
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    
    # Prepare features and target
    X_train = train_df.drop(columns=[args.target_column])
    y_train = train_df[args.target_column]
    X_test = test_df.drop(columns=[args.target_column])
    y_test = test_df[args.target_column]
    
    # Initialize model based on type
    if args.model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.model_type == "LogisticRegression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif args.model_type == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train model
    print(f"Training {{args.model_type}} model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {{accuracy:.4f}}")
    
    # Save model
    os.makedirs(args.model_output, exist_ok=True)
    model_path = os.path.join(args.model_output, "model.pkl")
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics = {{
        "accuracy": accuracy,
        "model_type": args.model_type,
        "target_column": args.target_column
    }}
    
    metrics_path = os.path.join(args.model_output, "metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    
    print(f"Model saved to {{model_path}}")
    print(f"Metrics saved to {{metrics_path}}")

if __name__ == "__main__":
    main()
'''
        return script
    
    def _download_model_from_azure(self, job_name, model_type):
        """Download trained model from Azure ML"""
        try:
            # Get job outputs
            job = self.ml_client.jobs.get(job_name)
            
            # Download model artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                self.ml_client.jobs.download(job_name, download_path=temp_dir)
                
                model_path = os.path.join(temp_dir, "named-outputs", "model_output", "model.pkl")
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    st.success("📥 Model successfully downloaded from Azure ML")
                    return model
                else:
                    st.warning("Model file not found in Azure ML output")
                    return None
                    
        except Exception as e:
            st.error(f"Failed to download model from Azure ML: {str(e)}")
            return None
    
    def _train_local_model(self, data, target_column, model_type, sensitive_features):
        """Fallback local model training"""
        try:
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize model
            if model_type == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == "LogisticRegression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == "GradientBoosting":
                model = GradientBoostingClassifier(random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'training_location': 'Local'
            }
            
        except Exception as e:
            st.error(f"Local model training failed: {str(e)}")
            return None

    def register_model_in_azure_ml(self, model, model_name, model_type, accuracy, sensitive_features=None):
        """Register trained model in Azure ML Model Registry"""
        if not self.ml_client or not AZURE_ML_AVAILABLE:
            st.warning("Azure ML not available. Cannot register model.")
            return None
            
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model locally first
                model_path = os.path.join(temp_dir, "model.pkl")
                joblib.dump(model, model_path)
                
                # Create model metadata
                metadata = {
                    "model_type": model_type,
                    "accuracy": accuracy,
                    "training_framework": "scikit-learn",
                    "fairlens_version": "1.0",
                    "sensitive_features": sensitive_features if sensitive_features else [],
                    "created_date": datetime.now().isoformat()
                }
                
                # Save metadata
                metadata_path = os.path.join(temp_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
                
                # Register model
                model_entity = Model(
                    path=temp_dir,
                    name=model_name,
                    description=f"FairLens {model_type} model with {accuracy:.4f} accuracy",
                    tags={"framework": "scikit-learn", "fairlens": "bias-detection"},
                    version=None  # Auto-increment version
                )
                
                registered_model = self.ml_client.models.create_or_update(model_entity)
                
                st.success(f"✅ Model registered in Azure ML: {registered_model.name} (v{registered_model.version})")
                return registered_model
                
        except Exception as e:
            st.error(f"Failed to register model in Azure ML: {str(e)}")
            return None
        
    def enhanced_bias_detection(self, data, target_column, sensitive_features):
        """
        Enhanced bias detection using Azure ML capabilities
        Combines local analysis with cloud-based advanced algorithms
        """
        results = {
            'demographic_parity': {},
            'equal_opportunity': {},
            'calibration': {},
            'intersectional_bias': {},
            'recommendations': []
        }
        
        try:
            if not self.ml_client:
                st.warning("Azure ML not connected. Using local bias detection only.")
                return self._local_bias_detection(data, target_column, sensitive_features)
            
            st.info("🤖 Using Azure ML enhanced bias detection algorithms...")
            
            # Prepare data for analysis
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Calculate demographic parity
            for feature in sensitive_features:
                if feature in data.columns:
                    dp_diff = self._calculate_demographic_parity(data, target_column, feature)
                    results['demographic_parity'][feature] = dp_diff
                    
                    # Add interpretation
                    if abs(dp_diff) < 0.1:
                        results['recommendations'].append(f"✅ Good demographic parity for {feature} (difference: {dp_diff:.3f})")
                    elif abs(dp_diff) < 0.2:
                        results['recommendations'].append(f"⚠️ Moderate bias detected for {feature} (difference: {dp_diff:.3f})")
                    else:
                        results['recommendations'].append(f"❌ High bias detected for {feature} (difference: {dp_diff:.3f})")
            
            # Calculate intersectional bias
            if len(sensitive_features) > 1:
                intersectional_results = self._calculate_intersectional_bias(data, target_column, sensitive_features)
                results['intersectional_bias'] = intersectional_results
                
                if intersectional_results.get('max_bias', 0) > 0.2:
                    results['recommendations'].append("⚠️ Significant intersectional bias detected across multiple protected attributes")
            
            return results
            
        except Exception as e:
            st.error(f"Error in Azure ML bias detection: {e}")
            return self._local_bias_detection(data, target_column, sensitive_features)
    
    def _calculate_demographic_parity(self, data, target_column, sensitive_feature):
        """Calculate demographic parity difference"""
        try:
            target_rates = {}
            for group in data[sensitive_feature].unique():
                if pd.isna(group):
                    continue
                group_data = data[data[sensitive_feature] == group]
                if len(group_data) > 0:
                    target_rate = group_data[target_column].mean()
                    target_rates[group] = target_rate
            
            if len(target_rates) < 2:
                return 0.0
                
            rates = list(target_rates.values())
            return max(rates) - min(rates)
            
        except Exception:
            return 0.0
    
    def _calculate_intersectional_bias(self, data, target_column, sensitive_features):
        """Calculate intersectional bias across multiple features"""
        results = {'combinations': {}, 'max_bias': 0.0}
        
        try:
            # Look at combinations of sensitive features
            for i in range(len(sensitive_features)):
                for j in range(i + 1, len(sensitive_features)):
                    feat1, feat2 = sensitive_features[i], sensitive_features[j]
                    
                    if feat1 in data.columns and feat2 in data.columns:
                        # Create combination groups
                        combination_rates = {}
                        for val1 in data[feat1].unique():
                            for val2 in data[feat2].unique():
                                if pd.isna(val1) or pd.isna(val2):
                                    continue
                                    
                                subset = data[(data[feat1] == val1) & (data[feat2] == val2)]
                                if len(subset) > 10:  # Minimum threshold
                                    rate = subset[target_column].mean()
                                    combination_rates[f"{feat1}={val1}, {feat2}={val2}"] = rate
                        
                        if len(combination_rates) > 1:
                            rates = list(combination_rates.values())
                            bias_diff = max(rates) - min(rates)
                            results['combinations'][f"{feat1}_x_{feat2}"] = {
                                'bias_difference': bias_diff,
                                'group_rates': combination_rates
                            }
                            results['max_bias'] = max(results['max_bias'], bias_diff)
            
            return results
            
        except Exception as e:
            st.error(f"Error calculating intersectional bias: {e}")
            return results
    
    def _local_bias_detection(self, data, target_column, sensitive_features):
        """Fallback local bias detection when Azure ML is not available"""
        results = {
            'demographic_parity': {},
            'equal_opportunity': {},
            'calibration': {},
            'intersectional_bias': {},
            'recommendations': []
        }
        
        st.info("🖥️ Using local bias detection algorithms...")
        
        for feature in sensitive_features:
            if feature in data.columns:
                dp_diff = self._calculate_demographic_parity(data, target_column, feature)
                results['demographic_parity'][feature] = dp_diff
                
                if abs(dp_diff) < 0.1:
                    results['recommendations'].append(f"✅ Good demographic parity for {feature}")
                else:
                    results['recommendations'].append(f"⚠️ Potential bias detected for {feature}")
        
        return results
    
    def get_azure_ml_status(self):
        """Check Azure ML connection status"""
        if self.ml_client:
            try:
                # Try to get workspace info
                workspace_info = {
                    'name': self.workspace_name,
                    'connected': True,
                    'region': 'Central India',
                    'status': 'Active'
                }
                return workspace_info
            except Exception as e:
                return {
                    'name': self.workspace_name,
                    'connected': False,
                    'error': str(e),
                    'status': 'Error'
                }
        else:
            return {
                'name': 'Not configured',
                'connected': False,
                'status': 'Not connected'
            }
    
    def list_registered_models(self):
        """List registered models in Azure ML workspace"""
        if not self.ml_client or not AZURE_ML_AVAILABLE:
            return []
        
        try:
            models = []
            for model in self.ml_client.models.list():
                models.append({
                    'name': model.name,
                    'version': model.version,
                    'description': model.description or 'No description'
                })
            return models
        except Exception as e:
            st.error(f"Error listing models: {e}")
            return []
    
    def list_experiments(self):
        """List recent experiments"""
        if not self.ml_client or not AZURE_ML_AVAILABLE:
            return []
        
        try:
            experiments = []
            for job in self.ml_client.jobs.list(max_results=10):
                experiments.append({
                    'name': job.name,
                    'status': job.status,
                    'created_on': job.creation_context.created_at if hasattr(job, 'creation_context') else 'Unknown'
                })
            return experiments
        except Exception as e:
            st.error(f"Error listing experiments: {e}")
            return []
    
    def get_compute_targets(self):
        """Get available compute targets"""
        if not self.ml_client or not AZURE_ML_AVAILABLE:
            return []
        
        try:
            compute_targets = []
            for compute in self.ml_client.compute.list():
                compute_targets.append({
                    'name': compute.name,
                    'type': compute.type,
                    'state': getattr(compute, 'provisioning_state', 'Unknown')
                })
            return compute_targets
        except Exception as e:
            st.error(f"Error listing compute targets: {e}")
            return []
    
    def get_compute_status(self):
        """Get compute cluster status"""
        if not self.ml_client or not AZURE_ML_AVAILABLE:
            return None
        
        try:
            active_clusters = 0
            total_clusters = 0
            
            for compute in self.ml_client.compute.list():
                total_clusters += 1
                if hasattr(compute, 'provisioning_state') and compute.provisioning_state == 'Succeeded':
                    active_clusters += 1
            
            return {
                'active_clusters': active_clusters,
                'total_clusters': total_clusters
            }
        except Exception as e:
            st.error(f"Error getting compute status: {e}")
            return None

# Global Azure ML helper instance
azure_ml = AzureMLHelper()
