"""
Advanced Bias Detection and Mitigation Module
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedBiasDetector:
    def __init__(self):
        self.bias_metrics = {}
        self.mitigation_strategies = {}
    
    def calculate_bias_metrics(self, y_true, y_pred, sensitive_features):
        """Calculate comprehensive bias metrics"""
        metrics = {}
        
        # Convert target to binary if needed for consistency
        unique_targets = sorted(np.unique(y_true))
        if len(unique_targets) == 2 and not (set(unique_targets) == {0, 1}):
            # Convert to binary format
            pos_label = unique_targets[1]
            y_true_binary = (y_true == pos_label).astype(int)
            y_pred_binary = (y_pred == pos_label).astype(int)
        else:
            y_true_binary = y_true
            y_pred_binary = y_pred
        
        for feature_name, sensitive_values in sensitive_features.items():
            feature_metrics = {}
            
            # Group-wise metrics
            for group in np.unique(sensitive_values):
                group_mask = sensitive_values == group
                group_y_true = y_true_binary[group_mask]
                group_y_pred = y_pred_binary[group_mask]
                
                if len(group_y_true) > 0:
                    # Handle binary classification metrics
                    if len(np.unique(group_y_true)) == 2:
                        try:
                            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
                        except ValueError:
                            # If confusion matrix fails, calculate manually
                            tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                            tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
                            fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                            fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                        
                        # True Positive Rate (Sensitivity)
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        
                        # False Positive Rate
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        
                        # Precision
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    else:
                        # Single class in group
                        tpr = 0
                        fpr = 0
                        precision = 0
                    
                    # Selection Rate (Positive Rate)
                    selection_rate = np.mean(group_y_pred)
                    
                    feature_metrics[f'{feature_name}_{group}'] = {
                        'tpr': tpr,
                        'fpr': fpr,
                        'precision': precision,
                        'selection_rate': selection_rate,
                        'count': len(group_y_true)
                    }
            
            # Calculate fairness metrics
            groups = list(feature_metrics.keys())
            if len(groups) >= 2:
                # Demographic Parity (Statistical Parity)
                selection_rates = [feature_metrics[g]['selection_rate'] for g in groups]
                demographic_parity = max(selection_rates) - min(selection_rates)
                
                # Equalized Odds (TPR difference)
                tprs = [feature_metrics[g]['tpr'] for g in groups]
                equalized_odds = max(tprs) - min(tprs)
                
                # Equal Opportunity (TPR ratio)
                min_tpr = min(tprs)
                equal_opportunity = min_tpr / max(tprs) if max(tprs) > 0 else 1
                
                metrics[feature_name] = {
                    'demographic_parity': demographic_parity,
                    'equalized_odds': equalized_odds,
                    'equal_opportunity': equal_opportunity,
                    'group_metrics': feature_metrics
                }
        
        return metrics
    
    def generate_bias_report(self, metrics):
        """Generate detailed bias analysis report"""
        report = {
            'summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        for feature, feature_metrics in metrics.items():
            # Determine bias severity
            dp = feature_metrics['demographic_parity']
            eo = feature_metrics['equalized_odds']
            
            severity = 'Low'
            if dp > 0.2 or eo > 0.2:
                severity = 'High'
            elif dp > 0.1 or eo > 0.1:
                severity = 'Medium'
            
            report['summary'][feature] = {
                'bias_severity': severity,
                'demographic_parity': dp,
                'equalized_odds': eo
            }
            
            # Recommendations based on bias type
            if dp > 0.1:
                report['recommendations'].append(
                    f"🔴 {feature}: High demographic parity difference ({dp:.3f}). "
                    f"Consider pre-processing interventions or demographic parity constraints."
                )
            
            if eo > 0.1:
                report['recommendations'].append(
                    f"⚠️ {feature}: High equalized odds difference ({eo:.3f}). "
                    f"Consider post-processing threshold optimization."
                )
        
        return report
    
    def suggest_mitigation_strategies(self, bias_report):
        """Suggest specific mitigation strategies based on detected bias"""
        strategies = {
            'pre_processing': [],
            'in_processing': [],
            'post_processing': []
        }
        
        for feature, summary in bias_report['summary'].items():
            severity = summary['bias_severity']
            dp = summary['demographic_parity']
            eo = summary['equalized_odds']
            
            if severity in ['Medium', 'High']:
                # Pre-processing strategies
                strategies['pre_processing'].extend([
                    f"Resampling techniques for {feature}",
                    f"Synthetic data generation for underrepresented groups in {feature}",
                    f"Feature selection to reduce correlation with {feature}"
                ])
                
                # In-processing strategies
                if dp > 0.1:
                    strategies['in_processing'].append(
                        f"Demographic parity constraint during training for {feature}"
                    )
                
                if eo > 0.1:
                    strategies['in_processing'].append(
                        f"Equalized odds constraint during training for {feature}"
                    )
                
                # Post-processing strategies
                strategies['post_processing'].extend([
                    f"Threshold optimization for {feature}",
                    f"Calibration techniques for {feature} groups"
                ])
        
        return strategies

class FairnessMitigator:
    def __init__(self):
        self.strategies = {}
    
    def apply_resampling(self, X, y, sensitive_feature, strategy='smote'):
        """Apply resampling techniques to improve fairness"""
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.combine import SMOTETomek
            
            if strategy == 'smote':
                sampler = SMOTE(random_state=42)
            elif strategy == 'smote_tomek':
                sampler = SMOTETomek(random_state=42)
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return X_resampled, y_resampled
            
        except ImportError:
            print("imbalanced-learn package required for resampling. Returning original data.")
            return X, y
    
    def apply_fairness_constraints(self, model, X, y, sensitive_features):
        """Apply fairness constraints during training"""
        try:
            from fairlearn.reductions import ExponentiatedGradient, DemographicParity
            
            # Ensure binary target format
            unique_targets = sorted(np.unique(y))
            if len(unique_targets) == 2 and not (set(unique_targets) == {0, 1}):
                pos_label = unique_targets[1]
                y_binary = (y == pos_label).astype(int)
            else:
                y_binary = y
            
            # Apply demographic parity constraint
            constraint = DemographicParity()
            mitigator = ExponentiatedGradient(model, constraint)
            mitigator.fit(X, y_binary, sensitive_features=sensitive_features)
            
            return mitigator
            
        except ImportError:
            print("Fairlearn package required for fairness constraints. Returning original model.")
            return model
        except Exception as e:
            print(f"Error applying fairness constraints: {str(e)}. Returning original model.")
            return model
    
    def optimize_thresholds(self, model, X, y, sensitive_features):
        """Post-processing threshold optimization"""
        try:
            from fairlearn.postprocessing import ThresholdOptimizer
            
            # Ensure binary target format
            unique_targets = sorted(np.unique(y))
            if len(unique_targets) == 2 and not (set(unique_targets) == {0, 1}):
                pos_label = unique_targets[1]
                y_binary = (y == pos_label).astype(int)
            else:
                y_binary = y
            
            # Check if model has predict_proba method
            if not hasattr(model, 'predict_proba'):
                print("Model does not support probability predictions. Cannot optimize thresholds.")
                return model
            
            # Optimize thresholds
            threshold_optimizer = ThresholdOptimizer(
                estimator=model,
                constraints="demographic_parity",
                prefit=True
            )
            
            threshold_optimizer.fit(X, y_binary, sensitive_features=sensitive_features)
            
            return threshold_optimizer
            
        except ImportError:
            print("Fairlearn package required for threshold optimization. Returning original model.")
            return model
        except Exception as e:
            print(f"Error optimizing thresholds: {str(e)}. Returning original model.")
            return model

def calculate_intersectional_bias(y_true, y_pred, sensitive_features_dict):
    """Calculate bias metrics for intersectional groups"""
    intersectional_metrics = {}
    
    # Create intersectional groups
    feature_names = list(sensitive_features_dict.keys())
    if len(feature_names) >= 2:
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                feature1, feature2 = feature_names[i], feature_names[j]
                
                # Create combined feature
                combined_feature = [
                    f"{val1}_{val2}" for val1, val2 in 
                    zip(sensitive_features_dict[feature1], sensitive_features_dict[feature2])
                ]
                
                # Calculate metrics for intersectional groups
                for group in np.unique(combined_feature):
                    group_mask = np.array(combined_feature) == group
                    if np.sum(group_mask) > 10:  # Only consider groups with sufficient samples
                        group_y_true = y_true[group_mask]
                        group_y_pred = y_pred[group_mask]
                        
                        if len(np.unique(group_y_true)) > 1:  # Ensure both classes present
                            accuracy = np.mean(group_y_true == group_y_pred)
                            precision = np.mean(group_y_pred[group_y_true == 1]) if np.sum(group_y_true == 1) > 0 else 0
                            
                            intersectional_metrics[f"{feature1}_{feature2}_{group}"] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'count': len(group_y_true)
                            }
    
    return intersectional_metrics
