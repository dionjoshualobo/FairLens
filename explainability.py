"""
Advanced Model Explainability and Interpretability Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names or list(range(X_train.shape[1]))
        self.explanations = {}
    
    def global_feature_importance(self):
        """Calculate global feature importance using multiple methods"""
        importance_results = {}
        
        # 1. Model's built-in feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_results['model_importance'] = {
                'importance': self.model.feature_importances_,
                'method': 'Model Built-in'
            }
        
        # 2. Permutation importance
        try:
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42
            )
            importance_results['permutation_importance'] = {
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std,
                'method': 'Permutation'
            }
        except Exception as e:
            print(f"Error calculating permutation importance: {e}")
        
        # 3. Coefficient-based importance (for linear models)
        if hasattr(self.model, 'coef_'):
            importance_results['coefficient_importance'] = {
                'importance': np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_),
                'method': 'Coefficient Magnitude'
            }
        
        return importance_results
    
    def shap_analysis(self):
        """Perform SHAP analysis for model explanations"""
        shap_results = {}
        
        try:
            import shap
            
            # Initialize explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                # Tree-based models
                if hasattr(self.model, 'estimators_'):
                    explainer = shap.TreeExplainer(self.model)
                else:
                    # Use KernelExplainer for other models
                    explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.X_train[:100]  # Use sample for efficiency
                    )
            else:
                explainer = shap.KernelExplainer(
                    self.model.predict, 
                    self.X_train[:100]
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(self.X_test[:100])
            
            # Store results
            shap_results['explainer'] = explainer
            shap_results['shap_values'] = shap_values
            shap_results['feature_importance'] = np.abs(shap_values).mean(0) if len(shap_values.shape) == 2 else np.abs(shap_values[1]).mean(0)
            
            # Calculate feature interactions
            if len(self.X_test) > 50:
                try:
                    interaction_values = explainer.shap_interaction_values(self.X_test[:50])
                    shap_results['interactions'] = interaction_values
                except:
                    pass
            
        except ImportError:
            print("SHAP not available. Install with: pip install shap")
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")
        
        return shap_results
    
    def lime_analysis(self, instance_idx=0):
        """Perform LIME analysis for local explanations"""
        lime_results = {}
        
        try:
            import lime
            import lime.lime_tabular
            
            # Initialize LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=['Class 0', 'Class 1'],
                mode='classification'
            )
            
            # Explain instance
            exp = explainer.explain_instance(
                self.X_test[instance_idx], 
                self.model.predict_proba,
                num_features=len(self.feature_names)
            )
            
            lime_results['explanation'] = exp
            lime_results['feature_weights'] = exp.as_list()
            
        except ImportError:
            print("LIME not available. Install with: pip install lime")
        except Exception as e:
            print(f"Error in LIME analysis: {e}")
        
        return lime_results
    
    def analyze_feature_interactions(self):
        """Analyze feature interactions and correlations"""
        interaction_analysis = {}
        
        # Calculate correlation matrix
        if isinstance(self.X_train, pd.DataFrame):
            correlation_matrix = self.X_train.corr()
        else:
            correlation_matrix = pd.DataFrame(self.X_train).corr()
        
        # Find highly correlated feature pairs
        high_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_correlation_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        interaction_analysis['correlation_matrix'] = correlation_matrix
        interaction_analysis['high_correlation_pairs'] = high_correlation_pairs
        
        # Feature interaction strength using mutual information
        try:
            from sklearn.feature_selection import mutual_info_classif
            
            mi_scores = mutual_info_classif(self.X_train, self.y_train)
            interaction_analysis['mutual_information'] = {
                'scores': mi_scores,
                'feature_names': self.feature_names
            }
        except Exception as e:
            print(f"Error calculating mutual information: {e}")
        
        return interaction_analysis
    
    def model_complexity_analysis(self):
        """Analyze model complexity and interpretability"""
        complexity_metrics = {}
        
        # Number of parameters/features
        if hasattr(self.model, 'coef_'):
            complexity_metrics['n_parameters'] = len(self.model.coef_.flatten())
        elif hasattr(self.model, 'feature_importances_'):
            complexity_metrics['n_features'] = len(self.model.feature_importances_)
        
        # Model-specific complexity metrics
        if hasattr(self.model, 'n_estimators'):
            complexity_metrics['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            complexity_metrics['max_depth'] = self.model.max_depth
        
        # Decision path complexity for tree-based models
        if hasattr(self.model, 'decision_path'):
            try:
                decision_paths = self.model.decision_path(self.X_test[:100])
                avg_path_length = decision_paths.sum(axis=1).mean()
                complexity_metrics['avg_decision_path_length'] = avg_path_length
            except:
                pass
        
        # Model size estimation
        try:
            import pickle
            model_size = len(pickle.dumps(self.model))
            complexity_metrics['model_size_bytes'] = model_size
        except:
            pass
        
        return complexity_metrics
    
    def generate_explanation_report(self):
        """Generate comprehensive explanation report"""
        report = {
            'model_type': type(self.model).__name__,
            'dataset_info': {
                'n_features': self.X_train.shape[1],
                'n_train_samples': self.X_train.shape[0],
                'n_test_samples': self.X_test.shape[0]
            }
        }
        
        # Global feature importance
        report['global_importance'] = self.global_feature_importance()
        
        # Model complexity
        report['complexity'] = self.model_complexity_analysis()
        
        # Feature interactions
        report['interactions'] = self.analyze_feature_interactions()
        
        # SHAP analysis
        shap_results = self.shap_analysis()
        if shap_results:
            report['shap_analysis'] = shap_results
        
        # Interpretability score
        report['interpretability_score'] = self.calculate_interpretability_score(report)
        
        return report
    
    def calculate_interpretability_score(self, report):
        """Calculate overall interpretability score (0-1)"""
        score = 0
        max_score = 0
        
        # Model type factor
        max_score += 3
        if 'Linear' in report['model_type']:
            score += 3
        elif 'Tree' in report['model_type'] or 'Forest' in report['model_type']:
            score += 2
        elif 'Gradient' in report['model_type']:
            score += 1
        
        # Complexity factor
        max_score += 2
        complexity = report['complexity']
        if 'n_parameters' in complexity:
            if complexity['n_parameters'] < 10:
                score += 2
            elif complexity['n_parameters'] < 50:
                score += 1
        elif 'n_features' in complexity:
            if complexity['n_features'] < 10:
                score += 2
            elif complexity['n_features'] < 50:
                score += 1
        
        # Feature interaction complexity
        max_score += 1
        if 'high_correlation_pairs' in report['interactions']:
            if len(report['interactions']['high_correlation_pairs']) < 3:
                score += 1
        
        return min(score / max_score, 1.0) if max_score > 0 else 0.5
    
    def create_explanation_visualizations(self, save_path=None):
        """Create visualization plots for model explanations"""
        visualizations = {}
        
        # Feature importance plot
        importance_data = self.global_feature_importance()
        if importance_data:
            fig, axes = plt.subplots(1, len(importance_data), figsize=(15, 5))
            if len(importance_data) == 1:
                axes = [axes]
            
            for idx, (method, data) in enumerate(importance_data.items()):
                if idx < len(axes):
                    importance_df = pd.DataFrame({
                        'Feature': self.feature_names[:len(data['importance'])],
                        'Importance': data['importance']
                    }).sort_values('Importance', ascending=True)
                    
                    axes[idx].barh(importance_df['Feature'], importance_df['Importance'])
                    axes[idx].set_title(f'Feature Importance - {data["method"]}')
                    axes[idx].set_xlabel('Importance')
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_feature_importance.png", dpi=300, bbox_inches='tight')
            visualizations['feature_importance'] = fig
        
        # Correlation heatmap
        interaction_data = self.analyze_feature_interactions()
        if 'correlation_matrix' in interaction_data:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                interaction_data['correlation_matrix'], 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                ax=ax
            )
            ax.set_title('Feature Correlation Matrix')
            
            if save_path:
                plt.savefig(f"{save_path}_correlation_matrix.png", dpi=300, bbox_inches='tight')
            visualizations['correlation_matrix'] = fig
        
        return visualizations
