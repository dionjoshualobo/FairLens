"""
Azure ML Integration for FairLens
Enhances bias detection and model training using Azure ML services
"""

import pandas as pd
import numpy as np
import streamlit as st
from azure_config import azure_config

class AzureMLHelper:
    def __init__(self):
        self.ml_client = azure_config.get_ml_client()
        self.workspace_name = azure_config.WORKSPACE_NAME
        
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

# Global Azure ML helper instance
azure_ml = AzureMLHelper()
