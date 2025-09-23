"""
Advanced Privacy Analysis and Compliance Module
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import hashlib

class PrivacyAnalyzer:
    def __init__(self):
        self.privacy_patterns = {
            # Personal Identifiers
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone': [r'\b(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'],
            'ssn': [r'\b\d{3}-?\d{2}-?\d{4}\b'],
            'credit_card': [r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'],
            'ip_address': [r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'],
            'license_plate': [r'\b[A-Z0-9]{2,3}[-\s]?[A-Z0-9]{3,4}\b']
        }
        
        self.sensitive_keywords = {
            'demographic': ['age', 'gender', 'sex', 'race', 'ethnicity', 'religion', 'nationality'],
            'financial': ['income', 'salary', 'wage', 'credit', 'debt', 'loan', 'mortgage'],
            'health': ['health', 'medical', 'disease', 'condition', 'diagnosis', 'treatment'],
            'location': ['address', 'zip', 'postal', 'latitude', 'longitude', 'gps'],
            'behavioral': ['purchase', 'browse', 'click', 'view', 'search', 'preference'],
            'biometric': ['fingerprint', 'facial', 'iris', 'voice', 'dna', 'biometric']
        }
        
        self.compliance_frameworks = {
            'GDPR': {
                'requirements': [
                    'Data minimization',
                    'Purpose limitation', 
                    'Storage limitation',
                    'Accuracy',
                    'Integrity and confidentiality',
                    'Accountability'
                ],
                'rights': [
                    'Right to access',
                    'Right to rectification',
                    'Right to erasure',
                    'Right to restrict processing',
                    'Right to data portability',
                    'Right to object'
                ]
            },
            'CCPA': {
                'requirements': [
                    'Notice at collection',
                    'Notice of sale',
                    'Right to know',
                    'Right to delete',
                    'Right to opt-out',
                    'Non-discrimination'
                ]
            },
            'HIPAA': {
                'requirements': [
                    'Administrative safeguards',
                    'Physical safeguards', 
                    'Technical safeguards',
                    'Minimum necessary standard'
                ]
            }
        }
    
    def scan_for_patterns(self, data):
        """Scan dataset for privacy-sensitive patterns"""
        findings = {}
        
        for category, patterns in self.privacy_patterns.items():
            findings[category] = []
            
            for column in data.columns:
                if data[column].dtype == 'object':  # Only scan text columns
                    for pattern in patterns:
                        matches = data[column].astype(str).str.contains(pattern, regex=True, na=False)
                        if matches.any():
                            findings[category].append({
                                'column': column,
                                'matches': matches.sum(),
                                'percentage': (matches.sum() / len(data)) * 100
                            })
        
        return findings
    
    def identify_sensitive_features(self, data):
        """Identify potentially sensitive features based on column names and content"""
        sensitive_features = {}
        
        for category, keywords in self.sensitive_keywords.items():
            sensitive_features[category] = []
            
            for column in data.columns:
                # Check column name
                if any(keyword.lower() in column.lower() for keyword in keywords):
                    sensitivity_score = self.calculate_sensitivity_score(data[column])
                    sensitive_features[category].append({
                        'column': column,
                        'sensitivity_score': sensitivity_score,
                        'reason': 'Column name match'
                    })
                
                # Check content patterns for text columns
                elif data[column].dtype == 'object':
                    content_score = self.analyze_content_sensitivity(data[column], keywords)
                    if content_score > 0.1:
                        sensitive_features[category].append({
                            'column': column,
                            'sensitivity_score': content_score,
                            'reason': 'Content pattern match'
                        })
        
        return sensitive_features
    
    def calculate_sensitivity_score(self, series):
        """Calculate sensitivity score for a feature"""
        score = 0
        
        # Uniqueness (high uniqueness = higher risk)
        uniqueness = series.nunique() / len(series)
        score += min(uniqueness * 3, 3)
        
        # Data type considerations
        if series.dtype == 'object':
            score += 1
        
        # Missing values (might indicate sensitive data removal)
        missing_rate = series.isnull().sum() / len(series)
        if missing_rate > 0.1:
            score += 1
        
        return min(score, 5)  # Cap at 5
    
    def analyze_content_sensitivity(self, series, keywords):
        """Analyze content of text series for sensitivity"""
        if series.dtype != 'object':
            return 0
        
        text_content = ' '.join(series.astype(str).values).lower()
        
        matches = sum(1 for keyword in keywords if keyword.lower() in text_content)
        return matches / len(keywords)
    
    def assess_reidentification_risk(self, data):
        """Assess risk of re-identification using quasi-identifiers"""
        risk_assessment = {}
        
        # Identify potential quasi-identifiers
        quasi_identifiers = []
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64', 'object']:
                uniqueness = data[column].nunique() / len(data)
                if 0.01 < uniqueness < 0.9:  # Not too unique, not too common
                    quasi_identifiers.append(column)
        
        # Calculate k-anonymity for combinations of quasi-identifiers
        if len(quasi_identifiers) >= 2:
            for i in range(2, min(len(quasi_identifiers) + 1, 5)):  # Check combinations up to 4
                from itertools import combinations
                for combo in combinations(quasi_identifiers, i):
                    group_sizes = data.groupby(list(combo)).size()
                    k_anonymity = group_sizes.min()
                    
                    risk_level = 'Low'
                    if k_anonymity == 1:
                        risk_level = 'Critical'
                    elif k_anonymity < 3:
                        risk_level = 'High'
                    elif k_anonymity < 5:
                        risk_level = 'Medium'
                    
                    risk_assessment[f"Combo_{'+'.join(combo)}"] = {
                        'k_anonymity': k_anonymity,
                        'risk_level': risk_level,
                        'unique_combinations': len(group_sizes)
                    }
        
        return risk_assessment
    
    def generate_compliance_report(self, data, framework='GDPR'):
        """Generate compliance assessment report for specified framework"""
        report = {
            'framework': framework,
            'assessment_date': datetime.now().isoformat(),
            'compliance_score': 0,
            'requirements_assessment': {},
            'recommendations': []
        }
        
        if framework in self.compliance_frameworks:
            requirements = self.compliance_frameworks[framework]['requirements']
            
            for req in requirements:
                # Assess each requirement
                assessment = self.assess_requirement(data, req, framework)
                report['requirements_assessment'][req] = assessment
                report['compliance_score'] += assessment['score']
            
            # Normalize score
            report['compliance_score'] = report['compliance_score'] / len(requirements)
            
            # Generate recommendations
            report['recommendations'] = self.generate_compliance_recommendations(
                report['requirements_assessment'], framework
            )
        
        return report
    
    def assess_requirement(self, data, requirement, framework):
        """Assess compliance with specific requirement"""
        assessment = {'score': 0, 'status': 'Non-compliant', 'details': []}
        
        if framework == 'GDPR':
            if requirement == 'Data minimization':
                # Check if dataset seems minimal
                correlation_matrix = data.select_dtypes(include=[np.number]).corr().abs()
                highly_correlated = (correlation_matrix > 0.9).sum().sum() - len(correlation_matrix)
                
                if highly_correlated < len(correlation_matrix) * 0.1:
                    assessment['score'] = 1
                    assessment['status'] = 'Compliant'
                    assessment['details'].append('Low correlation between features suggests data minimization')
                else:
                    assessment['details'].append('High correlation suggests potential redundant features')
            
            elif requirement == 'Accuracy':
                # Check for data quality indicators
                missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
                
                if missing_rate < 0.05:
                    assessment['score'] = 1
                    assessment['status'] = 'Compliant'
                    assessment['details'].append('Low missing data rate indicates good accuracy')
                else:
                    assessment['details'].append(f'High missing data rate: {missing_rate:.2%}')
        
        return assessment
    
    def generate_compliance_recommendations(self, requirements_assessment, framework):
        """Generate specific recommendations based on assessment"""
        recommendations = []
        
        for req, assessment in requirements_assessment.items():
            if assessment['score'] < 1:
                if framework == 'GDPR':
                    if req == 'Data minimization':
                        recommendations.append(
                            "Consider removing highly correlated or redundant features"
                        )
                    elif req == 'Accuracy':
                        recommendations.append(
                            "Implement data validation and cleaning procedures"
                        )
                    elif req == 'Storage limitation':
                        recommendations.append(
                            "Implement data retention policies and automated deletion"
                        )
        
        # General recommendations
        recommendations.extend([
            "Conduct regular privacy impact assessments",
            "Implement privacy by design principles",
            "Ensure data subject rights can be exercised",
            "Maintain comprehensive data processing records"
        ])
        
        return recommendations
    
    def anonymize_data(self, data, method='k_anonymity', k=3):
        """Apply anonymization techniques to the dataset"""
        anonymized_data = data.copy()
        
        if method == 'k_anonymity':
            # Simple k-anonymity implementation
            # This is a basic implementation - production use should employ specialized libraries
            
            # Identify quasi-identifiers
            quasi_identifiers = []
            for column in data.columns:
                if data[column].dtype in ['int64', 'float64']:
                    uniqueness = data[column].nunique() / len(data)
                    if 0.1 < uniqueness < 0.9:
                        quasi_identifiers.append(column)
            
            # Apply generalization to numerical quasi-identifiers
            for column in quasi_identifiers:
                if data[column].dtype in ['int64', 'float64']:
                    # Bin numerical values
                    anonymized_data[column] = pd.cut(
                        data[column], 
                        bins=max(3, data[column].nunique() // k),
                        labels=False
                    )
        
        elif method == 'differential_privacy':
            # Add noise to numerical columns
            for column in data.select_dtypes(include=[np.number]).columns:
                noise = np.random.laplace(0, 1, len(data))
                anonymized_data[column] = data[column] + noise
        
        return anonymized_data
    
    def export_privacy_report(self, findings, risk_assessment, compliance_report):
        """Export comprehensive privacy analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_content = f"""
# Privacy Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides a comprehensive analysis of privacy risks and compliance status.

## Pattern Detection Results
"""
        
        for category, results in findings.items():
            if results:
                report_content += f"\n### {category.title()} Patterns\n"
                for result in results:
                    report_content += f"- {result['column']}: {result['matches']} matches ({result['percentage']:.1f}%)\n"
        
        report_content += "\n## Re-identification Risk Assessment\n"
        for combo, risk in risk_assessment.items():
            report_content += f"- {combo}: K-anonymity = {risk['k_anonymity']} ({risk['risk_level']} risk)\n"
        
        report_content += f"\n## {compliance_report['framework']} Compliance Assessment\n"
        report_content += f"Overall Score: {compliance_report['compliance_score']:.2f}/1.0\n\n"
        
        for req, assessment in compliance_report['requirements_assessment'].items():
            report_content += f"- {req}: {assessment['status']}\n"
        
        report_content += "\n## Recommendations\n"
        for rec in compliance_report['recommendations']:
            report_content += f"- {rec}\n"
        
        return report_content
