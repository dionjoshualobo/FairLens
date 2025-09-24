# AI Model Governance & Fairness Analysis Tool

## 🌐 Live Demo
Access the live applications:
- **Stable Version (v20)**: [https://fairlens-app-dion-v20.centralindia.azurecontainer.io:8501](https://fairlens-app-dion-v20.centralindia.azurecontainer.io:8501)
- **Enhanced Version (v22)**: [http://fairlens-v22-test.centralindia.azurecontainer.io:8501](http://fairlens-v22-test.centralindia.azurecontainer.io:8501)

*The enhanced version includes improved Overview Reports with specific feature identification and enhanced bias analysis.*

## Overview

This comprehensive tool provides bias detection, privacy analysis, and governance compliance for machine learning models. It's specifically designed to help organizations ensure their AI systems are fair, transparent, and compliant with regulations like GDPR, CCPA, and other privacy frameworks.

## Features

### 🔍 **Comprehensive Analysis**
- **Bias Detection**: Multi-dimensional fairness metrics and bias detection
- **Privacy Analysis**: Sensitive data identification and privacy risk assessment
- **Explainability**: Model interpretability using SHAP, LIME, and feature importance
- **Governance**: Compliance reporting and recommendations

### ⚖️ **Bias Analysis**
- Demographic parity analysis
- Equalized odds assessment
- Intersectional bias detection
- Fairness constraint implementation
- Bias mitigation strategies

### 🔒 **Privacy Features**
- Sensitive data pattern detection
- Re-identification risk assessment
- K-anonymity analysis
- GDPR/CCPA compliance checking
- Data anonymization recommendations

### 🧠 **Explainability**
- SHAP (SHapley Additive exPlanations) analysis
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance ranking
- Model complexity assessment
- Interactive visualizations

### 📊 **Governance & Compliance**
- Automated compliance reporting
- Risk assessment frameworks
- Audit trail generation
- Recommendation engine
- Export capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup with Virtual Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/dionjoshualobo/FairLens.git
cd FairLens

# Create and activate virtual environment
python -m venv fairlens-env
source fairlens-env/bin/activate  # On Windows: fairlens-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```
#### For Windows Users:
1. **Clone or download the project files**
2. **Create a virtual environment**:
   ```cmd
   python -m venv fairlens-env
   ```
3. **Activate the virtual environment**:
   ```cmd
   fairlens-env\Scripts\activate
   ```
4. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
5. **Start the application**:
   ```cmd
   streamlit run main.py
   ```

#### For Linux/Mac Users:
1. **Clone or download the project files**
2. **Create a virtual environment**:
   ```bash
   python -m venv fairlens-env
   ```
3. **Activate the virtual environment**:
   ```bash
   source fairlens-env/bin/activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Start the application**:
   ```bash
   streamlit run main.py
   ```

### Deactivating the Virtual Environment
When you're done working with the project, you can deactivate the virtual environment:
```bash
deactivate
```

### Alternative: Manual Installation (Not Recommended)
If you prefer not to use a virtual environment (though not recommended):
```bash
pip install -r requirements.txt
```

### Why Use a Virtual Environment?
- **Isolation**: Prevents conflicts between different project dependencies
- **Reproducibility**: Ensures consistent package versions across different systems
- **Clean Environment**: Keeps your system Python installation clean
- **Easy Management**: Simple to create, delete, and recreate environments

### First-Time Setup Notes
- The application will automatically detect your data types when you upload a CSV file
- Sample datasets are provided in the `Datasets/` folder for testing
- The web interface will be available at `http://localhost:8501` after running the Streamlit command

## Usage

### 1. **Data Loading**
- Upload your CSV file or use the default loan_data.csv
- The tool supports various data formats and automatically detects data types

### 2. **Privacy Analysis**
- Automatic detection of sensitive features (PII, demographic data, etc.)
- Privacy risk scoring for each feature
- Compliance assessment for GDPR, CCPA, HIPAA
- Re-identification risk analysis

### 3. **Bias Analysis**
- Select sensitive attributes for fairness analysis
- Choose target variable for prediction
- Review statistical bias indicators
- Get bias mitigation recommendations

### 4. **Model Training**
- Train multiple models (Random Forest, Logistic Regression, Gradient Boosting)
- Automatic model comparison and selection
- Fairness-aware model training options

### 5. **Explainability Analysis**
- Global feature importance analysis
- Local explanation for individual predictions
- Feature interaction analysis
- Model complexity assessment

### 6. **Governance Reporting**
- Comprehensive compliance reports
- Executive summaries
- Actionable recommendations
- Export functionality

## File Structure

```
├── main.py                 # Main Streamlit application
├── bias_detection.py       # Advanced bias detection module
├── privacy_analysis.py     # Privacy analysis and compliance
├── explainability.py       # Model explainability module
├── setup.py               # Installation and setup script
├── requirements.txt       # Python package dependencies
├── config.py             # Configuration settings
├── run.py                # Quick start script
└── Datasets/
    ├── loan_data.csv     # Sample dataset
    ├── Loan_Default.csv  # Additional sample data
    ├── Test.csv          # Test dataset
    └── Train.csv         # Training dataset
```

## Technical Stack

### Core Libraries
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and metrics

### Fairness & Bias
- **Fairlearn**: Microsoft's fairness toolkit
- **Custom Bias Detection**: Advanced bias metrics and detection algorithms

### Explainability
- **SHAP**: Model-agnostic explanations
- **LIME**: Local interpretable explanations
- **Feature Importance**: Multiple importance calculation methods

### Visualization
- **Plotly**: Interactive charts and visualizations
- **Matplotlib/Seaborn**: Statistical visualizations

### Privacy & Compliance
- **Custom Privacy Module**: Pattern detection and risk assessment
- **Compliance Frameworks**: GDPR, CCPA, HIPAA assessment tools

## Configuration

The tool can be configured through `config.py`:

```python
# Model Configuration
DEFAULT_MODEL = "RandomForest"
MODEL_PARAMS = {
    "RandomForest": {"n_estimators": 100, "random_state": 42},
    "LogisticRegression": {"random_state": 42, "max_iter": 1000}
}

# Privacy Thresholds
PRIVACY_THRESHOLDS = {
    "high_risk_threshold": 3,
    "k_anonymity_min": 3
}

# Bias Thresholds
BIAS_THRESHOLDS = {
    "demographic_parity_threshold": 0.1,
    "equalized_odds_threshold": 0.1
}
```

## Key Metrics & Methods

### Bias Detection
- **Demographic Parity**: Measures if different groups receive positive outcomes at equal rates
- **Equalized Odds**: Measures if error rates are equal across groups
- **Equal Opportunity**: Focuses on true positive rates across groups
- **Intersectional Analysis**: Examines bias across multiple sensitive attributes

### Privacy Assessment
- **K-Anonymity**: Measures uniqueness of individual records
- **Re-identification Risk**: Assesses likelihood of identifying individuals
- **Sensitive Data Detection**: Identifies PII and sensitive attributes
- **Compliance Scoring**: Rates adherence to privacy regulations

### Explainability Metrics
- **SHAP Values**: Quantifies feature contribution to predictions
- **Feature Importance**: Multiple methods for ranking feature significance
- **Model Complexity**: Assesses interpretability based on model structure
- **Local Explanations**: Individual prediction explanations

## Best Practices

### Data Preparation
1. **Clean your data** before analysis
2. **Identify sensitive attributes** early in the process
3. **Ensure sufficient sample sizes** for all demographic groups
4. **Document data sources** and collection methods

### Bias Analysis
1. **Select appropriate fairness metrics** based on your use case
2. **Consider intersectional effects** when multiple sensitive attributes exist
3. **Evaluate trade-offs** between different fairness criteria
4. **Implement bias monitoring** in production systems

### Privacy Protection
1. **Apply data minimization** principles
2. **Implement privacy by design**
3. **Regular privacy impact assessments**
4. **Maintain audit trails** for compliance

### Model Governance
1. **Document all modeling decisions**
2. **Implement continuous monitoring**
3. **Regular model retraining** and evaluation
4. **Stakeholder communication** of results and limitations

## Troubleshooting

### Common Issues

1. **Virtual Environment Issues**:
   - Make sure you've activated the virtual environment before installing packages or running the app
   - If packages seem missing, check that you're in the correct environment
   - To recreate the environment: delete the `fairlens-env` folder and follow setup steps again

2. **Import Errors**: 
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're running Python 3.8 or higher: `python --version`

3. **Memory Issues**: Reduce sample sizes for SHAP/LIME analysis

4. **Performance**: Use data sampling for large datasets

5. **Visualization Issues**: Ensure Plotly is properly installed: `pip install plotly>=5.0.0`

6. **Streamlit Port Issues**: If port 8501 is busy, Streamlit will automatically use the next available port

### Performance Optimization

- **Data Sampling**: Use representative samples for analysis
- **Feature Selection**: Reduce dimensionality before analysis
- **Parallel Processing**: Enable where available
- **Caching**: Results are cached to improve performance

## Advanced Features

### Custom Bias Metrics
You can implement custom bias detection algorithms by extending the `AdvancedBiasDetector` class:

```python
class CustomBiasDetector(AdvancedBiasDetector):
    def custom_fairness_metric(self, y_true, y_pred, sensitive_features):
        # Implement your custom metric
        pass
```

### Privacy Anonymization
The tool supports multiple anonymization techniques:
- K-anonymity through generalization
- Differential privacy through noise addition
- Custom anonymization strategies

### Model Explainability
Advanced explainability features include:
- Global and local explanations
- Feature interaction analysis
- Counterfactual explanations
- Model complexity scoring

## Compliance Frameworks

### GDPR Compliance
- Data minimization assessment
- Purpose limitation checking
- Storage limitation evaluation
- Data subject rights assessment

### CCPA Compliance
- Consumer rights evaluation
- Data sale assessment
- Opt-out mechanism review
- Non-discrimination analysis

### HIPAA Compliance
- Technical safeguards assessment
- Administrative safeguards review
- Physical safeguards evaluation
- Minimum necessary standard

### Indian Compliance Frameworks
#### DPDP Act (Digital Personal Data Protection Act, 2023)
- Data processing consent management
- Data principal rights assessment
- Data fiduciary obligations review
- Cross-border data transfer compliance

#### IT Act (Information Technology Act, 2000)
- Sensitive personal data protection
- Reasonable security practices evaluation
- Data breach notification review

## Export & Reporting

The tool generates comprehensive reports including:
- **Executive Summary**: High-level findings and recommendations
- **Technical Report**: Detailed analysis results
- **Compliance Report**: Regulatory compliance assessment
- **Action Plan**: Specific recommendations and next steps

## Support & Development

### Getting Help
1. Check the documentation and troubleshooting sections
2. Review the sample datasets and examples
3. Examine the configuration options

### Contributing
The tool is designed to be extensible. You can:
- Add new bias detection algorithms
- Implement additional privacy techniques
- Extend compliance frameworks
- Enhance visualization capabilities

### Roadmap
Future enhancements may include:
- Real-time monitoring capabilities
- Additional ML model support
- Enhanced visualization options
- Integration with MLOps platforms
- Advanced anonymization techniques

## License & Disclaimer

This tool is provided for educational and research purposes. Users are responsible for ensuring compliance with applicable laws and regulations. The tool provides guidance and analysis but does not guarantee legal compliance.

---

For technical support or questions about implementation, please refer to the documentation or create an issue in the project repository.
