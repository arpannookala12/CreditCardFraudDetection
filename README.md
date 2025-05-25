# Credit Card Fraud Detection Using Machine Learning

![Credit Card Fraud Detection](https://img.shields.io/badge/Machine%20Learning-Credit%20Card%20Fraud%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.7+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive machine learning project for detecting fraudulent credit card transactions using advanced regression techniques, feature engineering with Weight of Evidence (WOE), and SMOTE for handling imbalanced datasets.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Performance](#model-performance)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## 🎯 Overview

Credit card fraud detection is crucial for financial institutions to protect customers from unauthorized transactions. This project implements a robust machine learning pipeline that:

- Handles highly imbalanced datasets (0.172% fraud cases)
- Uses advanced feature engineering techniques
- Implements SMOTE for synthetic minority oversampling
- Applies Weight of Evidence (WOE) transformation
- Provides risk scoring from 0-100 with categorized risk levels

**Key Quote**: *"The greatest value of a picture is when it forces us to notice what we never expected to see."* - John W. Tukey

## 📊 Dataset Description

The dataset comprises credit card transactions over a **two-day period in September 2013** with the following characteristics:

- **Total Transactions**: 284,807
- **Fraud Cases**: 492 (0.172% of total)
- **Features**: 30 total
  - `V1-V28`: PCA-transformed features (confidential)
  - `Time`: Elapsed seconds since first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Non-fraud, 1 = Fraud)

### Dataset Highlights
- **Average Transaction Amount**: ~$66 USD
- **No Missing Values**: Clean dataset requiring no imputation
- **Highly Imbalanced**: 99.83% non-fraud vs 0.17% fraud transactions

## ✨ Key Features

### 1. **Advanced Data Preprocessing**
- **SMOTE Implementation**: Synthetic Minority Over-sampling Technique
  - Increased fraud cases from 469 to 65,598
  - Balanced dataset for improved model training

### 2. **Feature Engineering**
- **Weight of Evidence (WOE)** transformation
- **Information Value (IV)** calculation for feature selection
- **Optimal Binning** algorithms for continuous variables

### 3. **Risk Scoring System**
- Fraud scores normalized to 0-100 range
- Risk categorization:
  - **No Risk** (0-25)
  - **Low Risk** (25-50)
  - **Moderate Risk** (50-75)
  - **High Risk** (75-100)

## 🔬 Methodology

### 1. Data Preprocessing
```python
# SMOTE Implementation
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.3)
X_res, y_res = smote.fit_resample(X, y)
```

### 2. Feature Engineering Pipeline
- **Binning**: Continuous variables split into optimal bins
- **WOE Calculation**: `WOE = ln(% of non-events / % of events)`
- **Information Value**: `IV = Σ(% of non-events - % of events) * WOE`

### 3. Feature Selection Criteria
- **IV > 0.3**: Strong relationship to fraud detection
- **IV 0.1-0.3**: Medium strength relationship
- **IV < 0.1**: Weak or no relationship

### 4. Model Training
- **Algorithm**: Logistic Regression
- **Input**: WOE-transformed features
- **Output**: Probability scores converted to risk categories

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
```

### Required Libraries
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

## 💻 Usage

### Basic Implementation
```python
# Load and preprocess data
from fraud_detection import FraudDetector

# Initialize detector
detector = FraudDetector()

# Load dataset
data = detector.load_data('creditcard.csv')

# Apply SMOTE
X_balanced, y_balanced = detector.apply_smote(data)

# Feature engineering
X_transformed = detector.woe_transform(X_balanced)

# Train model
model = detector.train_model(X_transformed, y_balanced)

# Generate predictions
predictions = detector.predict(X_test)
risk_scores = detector.calculate_risk_scores(predictions)
```

### Risk Score Calculation
```python
# Calculate fraud scores
coefficients = model.coef_
log_odds = np.dot(X_test_trans, coefficients.T)
offset = 600
factor = 50
fraud_scores = offset + factor * log_odds

# Normalize to 0-100 range
normalized_scores = np.clip(fraud_scores, 0, 100)
```

## 📈 Results

### Model Performance
- **ROC AUC Score**: 0.78
- **Algorithm**: Logistic Regression with WOE transformation

### Risk Distribution
Based on the fraud risk donut chart:
- **Moderate Risk**: ~58.1%
- **High Risk**: ~24%
- **Low Risk**: ~14.7%
- **No Risk**: ~5.18%

### Sample Predictions
| Transaction ID | V2 | V3 | V10 | V14 | V18 | V17 | Fraud Score | Risk Category |
|---------------|----|----|-----|-----|-----|-----|-------------|---------------|
| 85268 | 0.481922 | 0.655199 | 0.372493 | -0.262860 | 0.676819 | 0.264184 | 95.170772 | High Risk |
| 85269 | -0.958113 | -0.993901 | -2.041699 | 0.309143 | -1.099517 | 0.264184 | 41.752347 | No Risk |

## 🎯 Model Performance

### ROC Curve Analysis
- **Area Under Curve (AUC)**: 0.78
- Demonstrates good separation between fraud and non-fraud cases
- Significant improvement over random classification

### Feature Importance
- Selected features based on **Information Value > 0.3**
- WOE transformation provides interpretable coefficients
- Monotonic relationship maintained through optimal binning

## 🔮 Future Scope

### 1. **Real-Time Integration**
- Adapt models for real-time fraud detection
- Implement streaming data processing
- Advanced anomaly detection techniques

### 2. **Enhanced Security Measures**
- Biometric authentication integration
- Behavioral analysis patterns
- Multi-factor authentication systems

### 3. **Advanced Technologies**
- **Blockchain Integration**: Enhanced transaction security
- **Deep Learning Models**: Neural networks for pattern recognition
- **Ensemble Methods**: Combining multiple algorithms

### 4. **Cross-Industry Applications**
- Insurance fraud detection
- Healthcare fraud prevention
- E-commerce transaction monitoring

### 5. **Dynamic Adaptation**
- Self-learning models
- Adaptive credit limit management
- Economic condition integration

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting

## 📚 References

1. [Monotone Optimal Binning Algorithm for Credit Risk Modeling](https://www.researchgate.net/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling)
2. [Credit Card Fraud Detection: A Review of the State of the Art](https://arxiv.org/abs/1611.06439)
3. [Building a Machine Learning Model for Credit Card Fraud Detection](https://towardsdatascience.com/credit-card-fraud-detection-using-machine-learning-python-5b098d4a8edc)
4. [Weight of Evidence & Information Value](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
5. [SMOTE: Synthetic Minority Over-sampling Technique](https://medium.com/@corymaklin/synthetic-minority-over-sampling-technique-smote-7d419696b88c)

## 👥 Team

**Course**: Regression and Time Series Analysis  
**Project Date**: December 12, 2023

| Name | Net ID |
|------|--------|
| Ronit Gandhi | rg1225 |
| Ganesh Arpan Nookala | gn178 |
| Kamala Prerna Nookala | kn491 |
| Aditya Singhal | as4622 |
| Hang Wang | hw598 |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by the Machine Learning community
- Research papers and academic sources for methodology guidance
- Open-source libraries that made this project possible

---

**⭐ If you found this project helpful, please consider giving it a star!**

For questions or support, please open an issue or contact the development team.
