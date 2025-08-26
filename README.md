# College Student Stress Classifier

A machine learning project that analyzes and classifies stress levels in college students using various psychological and physiological indicators. This project implements multiple classification algorithms to predict student stress levels and provides insights into the most important factors contributing to stress.

## Project Overview

This project uses a dataset of 1,100 college students with 21 different features including:
- **Psychological factors**: Anxiety level, self-esteem, depression, mental health history
- **Physical symptoms**: Headache, blood pressure, sleep quality, breathing problems
- **Environmental factors**: Noise level, living conditions, safety, basic needs
- **Academic factors**: Academic performance, study load, teacher-student relationship
- **Social factors**: Social support, peer pressure, extracurricular activities, bullying
- **Future concerns**: Future career concerns

The target variable is **stress level** classified into three categories (0, 1, 2).

## Objectives

- Analyze college student stress patterns using machine learning
- Compare performance of different classification algorithms
- Identify the most important factors contributing to student stress
- Provide insights for educational institutions and mental health professionals

## Technologies Used

- **Python 3.11**
- **Machine Learning Libraries**:
  - scikit-learn (Random Forest, Logistic Regression)
  - TensorFlow (Neural Networks)
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Development Environment**: Jupyter Notebook

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stress
   ```

2. **Set up virtual environment** (recommended):
   ```bash
   python -m venv tfenv
   source tfenv/bin/activate  # On macOS/Linux
   # or
   tfenv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Activate the virtual environment**:
   ```bash
   source tfenv/bin/activate  # On macOS/Linux
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open and run the main notebook**:
   - Open `main.ipynb`
   - Run all cells to execute the complete analysis

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **89.1%** | **89.0%** | **89.0%** | **89.0%** |
| Logistic Regression | 88.2% | 88.0% | 88.0% | 88.0% |
| Neural Network | 88.2% | 88.0% | 88.0% | 88.0% |

### Key Findings

1. **Best Performing Model**: Random Forest achieved the highest accuracy (89.1%)
2. **Balanced Performance**: All models performed consistently across different stress levels
3. **Feature Importance**: Blood pressure emerged as the most critical indicator across all models

### Top 5 Most Important Features (Random Forest)

1. **Blood Pressure** (14.52%)
2. **Sleep Quality** (8.09%)
3. **Teacher-Student Relationship** (7.13%)
4. **Academic Performance** (6.64%)
5. **Basic Needs** (6.61%)

## Project Structure

```
stress/
├── data/
│   ├── StressLevelDataset.csv    # Main dataset (1,100 samples)
│   └── Stress_Dataset.csv        # Additional dataset
├── main.ipynb                    # Main analysis notebook
├── requirements.txt              # Python dependencies
├── tfenv/                        # Virtual environment
└── README.md                     # This file
```

## Data Preprocessing

The project includes comprehensive data preprocessing:
- **Data Quality**: No missing values or duplicates found
- **Feature Engineering**: Categorical and continuous variables properly identified
- **Scaling**: StandardScaler applied to continuous variables for appropriate models
- **Train-Test Split**: 80-20 split with stratification to preserve class distribution

## Models Implemented

### 1. Random Forest Classifier
- **Advantages**: Handles non-linear relationships, feature importance ranking
- **Performance**: Best overall accuracy (89.1%)
- **No preprocessing required** for categorical features

### 2. Logistic Regression
- **Advantages**: Interpretable coefficients, fast training
- **Performance**: 88.2% accuracy
- **Preprocessing**: StandardScaler applied to continuous variables

### 3. Neural Network (TensorFlow)
- **Architecture**: 2 hidden layers (64, 32 neurons) with ReLU activation
- **Performance**: 88.2% accuracy
- **Preprocessing**: StandardScaler applied to continuous variables

## Visualizations

The project includes comprehensive visualizations:
- Stress level distribution
- Per-class performance metrics (Precision, Recall, F1-Score, Accuracy)
- Feature importance rankings
- Confusion matrices

## Applications

This analysis can be valuable for:
- **Educational Institutions**: Identifying at-risk students
- **Mental Health Professionals**: Understanding stress indicators
- **Student Support Services**: Developing targeted intervention programs
- **Research**: Academic studies on student mental health

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- **Shreyas Ramulu** - Initial work

## 🙏 Acknowledgments

- Dataset providers for making this research possible
- Open source community for the excellent libraries used
- Educational institutions supporting student mental health research

---

**Note**: This project is for educational and research purposes. For actual mental health applications, please consult with qualified professionals.
