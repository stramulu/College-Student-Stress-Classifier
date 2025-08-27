# College Student Stress Classifier & AI Chatbot

A comprehensive machine learning project that analyzes and classifies stress levels in college students using various psychological and physiological indicators. This project includes both a machine learning model for stress prediction and an interactive Streamlit web application with an AI chatbot for personalized stress management advice.

## Project Overview

This project addresses the critical issue of student mental health by developing a machine learning system that can predict stress levels based on multiple factors. The system uses a dataset of 1,100 college students with 21 different features covering:

- **Psychological factors**: Anxiety level, self-esteem, depression, mental health history
- **Physical symptoms**: Headache, blood pressure, sleep quality, breathing problems
- **Environmental factors**: Noise level, living conditions, safety, basic needs
- **Academic factors**: Academic performance, study load, teacher-student relationship
- **Social factors**: Social support, peer pressure, extracurricular activities, bullying
- **Future concerns**: Future career concerns

The target variable is **stress level** classified into three categories:
- 0: Low Stress
- 1: Moderate Stress  
- 2: High Stress

## Objectives

- Analyze college student stress patterns using machine learning algorithms
- Compare performance of different classification algorithms (Random Forest, Logistic Regression, Neural Networks)
- Identify the most important factors contributing to student stress through feature importance analysis
- Provide an interactive web application for real-time stress prediction
- Offer AI-powered chatbot for personalized stress management advice and interpretation of results
- Provide insights for educational institutions and mental health professionals
- Create a deployable model that can be used for practical applications

## Technologies Used

### Core Technologies
- **Python 3.11** - Primary programming language
- **Jupyter Notebook** - Development and analysis environment

### Machine Learning Libraries
- **scikit-learn** - Random Forest, Logistic Regression, data preprocessing
- **TensorFlow** - Neural Network implementation
- **joblib** - Model serialization and persistence

### Data Analysis & Visualization
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **matplotlib** - Static visualizations
- **seaborn** - Statistical visualizations

### Web Application
- **Streamlit** - Interactive web application framework
- **OpenAI API** - GPT-3.5-turbo for AI chatbot functionality

### Environment Management
- **virtualenv** - Python virtual environment
- **dotenv** - Environment variable management

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stress
   ```

2. **Set up virtual environment**:
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

4. **Set up OpenAI API key** (for chatbot functionality):
   ```bash
   # Create a .env file in the project root
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Environment Variables
Create a `.env` file in the project root with the following content:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Web Application

1. **Activate the virtual environment**:
   ```bash
   source tfenv/bin/activate  # On macOS/Linux
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`
   - Use the "Stress Prediction" tab to input your information and get stress predictions
   - Use the "AI Chatbot" tab to ask questions about stress management and wellbeing

### Using the Jupyter Notebook

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open and run the analysis notebook**:
   - Open `models.ipynb`
   - Run all cells to execute the complete analysis

### Model Files
- The trained Random Forest model is saved as `stress_random_forest_model.pkl`
- The model can be loaded and used for predictions without retraining

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **89.0%** | **89.0%** | **89.0%** | **89.0%** |
| Logistic Regression | 88.2% | 88.0% | 88.0% | 88.0% |
| Neural Network | 88.2% | 88.0% | 88.0% | 88.0% |

### Key Findings

1. **Best Performing Model**: Random Forest achieved the highest accuracy (89.0%)
2. **Balanced Performance**: All models performed consistently across different stress levels
3. **Feature Importance**: Blood pressure emerged as the most critical indicator across all models
4. **Deployed Model**: The Random Forest model has been saved and deployed in the web application

### Top 10 Most Important Features (Random Forest)

1. **Blood Pressure** (14.52%)
2. **Sleep Quality** (8.09%)
3. **Teacher-Student Relationship** (7.13%)
4. **Academic Performance** (6.64%)
5. **Basic Needs** (6.61%)
6. **Depression** (6.54%)
7. **Self-Esteem** (6.38%)
8. **Social Support** (5.72%)
9. **Anxiety Level** (5.63%)
10. **Bullying** (5.00%)

### Model Evaluation Metrics

The Random Forest model achieved the following evaluation metrics on the test set:
- **Accuracy**: 0.89
- **Precision**: 0.89
- **Recall**: 0.89
- **F1 Score**: 0.89
- **Confusion Matrix**: Shows strong correct classification for all three stress levels

## Project Structure

```
stress/
├── data/
│   ├── StressLevelDataset.csv    # Main dataset (1,100 samples)
│   └── Stress_Dataset.csv        # Additional dataset (843 samples)
├── app.py                        # Streamlit web application
├── models.ipynb                  # Jupyter notebook with model analysis
├── stress_random_forest_model.pkl # Trained Random Forest model
├── requirements.txt              # Python dependencies
├── tfenv/                        # Virtual environment
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## Data Preprocessing

### Data Quality Assessment
- **Missing Values**: No missing values found in the dataset
- **Duplicate Rows**: No duplicate entries detected
- **Data Types**: All features are properly typed as integers
- **Dataset Size**: 1,100 samples with 21 features

### Feature Engineering
- **Categorical Variables**: 16 features identified as categorical
- **Binary Variables**: 1 feature (mental health history)
- **Continuous Variables**: 3 features (anxiety level, self-esteem, depression)
- **Target Variable**: Stress level (0, 1, 2)

### Data Scaling
- **StandardScaler**: Applied to continuous variables for Logistic Regression and Neural Network
- **No Scaling Required**: Random Forest handles mixed data types without preprocessing

### Train-Test Split
- **Split Ratio**: 80% training, 20% testing
- **Stratification**: Preserves class distribution across splits
- **Random State**: 42 for reproducible results

## Models Implemented

### 1. Random Forest Classifier (Deployed)
- **Algorithm**: Ensemble learning with multiple decision trees
- **Advantages**: 
  - Handles non-linear relationships
  - Provides feature importance ranking
  - No preprocessing required for categorical features
  - Robust to overfitting
- **Performance**: Best overall accuracy (89.0%)
- **Hyperparameters**: 100 estimators, random state 42
- **Deployed**: Saved as `stress_random_forest_model.pkl`

### 2. Logistic Regression
- **Algorithm**: Linear classification with regularization
- **Advantages**: 
  - Interpretable coefficients
  - Fast training and prediction
  - Probabilistic outputs
- **Performance**: 88.2% accuracy
- **Preprocessing**: StandardScaler applied to continuous variables
- **Solver**: lbfgs with default parameters

### 3. Neural Network (TensorFlow)
- **Architecture**: 
  - Input layer: 20 features
  - Hidden layer 1: 64 neurons with ReLU activation
  - Hidden layer 2: 32 neurons with ReLU activation
  - Output layer: 3 neurons with softmax activation
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Categorical crossentropy
- **Performance**: 88.2% accuracy
- **Training**: 30 epochs, batch size 32

## Web Application Features

### Stress Prediction Tab
- **Interactive Input Forms**: 
  - Sliders for numerical features (0-30 scale for psychological factors, 0-5 scale for others)
  - Dropdown for binary features (mental health history)
- **Real-time Prediction**: Instant stress level classification
- **User-friendly Interface**: Clear labels and intuitive controls
- **Results Display**: Color-coded stress level output with confidence indicators

### AI Chatbot Tab
- **OpenAI Integration**: Powered by GPT-3.5-turbo
- **Specialized Knowledge**: 
  - Trained on stress management and wellbeing
  - Understands the prediction model and dataset
  - Provides contextual responses about the application
- **Personalized Advice**: Offers tailored recommendations based on user queries
- **Conversation History**: Maintains chat context throughout the session

### System Prompt
The chatbot uses a comprehensive system prompt that includes:
- Dataset information and methodology
- Feature descriptions and coding
- Model performance metrics
- Feature importance analysis
- Application functionality explanation

## Visualizations

### Analysis Visualizations
- **Stress Level Distribution**: Bar chart showing class balance
- **Per-class Performance Metrics**: Precision, Recall, F1-Score, Accuracy by stress level
- **Feature Importance Rankings**: Bar charts for each model
- **Confusion Matrices**: Model performance visualization
- **Correlation Analysis**: Feature relationships

### Web Application Visualizations
- **Interactive Charts**: Real-time data visualization
- **Performance Metrics**: Model evaluation displays
- **Feature Importance**: Visual representation of key factors

## Applications

### Educational Institutions
- **Early Intervention**: Identify students at risk of high stress
- **Resource Allocation**: Target mental health resources effectively
- **Policy Development**: Inform stress management policies
- **Academic Support**: Provide targeted academic assistance

### Mental Health Professionals
- **Assessment Tool**: Quick stress level screening
- **Treatment Planning**: Inform intervention strategies
- **Research**: Support academic studies on student mental health
- **Prevention**: Proactive stress management approaches

### Student Support Services
- **Counseling Centers**: Screening and triage tool
- **Academic Advisors**: Student wellbeing assessment
- **Residence Life**: Dormitory stress monitoring
- **Health Services**: Integrated mental health screening

### Individual Students
- **Self-assessment**: Personal stress level evaluation
- **Awareness**: Understanding stress indicators
- **Management**: Personalized coping strategies
- **Prevention**: Proactive stress reduction

## Configuration

### Environment Setup
- **Python Version**: 3.11 or higher
- **Virtual Environment**: tfenv (included)
- **Dependencies**: All listed in requirements.txt
- **API Keys**: OpenAI API key required for chatbot

### Model Configuration
- **Model File**: stress_random_forest_model.pkl
- **Input Features**: 20 features in specific order
- **Output Classes**: 3 stress levels (0, 1, 2)
- **Prediction Threshold**: No threshold (direct classification)

### Web Application Configuration
- **Port**: Default Streamlit port (8501)
- **Host**: localhost
- **Session State**: Maintains chat history and user inputs
- **Error Handling**: Graceful handling of API failures

### Code Style
- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add comments for complex logic
- Include docstrings for functions and classes

## License

This project is open source and available under the [MIT License](LICENSE).

## Authors

- **Shreyas Ramulu** - Initial work and project development

## Acknowledgments

- **Dataset Providers**: For making this research possible with comprehensive student stress data
- **OpenAI**: For providing the GPT-3.5-turbo API for chatbot functionality
- **Streamlit**: For the excellent web application framework
- **Open Source Community**: For the excellent libraries and tools used in this project
- **Educational Institutions**: For supporting student mental health research
- **Machine Learning Community**: For the algorithms and methodologies that made this project possible

## Important Notes

### Educational Purpose
This project is designed for educational and research purposes. It demonstrates machine learning techniques, data analysis, and web application development.

### Professional Consultation
For actual mental health applications, please consult with qualified mental health professionals. This tool should not replace professional medical or psychological advice.

### Data Privacy
The application does not store user data permanently, but users should be mindful of privacy when using the chatbot. No personal information is collected or stored.

### API Key Security
- Keep your OpenAI API key secure
- Never commit API keys to version control
- Use environment variables for sensitive information
- Regularly rotate your API keys

### Model Limitations
- The model is trained on a specific dataset and may not generalize to all populations
- Stress levels are self-reported and may not reflect clinical assessments
- The model should be used as a screening tool, not a diagnostic instrument

## Getting Started

To get started with this project:

1. **Clone the repository** and follow the installation instructions
2. **Set up your environment** with the required dependencies
3. **Configure your API key** for the chatbot functionality
4. **Run the web application** with `streamlit run app.py`
5. **Explore the Jupyter notebook** for detailed analysis
6. **Experiment with the models** and contribute improvements

For questions or support, please open an issue in the GitHub repository.
