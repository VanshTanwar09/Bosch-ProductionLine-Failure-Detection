# Bosch ProductionLine Failure Detection-Research Based Project

## Important
"I will extend my Bosch Production Failure Detection 
project (AUC 0.646) through independent research on hyperparameter optimization 
and explainable AI, targeting publication in IEEE Sensors Journal."

## Overview
This is a Streamlit application for predicting production failures in Bosch manufacturing lines. The app uses 
mathematical modeling and machine learning to analyze sensor data and provide real-time failure risk assessments, helping prevent costly downtime in Industry 4.0 environments.

## Features

### Live Prediction
- Real-time failure risk prediction using 3 key sensor inputs
- Interactive dashboard with risk visualization
- Configurable alert thresholds
- Professional UI with Industry 4.0 styling

### Risk Visualization
- Polar chart showing safe vs risky probability distribution
- Dynamic updates based on prediction results
- Color-coded risk levels (green for safe, red for risky)

### Batch Analysis
- Upload CSV files for bulk prediction analysis
- Automatic risk scoring and alerting
- Downloadable results with risk metrics
- Statistical summaries of high-risk stations

### Mathematical Modeling
- ROC-AUC Score: 0.646
- 100 Trees Trained
- Advanced ML techniques including GridSearchCV, ROC analysis, and feature importance ranking

## Technical Details

### Model Performance
- **AUC Score**: 0.646
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Features**: 200 sensor inputs processed
- **Training Data**: 20,000+ sensor matrix samples

### Technologies Used
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly Express
- **Machine Learning**: XGBoost
- **Model Storage**: JSON format

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy plotly xgboost
   ```
3. **Important**: The model files (`bosch_model.json` and `bosch_model.pkl`) are not included in this repository due to their large size (exceeds GitHub's 25MB limit). You must create them yourself by running the training notebook.

## Data Files

The following data files are not included in this repository due to their large size:
- `X_clean.csv` (processed features)
- `y_clean.csv` (target labels)
- `train_numeric.csv` (original Bosch numeric data)
- `train_categorical.csv` (original Bosch categorical data)

To obtain these files:

1. Download the Bosch Production Line Performance dataset from Kaggle: https://www.kaggle.com/c/bosch-production-line-performance
2. Place the downloaded `train_numeric.csv` and `train_categorical.csv` in the project directory
3. Run the preprocessing steps in `Bosch Production Failure Detection.ipynb` to generate `X_clean.csv` and `y_clean.csv`

## Creating the Model Files

Since the trained model files are too large for GitHub, you'll need to generate them locally:

1. Ensure all data files are available (see Data Files section above)
2. Open `Bosch Production Failure Detection.ipynb` in Jupyter Notebook
3. Run all cells to train the XGBoost model
4. The notebook will save `bosch_model.json` and `bosch_model.pkl` in the project directory
5. Once the model files are created, run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Live Prediction
1. Enter values for the 3 sensor inputs (L3_S29_F3464, L3_S29_F3470, L3_S29_F3379)
2. Click "PREDICT FAILURE" button
3. View risk assessment and status
4. Adjust alert threshold in sidebar if needed

### Batch Analysis
1. Upload a CSV file with sensor data
2. View risk distribution histogram
3. Download predictions with risk scores

## Model Information

The XGBoost model was trained on Bosch production data with:
- GridSearchCV hyperparameter optimization
- ROC Curve analysis for threshold selection
- Feature importance ranking for sensor prioritization
- 200×20K sensor matrix processing

## Contributing

This project demonstrates Industry 4.0 AI applications for predictive maintenance.

## License

Academic use only - Bosch Production Failure Detection Demo
