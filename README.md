# House Price Prediction

This project builds a simple machine learning pipeline to predict house prices using a housing dataset. The workflow includes data exploration, preprocessing, model training, evaluation, and visualization.

## Project Overview

The project uses a tabular housing dataset to train a Linear Regression model that predicts house prices based on features such as area, number of bedrooms, bathrooms, furnishing status, and amenities.

## Project Structure

- code/Data Loading & Exploration (EDA).py - loads and explores the dataset
- code/Data_preprocessing.py - converts categorical variables into numeric format and prepares the final dataset
- code/Model_Training.py - trains the regression model and saves the trained model and test data
- code/Prediction_&_Evaluation.py - evaluates the model using MAE, MSE, and R2 score
- code/Visualization.py - plots actual versus predicted house prices
- Data/ - contains the input and generated CSV files
- requirements.txt - Python dependencies for the project

## Requirements

Make sure you have Python installed, then install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

Run the scripts in the following order:

```bash
python "code/Data Loading & Exploration (EDA).py"
python code/Data_preprocessing.py
python code/Model_Training.py
python code/Prediction_&_Evaluation.py
python code/Visualization.py
```

> Note: The scripts read and write CSV files from the current working directory. If you run them from a different location, update the file paths accordingly.

## Output Files

The pipeline generates the following files:

- processed_data.csv - dataset after initial loading/exploration
- final_data.csv - preprocessed dataset used for training
- house_model.pkl - trained regression model
- X_test.csv and y_test.csv - test data used for evaluation

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- joblib
