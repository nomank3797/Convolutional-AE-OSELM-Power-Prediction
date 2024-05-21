import numpy as np
import pandas as pd
import data_processing # This is a custom module
import cnn_ae_oselm # This is a custom module

# Load the dataset from the 'raw data' directory
# Assuming 'household_power_consumption.txt' contains the dataset
dataset = pd.read_csv('raw data/household_power_consumption.txt', sep=';', header=0, low_memory=False,
                      infer_datetime_format=True, parse_dates={'datetime': [0, 1]}, index_col=['datetime'])

# Choose data frequency
frequency = 'H'  # Data frequency is set to hourly

# Number of epochs for training
epochs = 100

# Define the filename for the CSV file to save predictions
file_name = 'hourly_actual_predicted_values.csv'

# Clean the data
cleaned_data = data_processing.clean_data(dataset, frequency)

# Split the dataset into input sequences (X) and output sequences (y)
X, y = data_processing.split_sequences(cleaned_data)

# Model training and testing
cnn_ae_oselm.cnn_ae_oselm_model(X, y, epochs, file_name)
