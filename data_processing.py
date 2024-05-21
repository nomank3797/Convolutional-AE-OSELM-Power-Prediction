import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import nan, isnan
from scipy.signal import savgol_filter

# Fill missing values with a mean value
def fill_missing(data):
	data = data.fillna(data.mean())
	return data

# Normalize data	
def normalize_data(values):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    values (numpy.ndarray): Input data to be normalized.
    
    Returns:
    numpy.ndarray: Normalized data.
    """
    # Create scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on data
    scaler.fit(values)
    # Apply transform
    normalized = scaler.transform(values)
    return normalized

# Clean-up data	
def clean_data(data, frequency='M'):
    """
    Clean up the input data by filling missing values, resampling, and applying filtering.
    
    Parameters:
    data (pandas.DataFrame): Input data.
    frequency (str): Resampling frequency. Default is 'M' for monthly.
    
    Returns:
    numpy.ndarray: Cleaned and normalized data.
    """
    # Mark all missing values
    data.replace('?', nan, inplace=True)
    # Make dataset numeric
    data = data.astype('float32')
    # Fill missing values
    data = fill_missing(data)
    # Resample data
    resample_groups = data.resample(frequency)
    resample_data = resample_groups.mean()
    # Apply Savitzky-Golay filter with window size 3 and polynomial order 2
    filtered = savgol_filter(resample_data.values, window_length=3, polyorder=2, axis=0)
    # Normalize data
    normalized_data = normalize_data(filtered)
    return normalized_data
 
# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in=1, n_steps_out=1):
    """
    Split a multivariate sequence into input/output samples for a supervised learning problem.
    
    Parameters:
    sequences (numpy.ndarray): Input sequences.
    n_steps_in (int): Number of time steps to use as input.
    n_steps_out (int): Number of time steps to predict as output.
    
    Returns:
    numpy.ndarray: Input sequences.
    numpy.ndarray: Output sequences.
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # Find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
