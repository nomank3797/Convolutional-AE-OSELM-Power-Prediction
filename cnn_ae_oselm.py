# Import necessary libraries
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten
from tensorflow.keras.layers import Conv1D, UpSampling1D, MaxPooling1D
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyoselm import OSELMRegressor
import numpy as np
import pandas as pd
import pickle
import forecast_evaluation  # This is a custom module for forecast evaluation

# Define a function for building 1D CNN Autoencoder model
def build_cnn_ae_model(input_shape):
    """
    Build a 1D CNN Autoencoder model.
    
    Parameters:
    input_shape (tuple): Shape of the input data.
    
    Returns:
    Model: 1D CNN Autoencoder model.
    """
    # Define encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = Conv1D(filters=512, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(encoder_inputs)
    encoder = MaxPooling1D(pool_size=2, padding='same')(encoder)
    encoder = Conv1D(filters=256, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(encoder)
    encoder = MaxPooling1D(pool_size=2, padding='same')(encoder)
    encoder = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(encoder)
    encoder = MaxPooling1D(pool_size=2, padding='same')(encoder)
    encoder = Flatten()(encoder)   
    
    # Bottleneck layer
    encoder = Dense(100, activation='relu', kernel_initializer='he_uniform')(encoder)
    
    # Define decoder
    decoder = Reshape((10,10))(encoder)
    decoder = Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(decoder)
    decoder = UpSampling1D(size=2)(decoder)
    decoder = Conv1D(filters=256, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(decoder)
    decoder = UpSampling1D(size=2)(decoder)
    decoder = Conv1D(filters=512, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(decoder)
    decoder = UpSampling1D(size=2)(decoder)
    decoder_outputs = Conv1D(filters=1, kernel_size=7, activation='relu', padding='same', kernel_initializer='he_uniform')(decoder)
    
    # Create 1D CNN Autoencoder model
    cnn_ae_model = Model(encoder_inputs, decoder_outputs)

    # Compile the 1D CNN Autoencoder model
    opt = keras.optimizers.Adam(learning_rate=0.001)
    cnn_ae_model.compile(loss='mean_squared_error', optimizer=opt)
    
    return cnn_ae_model

# Define a function for training the model sequentially
def fit_sequential(model, X, y, n_hidden, chunk_size=1):
    """
    Train the model sequentially using chunk-wise data.
    
    Parameters:
    model (object): Model to be trained.
    X (array-like): Input data.
    y (array-like): Target data.
    n_hidden (int): Number of hidden units.
    chunk_size (int): Size of data chunks.
    
    Returns:
    object: Trained model.
    """
    N = len(y)
    # The first batch of data must have the same size as n_hidden to achieve the first phase (boosting)
    batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
    batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]

    for b_x, b_y in zip(batches_x, batches_y):
        if isinstance(model, OSELMRegressor):
            model.fit(b_x, b_y)
        else:
            model.partial_fit(b_x, b_y)
    
    return model

# Define a function for training and testing the models
def cnn_ae_oselm_model(X, y, epochs=1, file_name='model_prediction.csv'):
    """
    Train and evaluate the CNN Autoencoder + OSELM model.
    
    Parameters:
    X (array-like): Input data.
    y (array-like): Target data.
    epochs (int): Number of epochs for training the CNN Autoencoder.
    file_name (str): Name of the file to save predictions.
    """
    batches = X.shape[0]
    timesteps = X.shape[1]
    features = X.shape[2]
    
    # Reshape the input data if needed
    X = X.reshape(batches, timesteps, features)
    
    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    
    # Define CNN Autoencoder input shape
    input_shape = x_train.shape[1:]
    
    # Build 1D CNN Autoencoder model
    cnn_ae_model = build_cnn_ae_model(input_shape)
    
    # Fit CNN Autoencoder model on the training data
    print('[INFO]---|| *** Training CNN Autoencoder Model...\n')
    cnn_ae_model.fit(x_train, x_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test), verbose=0)
    print('[INFO]---|| *** CNN Autoencoder Model Trained!\n')
    
    # Define CNN encoder model without the decoder
    cnn_e_model = Model(inputs=cnn_ae_model.inputs, outputs=cnn_ae_model.layers[8].output)
    
    # Save the CNN encoder
    print('[INFO]---|| *** Saving the CNN Encoder Model...\n')
    cnn_e_model.save('Models/cnn_e_model.h5')
    print('[INFO]---|| *** CNN Encoder Model Saved!\n')
    
    # Extract features using the CNN encoder for training
    cnn_e_model_features = cnn_e_model.predict(x_train, verbose=0)

    # Define the hidden units and chunk size for OSELM model
    n_hidden = 100
    chunk_sizes = [10]
    oselm = OSELMRegressor(n_hidden=n_hidden, activation_func='sigmoid', random_state=123)
    
    print('[INFO]---|| *** Training the OSELM Model...\n')
    for chunk_size in chunk_sizes:
        oselm = fit_sequential(oselm, cnn_e_model_features, y_train, n_hidden, chunk_size)
    print('[INFO]---|| *** OSELM Model Trained!\n')
    
    # Save the model to a file
    print('[INFO]---|| *** Saving the OSELM Model...\n')
    with open('Models/oselm.pkl', 'wb') as f:
        pickle.dump(oselm, f)
    print('[INFO]---|| *** OSELM Model Saved!\n')

    # Extract features using the CNN encoder for testing
    cnn_e_model_features = cnn_e_model.predict(x_test, verbose=0)

    print('[INFO]---|| *** Testing the OSELM Model...\n')    
    yhat = oselm.predict(cnn_e_model_features)
    print('[INFO]---|| *** OSELM Model Testing Completed!\n')

    # Saving predictions to a CSV file
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': yhat.flatten()})
    df.to_csv(file_name, index=False)
    print("CSV file '{}' created successfully.".format(file_name))

    # Evaluating model predictions
    forecast_evaluation.evaluate_forecasts(y_test, yhat)
