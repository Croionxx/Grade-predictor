import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 1. Data Preprocessing
def preprocess_data(file_path,target):
    """
    Loads and preprocesses the dataset, including scaling.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        X (np.array): Preprocessed features.
        y (np.array): Scaled target variable.
        feature_scaler (StandardScaler): Scaler object for features.
        target_scaler (StandardScaler): Scaler object for target variable.
    """
    try:
        data = pd.read_csv(file_path, delimiter=';')
        logging.info("Data successfully loaded.")
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}. Please check the path.")
        return None, None, None, None

    # Split predictors and target
    X = data.drop(columns=[target])  # Features
    y = data[target]  # Target variable

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Scale features
    feature_scaler = StandardScaler()
    X = feature_scaler.fit_transform(X)

    # Scale target
    target_scaler = StandardScaler()
    y = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    logging.info("Data preprocessing complete.")
    return X, y, feature_scaler, target_scaler

# 2. Model Creation with Variable Neuron Configuration
def create_model(input_shape, neuron_config):
    """
    Creates and returns an ANN model with a variable number of neurons in each layer.

    Args:
        input_shape (int): The number of input features.
        neuron_config (list of int): Number of neurons for each hidden layer.

    Returns:
        model (keras.Sequential): Compiled Keras Sequential model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    # Add hidden layers dynamically based on neuron_config
    for i, neurons in enumerate(neuron_config):
        model.add(layers.Dense(
            neurons,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(1e-4)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))  # Dropout for regularization

    # Output layer
    model.add(layers.Dense(1))  # Regression output (1 neuron)

    logging.info("Model created successfully with variable neurons.")
    return model

# 3. Train Model
def train_model(X_train, y_train, X_val, y_val, neuron_config, learning_rate=0.001, batch_size=64, epochs=500):
    """
    Trains the model with the given data.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        neuron_config (list of int): Number of neurons for each hidden layer.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.

    Returns:
        model: Trained Keras model.
        history: Training history.
    """
    model = create_model(X_train.shape[1], neuron_config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Mean Squared Error for regression
        metrics=[keras.metrics.RootMeanSquaredError(name='rmse')]  # RMSE as a metric
    )

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=100, restore_best_weights=True, verbose=1
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=50, min_lr=1e-6, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    logging.info("Model training complete.")
    return model, history

# 4. Visualize Training History
def plot_training_history(history):
    """
    Plots the training and validation loss and RMSE over epochs.

    Args:
        history: Training history object.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['rmse'], label='Training RMSE')
    plt.plot(history.history['val_rmse'], label='Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training History')
    plt.show()

# 5. Evaluate Model
def evaluate_model(model, X_test, y_test, target_scaler):
    """
    Evaluates the model on test data and calculates performance metrics.

    Args:
        model: Trained model.
        X_test, y_test: Test data.
        target_scaler: Scaler object for target variable.

    Returns:
        None
    """
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2: {r2:.4f}")

# Main Execution
if __name__ == "__main__":
    file_path = 'student_data/student-mat.csv'
    target = 'G3'
    X, y, feature_scaler, target_scaler = preprocess_data(file_path,target)
    if X is None or y is None:
        exit()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define the neuron configuration for the hidden layers
    neuron_config = [256, 512, 256, 128, 64]

    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val, neuron_config, epochs=2000)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, target_scaler)
