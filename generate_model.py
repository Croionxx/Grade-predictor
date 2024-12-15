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
import joblib
import keras_tuner as kt  # Import Keras Tuner for hyperparameter tuning

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 1. Data Preprocessing
def preprocess_data(file_path, target):
    data = pd.read_csv(file_path, delimiter=';')
    logging.info("Data successfully loaded.")

    # Split predictors and target
    X = data.drop(columns=[target])
    y = data[target]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Save the correct training column names before transforming X
    training_columns = X.columns.tolist()
    joblib.dump(training_columns, 'training_columns.pkl')
    logging.info(f"Training columns saved successfully: {training_columns}")
    
    # Scale features and target
    feature_scaler = StandardScaler()
    X = feature_scaler.fit_transform(X)

    target_scaler = StandardScaler()
    y = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    logging.info("Data preprocessing complete.")
    return X, y, feature_scaler, target_scaler


# 2. Model Builder for Hyperparameter Tuning
def create_model(hp):
    """
    Creates and returns an ANN model with hyperparameters for tuning.

    Args:
        hp (HyperParameters): Hyperparameters from Keras Tuner.

    Returns:
        model: Compiled Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    # Dynamically add layers and neurons
    for i in range(hp.Int("num_layers", 1, 5)):  # Tune number of layers (1 to 5)
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(hp.Float("l2", 1e-5, 1e-2, sampling="log")),
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))

    # Output layer
    model.add(layers.Dense(1))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")),
        loss='mse',  # Loss for regression
        metrics=['mae']  # Replace rmse with mae or mse
    )
    return model


# 3. Train and Tune the Model
def tune_model(X_train, y_train, X_val, y_val):
    from keras_tuner import Hyperband

    tuner = Hyperband(
        create_model,
        objective="val_loss",  # Minimize validation loss
        max_epochs=50,
        factor=3,
        directory="tuning_results",
        project_name="student_performance",
        overwrite=True,
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        ],
        verbose=1,
    )

    return tuner



# 4. Evaluate Model
def evaluate_model(model, X_test, y_test, target_scaler):
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
    file_path = "student_data/student-mat.csv"
    target = "G3"
    X, y, feature_scaler, target_scaler = preprocess_data(file_path, target)

    # Save scalers and feature names
    joblib.dump(feature_scaler, "feature_scaler.pkl")
    joblib.dump(target_scaler, "target_scaler.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Perform hyperparameter tuning
    input_shape = X_train.shape[1]
    tuner = tune_model(X_train, y_train, X_val, y_val)

    # Get the best hyperparameters and train the final model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(f"Best hyperparameters: {best_hps.values}")

    # Train the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=500, batch_size=64, verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=1e-6)
        ]
    )

    # Save the final model
    model.save("optimized_ann_model.h5")

    # Evaluate the final model
    evaluate_model(model, X_test, y_test, target_scaler)
