import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess the data
file_path = 'student_data/student-mat.csv'  # Update this to the correct path
data = pd.read_csv(file_path, delimiter=';')

# Include G1 and G2 back as predictors
X = data.drop(columns=['G3'])
y = data['G3']

# Encode categorical features and normalize numerical features
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Normalize the features

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an optimized ANN model
def create_optimized_model():
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer with input size as number of features
        
        layers.Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-4)),
        layers.BatchNormalization(),  # Normalize activations
        layers.Dropout(0.1),  # Reduced Dropout to prevent underfitting
        
        layers.Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-4)),
        layers.BatchNormalization(),  # Normalize activations
        layers.Dropout(0.1),  # Reduced Dropout to prevent underfitting
        
        layers.Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-4)),
        layers.BatchNormalization(),  # Normalize activations
        layers.Dropout(0.1),  # Reduced Dropout to prevent underfitting
        
        layers.Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-4)),
        layers.BatchNormalization(),  # Normalize activations
        layers.Dropout(0.1),  # Reduced Dropout to prevent underfitting
        
        layers.Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-4)),
        layers.BatchNormalization(),  # Normalize activations
        
        layers.Dense(1)  # Output layer with 1 unit for regression output
    ])
    return model

# Compile the model
model = create_optimized_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='mae',  # Use MAE as the main loss
              metrics=[keras.metrics.RootMeanSquaredError(name='rmse'), 'mae'])

# Add callbacks for learning rate scheduling and early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=500,  # More epochs to ensure better convergence
                    batch_size=64,  # Larger batch size for more stable updates
                    validation_split=0.2, 
                    callbacks=[early_stopping, reduce_lr], 
                    verbose=1)

# Evaluate the model on the test set
y_pred = model.predict(X_test, verbose=0).flatten()

# Calculate performance metrics
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2: {test_r2:.4f}")

# Function to make predictions for a new input
def predict_new_input(model, input_features):
    input_tensor = np.array(input_features).reshape(1, -1)  # Reshape input to have batch dimension
    prediction = model.predict(input_tensor, verbose=0)
    return prediction[0][0]

# Example usage
example_input = X[0]  # Use the first sample from the dataset as input
predicted_G3 = predict_new_input(model, example_input)
print(f"Predicted G3 for input: {predicted_G3}")
