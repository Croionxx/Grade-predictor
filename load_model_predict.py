import numpy as np
import pandas as pd
import joblib  # To load saved scalers
from tensorflow import keras  # To load the saved model

# Function to load the trained model and scalers
def load_saved_model(model_path='model.h5', feature_scaler_path='feature_scaler.pkl', target_scaler_path='target_scaler.pkl'):
    """
    Loads the saved model and the associated scalers.

    Args:
        model_path (str): Path to the saved Keras model file.
        feature_scaler_path (str): Path to the saved feature scaler.
        target_scaler_path (str): Path to the saved target scaler.
    
    Returns:
        model (keras.Model): The loaded Keras model.
        feature_scaler (StandardScaler): Scaler for feature inputs.
        target_scaler (StandardScaler): Scaler for target outputs.
    """
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None
    
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Feature and target scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return model, None, None
    
    return model, feature_scaler, target_scaler


def predict_g3(model, feature_scaler, target_scaler, user_input):
    """
    Predicts the G3 value based on user input.

    Args:
        model (keras.Model): The trained Keras model.
        feature_scaler (StandardScaler): The scaler used to transform input features.
        target_scaler (StandardScaler): The scaler used to inverse transform the output.
        user_input (np.array): The user's input values.
    
    Returns:
        predicted_g3 (float): The predicted G3 value.
    """
    try:
        # Scale the user input using the feature scaler
        scaled_input = feature_scaler.transform(user_input)
        
        # Predict the scaled output using the model
        scaled_prediction = model.predict(scaled_input)
        
        # Inverse transform the prediction to get the original G3 value
        predicted_g3 = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).flatten()[0]
        print(f"\nPredicted G3 value: {predicted_g3:.2f}")
        return predicted_g3
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def main():
    """
    Main function to load the model, prompt user for inputs, and predict the G3 value.
    """
    # Load the trained model and scalers
    model, feature_scaler, target_scaler = load_saved_model()
    if model is None or feature_scaler is None or target_scaler is None:
        print("Error loading model or scalers. Exiting.")
        return
    
    # Example of feature names (replace with actual feature names from your dataset)
    feature_names = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']  # Add more if necessary
    
    # Example user input as a list (replace with actual inputs when needed)
    user_input = np.array([[18, 4, 4, 1, 2, 0, 4, 3, 2, 1, 1, 5, 6]])  # Example input for all features
    
    # Predict the G3 value
    predict_g3(model, feature_scaler, target_scaler, user_input)


if __name__ == "__main__":
    main()
