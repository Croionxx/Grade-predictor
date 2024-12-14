import numpy as np
import joblib
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.metrics import MeanSquaredError # type: ignore
import pandas as pd

def load_model_and_scalers(model_path, feature_scaler_path, target_scaler_path):
    try:
        custom_objects = {"mse": MeanSquaredError()}
        model = load_model(model_path, custom_objects=custom_objects)
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        print("Model and scalers loaded successfully.")
        return model, feature_scaler, target_scaler
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        return None, None, None


def preprocess_user_input(user_input, training_columns):
    """
    Preprocesses user input to match the training feature set.

    Args:
        user_input (dict): Raw user input as a dictionary.
        training_columns (list): List of feature names used during training.

    Returns:
        np.array: Preprocessed and scaled input array.
    """
    user_input_df = pd.DataFrame([user_input])
    user_input_encoded = pd.get_dummies(user_input_df)
    user_input_aligned = user_input_encoded.reindex(columns=training_columns, fill_value=0)
    return user_input_aligned


def predict_g3(model, feature_scaler, target_scaler, user_input, training_columns):
    """
    Predicts the G3 value based on user input.

    Args:
        model: Trained Keras model.
        feature_scaler: Scaler used to transform input features.
        target_scaler: Scaler used to inverse transform the output.
        user_input: Raw user input as a dictionary.
        training_columns: List of feature names used during training.

    Returns:
        float: Predicted G3 value.
    """
    try:
        # Preprocess the input to match training features
        preprocessed_input = preprocess_user_input(user_input, training_columns)

        # Scale the input using the feature scaler
        scaled_input = feature_scaler.transform(preprocessed_input)

        # Predict the scaled output using the model
        scaled_prediction = model.predict(scaled_input)

        # Inverse transform the prediction to get the original G3 value
        predicted_g3 = target_scaler.inverse_transform(scaled_prediction).flatten()[0]
        print(f"\nPredicted G3 value: {predicted_g3:.2f}")
        return predicted_g3
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def main():
    model_path = 'optimized_ann_model.h5'
    feature_scaler_path = 'feature_scaler.pkl'
    target_scaler_path = 'target_scaler.pkl'
    training_columns_path = 'training_columns.pkl'

    # Load the training feature names
    training_columns = joblib.load(training_columns_path)

    # Load the trained model and scalers
    model, feature_scaler, target_scaler = load_model_and_scalers(
        model_path, feature_scaler_path, target_scaler_path
    )
    if model is None or feature_scaler is None or target_scaler is None:
        print("Failed to load the required components. Exiting.")
        return

    # Example user input
    user_input = {
        'age': 18,
        'Medu': 4,
        'Fedu': 4,
        'traveltime': 1,
        'studytime': 2,
        'failures': 0,
        'famrel': 4,
        'freetime': 3,
        'goout': 2,
        'Dalc': 1,
        'Walc': 1,
        'health': 5,
        'absences': 6,
        'school_MS': 0,  # Example of one-hot encoded categorical features
        'sex_M': 1,
        # Add all other required one-hot encoded categorical features...
    }

    # Predict the G3 value
    predict_g3(model, feature_scaler, target_scaler, user_input, training_columns)


if __name__ == "__main__":
    main()
