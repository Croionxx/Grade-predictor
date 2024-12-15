import pandas as pd
import joblib
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.metrics import MeanSquaredError  # type: ignore

def load_and_predict(input_data):
    """
    Loads the pre-trained model and scalers, processes the input data, and returns the predicted G3 value.

    Args:
        input_data (dict): Input feature dictionary for prediction.

    Returns:
        float: Predicted G3 value.
    """
    try:
        # Provide custom objects if necessary
        model = load_model('optimized_ann_model.h5', custom_objects={'mse': MeanSquaredError()})
        print("[INFO] Model loaded successfully.")

        # Load the scalers and feature names
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        training_columns = joblib.load('training_columns.pkl')
        print("[INFO] Scalers and feature names loaded successfully.")
        print("[DEBUG] Training columns loaded:", training_columns)
    except Exception as e:
        print(f"[ERROR] Error loading model or scalers: {e}")
        return None

    try:
        # Convert input_data to DataFrame and align it with training columns
        input_df = pd.DataFrame([input_data])
        print("[DEBUG] User input DataFrame before reindexing:\n", input_df)
        input_df = input_df.reindex(columns=training_columns, fill_value=0)
        print("[DEBUG] User input DataFrame after reindexing:\n", input_df)

        if input_df.isnull().values.any():
            print("[WARNING] Input DataFrame contains NaN values after reindexing.")
        if (input_df == 0).all().all():
            print("[WARNING] Input DataFrame is all zeros after reindexing.")

        input_scaled = feature_scaler.transform(input_df)
        print("[DEBUG] Scaled input data:\n", input_scaled)

        # Predict using the loaded model
        prediction_scaled = model.predict(input_scaled).flatten()
        print("[DEBUG] Scaled prediction:\n", prediction_scaled)

        if (prediction_scaled == prediction_scaled[0]).all():
            print("[WARNING] All elements in prediction_scaled are identical, indicating potential model issue.")

        # Inverse transform the prediction to get the original scale
        prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        print("[INFO] Final prediction (in original scale):", prediction)
        return prediction

    except Exception as e:
        print(f"[ERROR] Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # Example input: Replace with actual feature values in dictionary form
    example_input = {
        'age': 21,
        'Medu': 2,
        'Fedu': 0,
        'traveltime': 1,
        'studytime': 0,
        'failures': 0,
        'famrel': 5,
        'freetime': 4,
        'goout': 3,
        'Dalc': 1,
        'Walc': 1,
        'health': 4,
        'absences': 3,
        'school_MS': 1,
        'sex_M': 1,
        'address_U': 1,
        'famsize_LE3': 0,
        'Pstatus_T': 0,
        'Mjob_health': 0,
        'Mjob_other': 1,
        'Mjob_services': 1,
        'Mjob_teacher': 3,
        'Fjob_health': 4,
        'Fjob_other': 1,
        'Fjob_services': 0,
        'Fjob_teacher': 1,
        'reason_home': 0
    }
    predicted_g3 = load_and_predict(example_input)

    if predicted_g3 is not None:
        print(f"Predicted G3 value: {predicted_g3:.2f}")
    else:
        print("Prediction failed.")
