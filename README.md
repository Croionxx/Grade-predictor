# README

## **Project Title: Student Grade Prediction Using an Artificial Neural Network (ANN)**

This project involves two key Python scripts designed to predict student final grades (G3) based on a range of student-related features. The system employs an Artificial Neural Network (ANN) for regression tasks. The project is divided into two main components:

1. **Model Training and Generation Script** (`generate_model.py`)
2. **Model Loading and Prediction Script** (`load_model_predict.py`)

---

## **1. Model Training and Generation (`generate_model.py`)

This script builds, trains, and saves an ANN model to predict the final student grade (G3). It processes raw data, scales features, and saves both the trained model and the scalers for later use.

### **How it Works**
1. **Data Loading and Preprocessing**:
   - Loads the dataset from a CSV file.
   - Splits data into features (`X`) and target (`y`), where `y` is the final grade (G3).
   - One-hot encodes categorical features.
   - Scales features and the target using `StandardScaler`.

2. **Model Creation**:
   - Builds an ANN with a configurable number of hidden layers and neurons.
   - Applies batch normalization, dropout, and L2 regularization to avoid overfitting.

3. **Training**:
   - Trains the ANN using mean squared error (MSE) as the loss function and root mean squared error (RMSE) as a metric.
   - Early stopping and learning rate reduction callbacks ensure optimal training.

4. **Saving the Model and Scalers**:
   - Saves the trained Keras model as `model.h5`.
   - Saves the feature and target scalers as `feature_scaler.pkl` and `target_scaler.pkl`.

### **Key Functions**
- `preprocess_data(file_path, target)`: Loads and preprocesses the data.
- `create_model(input_shape, neuron_config)`: Builds the ANN model with a configurable architecture.
- `train_model(X_train, y_train, X_val, y_val, neuron_config, ...)`: Trains the model and saves the trained model and scalers.
- `plot_training_history(history)`: Plots loss and RMSE for training and validation.

### **Usage**
1. Place your CSV dataset in the specified path.
2. Run the script:
   ```bash
   python generate_model.py
   ```
3. The following files will be saved:
   - **`model.h5`**: The trained model.
   - **`feature_scaler.pkl`**: The scaler for feature inputs.
   - **`target_scaler.pkl`**: The scaler for target outputs.

### **Example**
If the dataset is in `student_data/student-mat.csv`, the script will train an ANN to predict G3 using the provided features.

---

## **2. Model Loading and Prediction (`load_model_predict.py`)

This script loads the saved ANN model, takes custom input values for features, and predicts the final grade (G3).

### **How it Works**
1. **Loading the Model and Scalers**:
   - Loads the previously trained ANN model (`model.h5`).
   - Loads the feature scaler (`feature_scaler.pkl`) and target scaler (`target_scaler.pkl`).

2. **Input Handling**:
   - Accepts a pre-defined **NumPy array of feature inputs**.
   - The user can modify the array to test different feature values.

3. **Prediction**:
   - Scales the input features using the feature scaler.
   - Predicts the scaled output using the model.
   - Inversely scales the predicted G3 to obtain the actual predicted value.

### **Key Functions**
- `load_saved_model()`: Loads the saved model, feature scaler, and target scaler.
- `predict_g3(model, feature_scaler, target_scaler, user_input)`: Predicts the G3 grade for the input features.

### **Usage**
1. Ensure the `model.h5`, `feature_scaler.pkl`, and `target_scaler.pkl` files are present in the script's directory.
2. Modify the `user_input` array in the script to test with custom feature values.
3. Run the script:
   ```bash
   python load_model_predict.py
   ```

### **Example**
If you want to predict G3 for the following student features:
```python
user_input = np.array([[18, 4, 4, 1, 2, 0, 4, 3, 2, 1, 1, 3, 5]])
```
The script will output the predicted G3 value for this student.

---

## **File Structure**
```
project_directory/
  ├── generate_model.py         # Script to train and save the model and scalers
  ├── load_model_predict.py     # Script to load the model and make predictions
  ├── model.h5                  # Saved model file (after running generate_model.py)
  ├── feature_scaler.pkl        # Saved feature scaler (after running generate_model.py)
  ├── target_scaler.pkl         # Saved target scaler (after running generate_model.py)
  └── student_data/
      └── student-mat.csv       # Dataset file (not included, must be provided)
```

---

## **Required Libraries**
Make sure you have the following libraries installed before running the scripts:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow joblib
```

---

## **Results**
### Model Performance Metrics:
- **Test RMSE**: 2.5014
- **Test MAE**: 1.7590
- **Test R2**: 0.6949

### Training Details:
- **Restoring model weights from the end of the best epoch**: 1181.

### Training History Plot:
![Training History]([training_history_plot.png.](https://github.com/Croionxx/Grade-predictor/blob/f64022c02f2e1e55aa5ecfc52def98775ac9bffd/training_history_plot.png))

---

## **Dataset Information**
This project uses the following dataset:
- **Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository.**
  [https://doi.org/10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)

This dataset contains information on student demographics, social life, and academic factors that influence their final grade (G3).

---

## **Customization**
You can modify the following aspects to suit your needs:
- **Data Preprocessing**: Adjust the target variable or add more features.
- **ANN Architecture**: Modify the number of layers, neurons, dropout, and activation functions in the `create_model()` function.
- **Training Parameters**: Adjust epochs, batch size, and callbacks.

---

## **Possible Improvements**
- **Hyperparameter Tuning**: Adjust the neuron configuration and learning rate.
- **Feature Selection**: Select the most relevant features to improve model performance.
- **More User-Friendly Input**: Add a user interface for input instead of modifying the script.

---

## **Contact**
Please contact me for any queries and improvements, it will be a delight to hear form you.

Hope you have a blissfull coding session.

