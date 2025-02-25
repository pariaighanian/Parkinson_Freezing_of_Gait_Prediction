import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_multimodal_model(input_shapes):
    """
    Creates a multi-branch deep learning model that combines:
    - LSTM for raw time-series data
    - Dense layers for FFT, time-domain, and metadata features
    """
    # === LSTM Branch (Processes Raw Time-Series Data) ===
    input_raw = tf.keras.Input(shape=input_shapes["raw"])  
    x_raw = layers.LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(input_raw)
    x_raw = layers.BatchNormalization()(x_raw)
    x_raw = layers.Dropout(0.3)(x_raw)  
    x_raw = layers.LSTM(16, kernel_regularizer=regularizers.l2(0.01))(x_raw)
    x_raw = layers.BatchNormalization()(x_raw)
    x_raw = layers.Dropout(0.3)(x_raw)  

    # === FFT Feature Branch ===
    input_fft = tf.keras.Input(shape=input_shapes["fft"])  
    x_fft = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_fft)
    x_fft = layers.BatchNormalization()(x_fft)
    x_fft = layers.Dropout(0.3)(x_fft)  

    # === Time-Domain Feature Branch ===
    input_time = tf.keras.Input(shape=input_shapes["time"])  
    x_time = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_time)
    x_time = layers.BatchNormalization()(x_time)
    x_time = layers.Dropout(0.3)(x_time)  

    # === Metadata Branch (Age, Gender, UPDRS_III) ===
    input_meta = tf.keras.Input(shape=input_shapes["meta"])  
    x_meta = layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_meta)
    x_meta = layers.BatchNormalization()(x_meta)
    x_meta = layers.Dropout(0.3)(x_meta)

    # === Merge All Features ===
    merged = layers.concatenate([x_raw, x_fft, x_time, x_meta])
    merged = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.Dropout(0.3)(merged)  

    # === Output Layer ===
    output = layers.Dense(1, activation='sigmoid')(merged)  # Binary classification

    # === Create Model ===
    model = models.Model(inputs=[input_raw, input_fft, input_time, input_meta], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_and_evaluate_model(model, task_name, data_dict, test_patients):
    """
    Trains and evaluates the model for a given dataset.
    """
    print(f"\nTraining model for {task_name} dataset...")

    # Prepare input data for training
    X_train = [data_dict["X_train_raw"], data_dict["X_train_fft"], data_dict["X_train_time"], data_dict["X_train_meta"]]
    y_train = data_dict["labels_train"]

    X_val = [data_dict["X_val_raw"], data_dict["X_val_fft"], data_dict["X_val_time"], data_dict["X_val_meta"]]
    y_val = data_dict["labels_val"]

    X_test = [data_dict["X_test_raw"], data_dict["X_test_fft"], data_dict["X_test_time"], data_dict["X_test_meta"]]
    y_test = data_dict["labels_test"]

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Predict on test set (window-level predictions)
    y_pred_probs = model.predict(X_test)
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

    df_results = pd.DataFrame({'SubID': data_dict["sub_ids_test"], 'Prediction': y_pred_labels.flatten()})

    # Aggregate window-level predictions into patient-level predictions
    window_counts = df_results.groupby('SubID')['Prediction'].value_counts().unstack(fill_value=0)
    window_counts.columns = ['Predicted_0s', 'Predicted_1s']

    # Use mean probability to make a final decision per patient
    final_patient_preds = df_results.groupby('SubID')['Prediction'].agg(lambda x: round(x.mean()))

    # Merge with true patient labels
    patient_labels_test = pd.DataFrame({'SubID': test_patients['SubID'], 'True_Label': test_patients['Binary_Diagnosis']})
    df_final_results = final_patient_preds.reset_index().merge(patient_labels_test, on='SubID')
    df_final_results = df_final_results.merge(window_counts, on='SubID')

    # Compute patient-level accuracy
    patient_level_acc = (df_final_results['Prediction'] == df_final_results['True_Label']).mean()

    # Print evaluation results
    print(f"\n {task_name} - Patient-Level Accuracy: {patient_level_acc:.4f}")
    print("\n\033[1mWindow-Level Predictions Per Patient:\033[0m")
    print(df_final_results[['SubID', 'True_Label', 'Predicted_0s', 'Predicted_1s']])

    print("\n\033[1mClassification Report (Patient-Level):\033[0m")
    print(classification_report(df_final_results['True_Label'], df_final_results['Prediction'], target_names=["Control", "Parkinson's"]))

    cm = confusion_matrix(df_final_results['True_Label'], df_final_results['Prediction'])
    print("\n\033[1mConfusion Matrix (Patient-Level):\033[0m")
    print(cm)

    return {
        "patient_accuracy": patient_level_acc,
        "classification_report": classification_report(df_final_results['True_Label'], df_final_results['Prediction'], output_dict=True),
        "confusion_matrix": cm
    }
