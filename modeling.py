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

