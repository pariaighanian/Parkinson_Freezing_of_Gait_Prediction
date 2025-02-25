from data_processing import load_patient_data, map_diagnosis, normalize_per_patient, create_fixed_windows_with_overlap
from feature_extraction import apply_fft_to_windows
import pandas as pd
from sklearn.model_selection import train_test_split


def main(task_name="TUG"):
    """Main function to process a given task."""
    root_directory = "G:/My Drive/fog_dataset/"
    print(f"Loading data for task: {task_name}...")

    meta_df, sensor_df = load_patient_data(root_directory, task_name)

    if sensor_df.empty:
        print(f"⚠️ No data found for task: {task_name}")
        return

    print("✅ Data loaded successfully!")

    # Apply the mapping inside this script
    meta_df['Binary_Diagnosis'] = meta_df['Diagnosis'].map(map_diagnosis)

    # Ensure only valid SubIDs are mapped
    sensor_df = sensor_df[sensor_df['SubID'].isin(meta_df['SubID'])]

    # Map Binary_Diagnosis and ensure no NaN values
    sensor_df['Binary_Diagnosis'] = sensor_df['SubID'].map(meta_df.set_index('SubID')['Binary_Diagnosis'])

    # Debugging output
    missing_diagnosis_count = sensor_df['Binary_Diagnosis'].isna().sum()
    print(f"⚠️ Missing values in Binary_Diagnosis: {missing_diagnosis_count}")

    if missing_diagnosis_count > 0:
        raise ValueError("❌ Found NaN values in Binary_Diagnosis. Check data consistency.")

    # Normalize sensor data
    sensor_columns = [col for col in sensor_df.columns if col.startswith(('L_', 'R_'))]
    sensor_df = normalize_per_patient(sensor_df, sensor_columns)

    # Split into train, validation, and test sets
    train_patients, test_patients = train_test_split(
        sensor_df['SubID'].unique(), test_size=0.2, random_state=42,
        stratify=sensor_df.groupby('SubID')['Binary_Diagnosis'].first()
    )
    train_patients, val_patients = train_test_split(
        train_patients, test_size=0.2, random_state=42,
        stratify=sensor_df[sensor_df['SubID'].isin(train_patients)]['Binary_Diagnosis']
    )

    train_data = sensor_df[sensor_df['SubID'].isin(train_patients)]
    val_data = sensor_df[sensor_df['SubID'].isin(val_patients)]
    test_data = sensor_df[sensor_df['SubID'].isin(test_patients)]

    print("✅ Data successfully split into Train, Validation, and Test sets.")
    print(f"Train patients: {len(train_patients)} | Validation patients: {len(val_patients)} | Test patients: {len(test_patients)}")

    # Create windows
    window_size, overlap = 300, 150
    train_windows, train_labels, _ = create_fixed_windows_with_overlap(train_data, sensor_columns, window_size, overlap)
    val_windows, val_labels, _ = create_fixed_windows_with_overlap(val_data, sensor_columns, window_size, overlap)
    test_windows, test_labels, _ = create_fixed_windows_with_overlap(test_data, sensor_columns, window_size, overlap)

    # Apply FFT transformation
    fft_train, freq_bins = apply_fft_to_windows(train_windows)
    fft_val, _ = apply_fft_to_windows(val_windows)
    fft_test, _ = apply_fft_to_windows(test_windows)

    print("✅ Preprocessing complete!")
    print(f"FFT Train Shape: {fft_train.shape}")
    print(f"FFT Validation Shape: {fft_val.shape}")
    print(f"FFT Test Shape: {fft_test.shape}")

if __name__ == "__main__":
    main("TUG")
