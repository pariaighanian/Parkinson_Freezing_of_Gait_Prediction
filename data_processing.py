import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_sensor_data(root_directory, task_name):
    """
    Loads sensor data for a specific task.
    """
    all_sensor_data = []
    selected_columns = ["Counter", "AccX_filt", "AccY_filt", "AccZ_filt", "GyrX_filt", "GyrY_filt", "GyrZ_filt"]

    patient_folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f)) and f.startswith("ND")]
    
    for patient_folder in tqdm(patient_folders, desc=f"Loading Sensor Data for {task_name}"):
        patient_path = os.path.join(root_directory, patient_folder)

        for version_folder in os.listdir(patient_path):
            task_path = os.path.join(patient_path, version_folder, task_name)
            if os.path.exists(task_path):
                sensor_left_path = os.path.join(task_path, "_SensorDataLeft.json")
                sensor_right_path = os.path.join(task_path, "_SensorDataRight.json")

                if os.path.exists(sensor_left_path) and os.path.exists(sensor_right_path):
                    with open(sensor_left_path, "r") as file:
                        left_data = json.load(file)
                    with open(sensor_right_path, "r") as file:
                        right_data = json.load(file)

                    left_df = pd.DataFrame(left_data)
                    right_df = pd.DataFrame(right_data)
                    
                    left_df = left_df[[col for col in left_df.columns if col in selected_columns or col == "Counter"]]
                    right_df = right_df[[col for col in right_df.columns if col in selected_columns or col == "Counter"]]
                    
                    left_df = left_df.add_prefix("L_")
                    right_df = right_df.add_prefix("R_")
                    
                    patient_data = pd.merge(left_df, right_df, left_on="L_Counter", right_on="R_Counter", how="inner")
                    
                    patient_data = patient_data.drop(columns=["R_Counter"]).rename(columns={"L_Counter": "Counter"})
                    
                    sub_id = f"{patient_folder}_{version_folder}"
                    patient_data.insert(0, "SubID", sub_id)
                    
                    all_sensor_data.append(patient_data)

    sensor_df = pd.concat(all_sensor_data, ignore_index=True) if all_sensor_data else pd.DataFrame()

    return sensor_df

def load_meta_data(root_directory, task_name):
    """
    Loads patients metadata for a specific task.    
    """
    meta_data = []

    patient_folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f)) and f.startswith("ND")]

    for patient_folder in tqdm(patient_folders, desc=f"Loading Metadata for {task_name}"):
        patient_path = os.path.join(root_directory, patient_folder)

        for version_folder in os.listdir(patient_path):
            version_path = os.path.join(patient_path, version_folder, task_name)

            if os.path.exists(version_path):
                pat_info_path = os.path.join(version_path, "pat_info.json")

                if os.path.exists(pat_info_path):
                    with open(pat_info_path, "r") as file:
                        pat_info = json.load(file)

                    patient_meta = {key: pat_info[key]["0"] for key in pat_info.keys()}

                    meta_data.append(patient_meta)

    meta_df = pd.DataFrame(meta_data) if meta_data else pd.DataFrame()

    if not meta_df.empty:
        meta_df["SubID_Session"] = meta_df["SubID"] + "_" + meta_df["Session"]

    return meta_df


def read_sensor_data(root_directory, task_name):
    """
    Reads sensor data for a specific task from a CSV file.
    """
    diagnosis_dir = os.path.join(root_directory, "diagnosis")
    sensor_file = os.path.join(diagnosis_dir, f"sensor_{task_name}.csv")

    if os.path.exists(sensor_file):
        print(f"Reading sensor data for {task_name}...")
        return pd.read_csv(sensor_file)
    else:
        print(f"Sensor data file not found for {task_name}.")
        return pd.DataFrame()


def read_meta_data(root_directory, task_name):
    """
    Reads metadata for a specific task from a CSV file.
    """
    diagnosis_dir = os.path.join(root_directory, "diagnosis")
    meta_file = os.path.join(diagnosis_dir, f"meta_{task_name}.csv")

    if os.path.exists(meta_file):
        print(f"Reading metadata for {task_name}...")
        return pd.read_csv(meta_file)
    else:
        print(f"Metadata file not found for {task_name}.")
        return pd.DataFrame()

def add_diagnosis_to_sensor(sensor_df, meta_df):
    """
    Merges sensor data with metadata to add the Diagnosis column based on SubID_Session.
    """

    if sensor_df.empty or meta_df.empty:
        print("One of the input DataFrames is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    merged_df = sensor_df.merge(meta_df[["SubID_Session", "Diagnosis"]], 
                                left_on="SubID", right_on="SubID_Session", how="left")

    merged_df = merged_df.drop(columns=["SubID_Session"])
    
    return merged_df

def filter_and_label_diagnosis(sensor_df, diagnosis_to_keep, label_1_diagnoses, label_0_diagnoses):
    """
    Filters sensor data based on the provided diagnosis list and dynamically maps 
    diagnoses to binary labels.
    """

    if "Diagnosis" not in sensor_df.columns:
        print("No 'Diagnosis' column found in the dataset. Returning original DataFrame.")
        return sensor_df

    filtered_df = sensor_df[sensor_df["Diagnosis"].isin(diagnosis_to_keep)].reset_index(drop=True)

    diagnosis_mapping = {diag: 1 for diag in label_1_diagnoses}
    diagnosis_mapping.update({diag: 0 for diag in label_0_diagnoses})

    filtered_df["Diagnosis"] = filtered_df["Diagnosis"].map(diagnosis_mapping)
    
    return filtered_df

def add_feature_to_sensor(sensor_df, meta_df, feature_name):
    """
    Adds a specified feature from metadata to sensor data, merging based on SubID_Session.
    """

    if sensor_df.empty or meta_df.empty:
        print(f"One of the input DataFrames is empty. Cannot add feature {feature_name}.")
        return sensor_df

    if feature_name not in meta_df.columns:
        print(f"Feature '{feature_name}' not found in metadata. Returning original DataFrame.")
        return sensor_df

    merged_df = sensor_df.merge(meta_df[["SubID_Session", feature_name]], 
                                left_on="SubID", right_on="SubID_Session", how="left")

    merged_df = merged_df.drop(columns=["SubID_Session"])
    
    return merged_df

def convert_yob_to_age(sensor_df):
    """
    Converts Year of Birth (YOB) to Age.
    """

    if "YOB" not in sensor_df.columns:
        print("'YOB' column not found in dataset. Skipping age calculation.")
        return sensor_df

    current_year = datetime.now().year

    sensor_df["YOB"] = pd.to_numeric(sensor_df["YOB"], errors="coerce")

    sensor_df["Age"] = current_year - sensor_df["YOB"]
    
    return sensor_df

def impute_age_by_patient_avg(sensor_df):
    """
    Imputes missing Age values using the average Age of unique patients.
    """

    if "Age" not in sensor_df.columns:
        print("'Age' column not found. Cannot impute missing values.")
        return sensor_df

    unique_patients = sensor_df.drop_duplicates(subset=["SubID"])

    avg_age = int(round(unique_patients["Age"].mean()))

    sensor_df["Age"] = sensor_df["Age"].fillna(avg_age)
    
    return sensor_df


def binary_gender_encode(sensor_df):
    """
    Encodes Gender column from categorical (M/F) to binary (0/1).
    """

    if "Gender" not in sensor_df.columns:
        print("'Gender' column not found in dataset. Skipping conversion.")
        return sensor_df

    sensor_df["Gender"] = sensor_df["Gender"].map({"M": 0, "F": 1})

    if sensor_df["Gender"].isna().sum() > 0:
        print(f"Found {sensor_df['Gender'].isna().sum()} missing values in Gender. Consider handling them.")
        
    return sensor_df


def impute_updrs3_by_knn(sensor_df):
    """
    Imputes missing UPDRS3 values using KNN Imputation based on unique SubIDs separately for Control and Parkinson's patients.
    """

    if "UPDRS3" not in sensor_df.columns or "Diagnosis" not in sensor_df.columns:
        print("Required columns ('UPDRS3', 'Diagnosis') not found. Skipping imputation.")
        return sensor_df

    if sensor_df["UPDRS3"].isna().sum() == 0:
        print("No missing UPDRS3 values. Skipping imputation.")
        return sensor_df

    features = ["Age", "Gender"]

    unique_patients = sensor_df.drop_duplicates(subset=["SubID"])[["SubID", "Diagnosis", "UPDRS3"] + features].copy()

    control_df = unique_patients[unique_patients["Diagnosis"] == 0].copy()
    parkinson_df = unique_patients[unique_patients["Diagnosis"] == 1].copy()

    knn_imputer = KNNImputer(n_neighbors=8) 

    if control_df["UPDRS3"].isna().sum() > 0:
        control_df["UPDRS3"] = knn_imputer.fit_transform(control_df[["UPDRS3"] + features])[:, 0].round()

    if parkinson_df["UPDRS3"].isna().sum() > 0:
        parkinson_df["UPDRS3"] = knn_imputer.fit_transform(parkinson_df[["UPDRS3"] + features])[:, 0].round()

    imputed_values = pd.concat([control_df, parkinson_df])[["SubID", "UPDRS3"]]
    sensor_df = sensor_df.merge(imputed_values, on="SubID", how="left", suffixes=("", "_imputed"))

    sensor_df["UPDRS3"] = sensor_df["UPDRS3_imputed"].combine_first(sensor_df["UPDRS3"])

    sensor_df.drop(columns=["UPDRS3_imputed"], inplace=True)

    return sensor_df


def split_dataset(df, dataset_name="Dataset", save_path=None, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits a dataset into Train, Validation, and Test sets with label-wise stratification.
    """

    if save_path:
        train_file = f"{save_path}_train.csv"
        val_file = f"{save_path}_val.csv"
        test_file = f"{save_path}_test.csv"

        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            print(f"\nFound existing split files for {dataset_name}. Loading data...")
            train_data = pd.read_csv(train_file)
            val_data = pd.read_csv(val_file)
            test_data = pd.read_csv(test_file)

        else:
            print(f"\nSplitting dataset: {dataset_name}...")

            patient_labels = df.groupby('SubID')['Diagnosis'].first().reset_index()

            train_patients, test_patients = train_test_split(
                patient_labels, test_size=test_size, random_state=random_state, stratify=patient_labels['Diagnosis']
            )

            train_patients, val_patients = train_test_split(
                train_patients, test_size=val_size, random_state=random_state, stratify=train_patients['Diagnosis']
            )

            train_data = df[df['SubID'].isin(train_patients['SubID'])]
            val_data = df[df['SubID'].isin(val_patients['SubID'])]
            test_data = df[df['SubID'].isin(test_patients['SubID'])]

            train_data.to_csv(train_file, index=False)
            val_data.to_csv(val_file, index=False)
            test_data.to_csv(test_file, index=False)

    print(f"Dataset Split Summary for {dataset_name}:")
    print(f"  - Total Patients: {df.SubID.nunique()} (Original)")
    print(f"  - Train Patients: {train_data.SubID.nunique()}")
    print(f"  - Validation Patients: {val_data.SubID.nunique()}")
    print(f"  - Test Patients: {test_data.SubID.nunique()}")

    return {"train": train_data, "val": val_data, "test": test_data}



def normalize_data(df, sensor_columns):
    """
    Normalizes sensor features per patient using MinMaxScaler.
    """
    normalized_data = []

    for patient_id, patient_df in df.groupby('SubID'):
        scaler = MinMaxScaler()
        patient_df[sensor_columns] = scaler.fit_transform(patient_df[sensor_columns])
        normalized_data.append(patient_df)

    return pd.concat(normalized_data)

def create_fixed_windows_with_overlap(data, feature_columns, window_size, overlap):
    """
    Creates overlapping fixed-size windows from sequential data.
    """
    step_size = window_size - overlap
    all_windows, all_labels, all_sub_ids = [], [], []
    
    for patient_id, patient_data in data.groupby('SubID'):
        features = patient_data[feature_columns].values
        label = patient_data['Diagnosis'].iloc[0]  
        for start in range(0, len(features) - window_size + 1, step_size):
            end = start + window_size
            window = features[start:end]
            all_windows.append(window)
            all_labels.append(label)
            all_sub_ids.append(patient_id)

    return np.array(all_windows), np.array(all_labels), all_sub_ids

