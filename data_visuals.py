import matplotlib.pyplot as plt

def plot_sensor_signals(patient_sensor_df, signals_list, patient_id):
    """
    Plots the given list of signals from the patient sensor data, separated into left and right signals.
    """
    left_signals = [signal for signal in signals_list if signal.startswith('L_')]
    right_signals = [signal for signal in signals_list if signal.startswith('R_')]

    if left_signals and right_signals: 
        fig, axs = plt.subplots(2, 1, figsize=(15, 8))
        
        for signal in left_signals:
            axs[0].plot(patient_sensor_df.index, patient_sensor_df[signal], label=signal)  
        axs[0].set_title(f'Patient {patient_id} - Left Signals')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()

        for signal in right_signals:
            axs[1].plot(patient_sensor_df.index, patient_sensor_df[signal], label=signal)  
        axs[1].set_title(f'Patient {patient_id} - Right Signals')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()

    elif left_signals: 
        fig, ax = plt.subplots(figsize=(15, 6))
        for signal in left_signals:
            ax.plot(patient_sensor_df.index, patient_sensor_df[signal], label=signal)  
        ax.set_title(f'Patient {patient_id} - Left Signals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

    elif right_signals: 
        fig, ax = plt.subplots(figsize=(15, 6))
        for signal in right_signals:
            ax.plot(patient_sensor_df.index, patient_sensor_df[signal], label=signal)  
        ax.set_title(f'Patient {patient_id} - Right Signals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()
    plt.show()
