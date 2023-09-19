import os
from scipy.signal import butter, filtfilt, gaussian, convolve
import numpy as np

class NeuralAnalysis:
    def __init__(self, file_path, n_channels=32, sampling_rate=30000, dtype=np.int16):
        self.file_path = file_path
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.dtype = dtype
        self.data = None
        self.good_channels = None
        self.noisy_channels = None

    def process_dat_file(self, project_folder_path):
        """
        Iteratively process .dat files in the project folder.
        
            Processes a .dat file to identify noisy channels based on RMS values.
    
        Parameters:
        - file_path: str, path to the .dat file
        - n_channels: int, number of recording channels (default: 32)
        - sampling_rate: int, data sampling rate in Hz (default: 32000)
        - dtype: numpy dtype, data type of the samples (default: np.int16)
        
        Returns:
        - noisy_channels: numpy array, indices of the noisy channels
        - good_channels: numpy array, indices of the good channels

        Parameters:
        - project_folder_path: str, the path to the project folder containing all the group and recording folders
        
        """
        
        # Define the path to the SpikeStuff folder
        spikestuff_path = os.path.join(project_folder_path, "SpikeStuff")

        # Loop through all groups in the SpikeStuff folder
        for group_name in os.listdir(spikestuff_path):
            group_path = os.path.join(spikestuff_path, group_name)

            # Ensure it's a directory and not a file
            if os.path.isdir(group_path):

                # Loop through all recordings in the group folder
                for recording_name in os.listdir(group_path):
                    recording_path = os.path.join(group_path, recording_name)

                    # Ensure it's a directory and not a file
                    if os.path.isdir(recording_path):

                        # Define the path to the SUA folder within the current recording folder
                        sua_path = os.path.join(recording_path, 'SUA')

                        # Check if the SUA directory exists before trying to access it
                        if os.path.isdir(sua_path):

                            # Loop through all files in the SUA folder to find .dat files
                            for file_name in os.listdir(sua_path):
                                if file_name.endswith(".dat"):
                                    dat_file_path = os.path.join(sua_path, file_name)
                                    
                                    # Now, dat_file_path is the path to a .dat file
                                    # Call your .dat file processing function here
                                        # Step 1: Read the .dat file
                                    data = np.fromfile(dat_file_path, dtype=dtype) #dtype=np.int16 by default
                                    reshaped_data = data.reshape((-1, self.n_channels))
                                    
                                    # Step 2: Compute the RMS value for each channel
                                    rms_values = np.sqrt(np.mean(np.square(reshaped_data), axis=0))
                                    
                                    # Step 3: Identify the 1st and 3rd quartiles of the RMS values
                                    q1 = np.percentile(rms_values, 25)
                                    q3 = np.percentile(rms_values, 75)
                                    iqr = q3 - q1
                                    
                                    # Step 4: Mark channels as excessively noisy
                                    lower_bound = q1 - 3 * iqr
                                    upper_bound = q3 + 3 * iqr
                                    
                                    noisy_channels = np.where((rms_values < lower_bound) | (rms_values > upper_bound))[0]
                                    good_channels = np.setdiff1d(np.arange(self.n_channels), noisy_channels)

                                    print(f"Found .dat file: {dat_file_path}")  # For testing
                        else:
                            print(f"No SUA directory found in {recording_path}")  # For testing
                            
    def read_dat_file(self):
        # code to read the .dat file and set self.data

    def process_dat_file(self):
        # code to process the .dat file and set self.good_channels and self.noisy_channels

    def common_average_reference(self):
        # code to apply common average reference to self.data using self.good_channels

    def isolate_MUA(self):
        # code to isolate MUA from self.data

    def estimate_firing_rate(self):
        # code to estimate firing rate from the data obtained in isolate_MUA