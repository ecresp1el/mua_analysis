import os
from scipy.signal import butter, filtfilt, gaussian, convolve
import numpy as np

class NeuralAnalysis:
    """ 
    Initializes new objects created from the NeuralAnalysis class. It is at the class level within your core.py file.
    """
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
        - project_folder_path: str, the path to the project folder containing all the group and recording folders

        """
        spikestuff_path = os.path.join(project_folder_path, "SpikeStuff")

        # Loop through all groups in the SpikeStuff folder
        all_group_names = os.listdir(spikestuff_path)
        total_groups = len(all_group_names)
        print(f"Total groups: {total_groups}")
     
        for group_index, group_name in enumerate(all_group_names):
            group_path = os.path.join(spikestuff_path, group_name)

            # Ensure it's a directory and not a file
            if os.path.isdir(group_path):

                # Loop through all recordings in the group folder
                all_recording_names = os.listdir(group_path)
                total_recordings_in_group = len(all_recording_names)
                print(f"Group {group_index + 1}/{total_groups}: {group_name} has {total_recordings_in_group} recordings")

                for recording_index, recording_name in enumerate(all_recording_names):
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
                                    data = np.fromfile(dat_file_path, dtype=self.dtype)
                                    reshaped_data = data.reshape((-1, self.n_channels))
                                    
                                    # Log the shape and data type of the loaded data
                                    print(f"Data shape: {reshaped_data.shape}, Data type: {reshaped_data.dtype}")

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

                                    print(f"Processing recording {recording_index + 1}/{total_recordings_in_group} in group {group_index + 1}/{total_groups}")
                    else:
                            print(f"No SUA directory found in {recording_path}")  # For testing
	      
		                    
    #def read_dat_file(self):
        # code to read the .dat file and set self.data

    #def process_dat_file(self):
        # code to process the .dat file and set self.good_channels and self.noisy_channels

    #def common_average_reference(self):
        # code to apply common average reference to self.data using self.good_channels

    #def isolate_MUA(self):
        # code to isolate MUA from self.data

    #def estimate_firing_rate(self):
        # code to estimate firing rate from the data obtained in isolate_MUA
