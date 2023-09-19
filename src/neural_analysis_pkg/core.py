import os
from scipy.signal import butter, filtfilt, gaussian, convolve, resample_poly
import numpy as np
from time import time 

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
        spikestuff_path = os.path.join(project_folder_path, "SpikeStuff") # Define the path to the SpikeStuff folder, output is a string
        
        # Get the total number of groups (only count directories, not files)
        total_groups = sum(1 for item in os.listdir(spikestuff_path) if os.path.isdir(os.path.join(spikestuff_path, item)))
        print(f"Total number of groups: {total_groups}")

        recording_results = {}
        
        # Loop through all the groups in the SpikeStuff folder
        group_index = 0  # Add a group index counter here
        for group_name in os.listdir(spikestuff_path):
            group_path = os.path.join(spikestuff_path, group_name)

            # Ensure it's a directory and not a file , great job!
            if os.path.isdir(group_path):
                
            # Get the total number of recordings in the current group (only count directories, not files)
                total_recordings_in_group = sum(1 for item in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, item)))
                print(f"Group {group_index + 1}/{total_groups}: {group_name} has {total_recordings_in_group} recordings")

                #add the group name to the dicionary
                recording_results[group_name] = {}


                for recording_index, recording_name in enumerate(os.listdir(group_path)):
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
                                    
                                    # Step 1.1: Reshape the data to create a 2D array where each row is a channel and each column is a time point
                                    reshaped_data = data.reshape((-1, self.n_channels))
                                    
                                    # Log the shape and data type of the loaded data
                                    print(f"Data shape: {reshaped_data.shape}, Data type: {reshaped_data.dtype}")
                                    
                                    # Step 1.2: Calculate and print the full length of the recording in seconds
                                    recording_length_in_seconds = reshaped_data.shape[0] / self.sampling_rate
                                    print(f"The entire length of the recording is: {recording_length_in_seconds} seconds")

                                    # Step 2: Downsample the data from 30kHz to 10kHz using resample_poly
                                    # Calculate the expected length of the downsampled data
                                    downsample_factor = 3
                                    expected_length = int(np.ceil(reshaped_data.shape[0] / downsample_factor))

                                    # Initialize an empty array to store the downsampled data
                                    downsampled_data = np.empty((expected_length, self.n_channels))

                                    start_time = time() # Start the timer here
                                    # Loop through each channel to downsample the data
                                    for channel_idx in range(self.n_channels):
                                        print(f"Processing channel {channel_idx + 1}/{self.n_channels}")

                                        # Get the data for the current channel
                                        channel_data = reshaped_data[:, channel_idx]

                                        # Downsample the channel data using resample_poly
                                        downsampled_channel_data = resample_poly(channel_data, up=1, down=downsample_factor)

                                        # If the downsampled data is slightly longer or shorter than the expected length, trim or pad it
                                        if len(downsampled_channel_data) > expected_length:
                                            downsampled_channel_data = downsampled_channel_data[:expected_length]
                                        elif len(downsampled_channel_data) < expected_length:
                                            downsampled_channel_data = np.pad(downsampled_channel_data, (0, expected_length - len(downsampled_channel_data)))

                                        # Store the downsampled data in the appropriate column of the downsampled_data array
                                        downsampled_data[:, channel_idx] = downsampled_channel_data

                                    # Step 2.1: Save the downsampled data to a file
                                    output_file_path = os.path.join(recording_path, f"{file_name.split('.')[0]}_downsampled.npy")
                                    
                                    np.save(output_file_path, downsampled_data)
                                    end_time = time() # Stop the timer here
                                    print(f"Downsampled data saved to {output_file_path} in {end_time - start_time:.2f} seconds")

                                    # Step 3: Compute the RMS value for each channel
                                    #rms_values = np.sqrt(np.mean(np.square(downsampled_data), axis=0))
                                    
                                    # Step 4: Identify the 1st and 3rd quartiles of the RMS values
                                    #q1 = np.percentile(rms_values, 25)
                                    #q3 = np.percentile(rms_values, 75)
                                    #iqr = q3 - q1
                                    
                                    # Step 5: Mark channels as excessively noisy
                                    #lower_bound = q1 - 3 * iqr
                                    #upper_bound = q3 + 3 * iqr
                                    
                                    #noisy_channels = np.where((rms_values < lower_bound) | (rms_values > upper_bound))[0]
                                    #good_channels = np.setdiff1d(np.arange(self.n_channels), noisy_channels)

                                    
                                    # Store the results for this recording in the group's dictionary
                                    #recording_results[group_name][recording_name] = {
                                    #    "rms_values": rms_values,
                                    #    "iqr": iqr,
                                    #    "good_channels": good_channels,
                                    #    "noisy_channels": noisy_channels,
                                    #}
                                    
                                    # Adding print statements to log the identified good and noisy channels
                                    #print(f"Good channels: {good_channels}")
                                    #print(f"Noisy channels: {noisy_channels}")
                                    
                                    # Call the common_average_reference method with necessary inputs
                                    #self.common_average_reference(reshaped_data, good_channels)
                                    
                                    print(f"Processed recording {recording_index + 1}/{total_recordings_in_group} in group {group_index + 1}/{total_groups}")
                        else:
                            print(f"No SUA directory found in {recording_path}")  # For testing purposes
            group_index += 1  # Increment the group index counter here
        
        # Return the recording results dictionary
        # return recording_results
	      
		                    
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
