import os
from scipy.signal import butter, filtfilt, gaussian, convolve, resample_poly
from scipy.io import loadmat
import numpy as np
from time import time 
import gc 
import pandas as pd 
import matplotlib.pyplot as plt
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
        
        
        # Try to load the existing recording_results_df from a CSV file
        self.recording_results_df = self._load_recording_results_df()
        
    def _load_recording_results_df(self):
        """
        Private method to load the recording results dataframe from an existing CSV file.
        """
        csv_file_path = os.path.join(self.file_path, "SpikeStuff", "recording_results.csv")

        
        if os.path.exists(csv_file_path):
            recording_results_df = pd.read_csv(csv_file_path)
            # Convert 'good_channels' and 'noisy_channels' from string representation to lists
            recording_results_df['good_channels'] = recording_results_df['good_channels'].apply(eval)
            recording_results_df['noisy_channels'] = recording_results_df['noisy_channels'].apply(eval)
            recording_results_df['rms_values'] = recording_results_df['rms_values'].apply(eval)
            print(f"Loaded existing recording results dataframe from {csv_file_path}")
            return recording_results_df
        else:
            print("No existing recording results dataframe found. Run process_dat_file method to generate it.")
            return None
        
    def process_dat_file(self, project_folder_path):
        """
        Iteratively process .dat files in the project folder.

        This method processes .dat files within a given project folder. It identifies noisy channels based on root-mean-square (RMS) values, and 
        performs downsampling. The method works on multiple groups and recordings, and results are saved as a DataFrame.

        Parameters
        ----------
        project_folder_path : str
            The absolute path to the project folder. This folder should contain all the group and recording folders.

        Returns
        -------
        dict
            A dictionary containing processed information from .dat files, keyed by group name and recording name.

        Notes
        -----
        - The project folder should contain a "SpikeStuff" folder.
        - Each group folder within "SpikeStuff" should contain individual recording folders.
        - Each recording folder should have an 'SUA' sub-folder containing .dat files.
        - The dtype, n_channels, and sampling_rate attributes should be initialized before calling this method.
        
        Examples
        --------
        >>> obj = YourClass(dtype=np.int16, n_channels=32, sampling_rate=30000)
        >>> obj.process_dat_file("/path/to/project_folder")

        """
        spikestuff_path = os.path.join(project_folder_path, "SpikeStuff") # Define the path to the SpikeStuff folder, output is a string
        
        # Get the total number of groups (only count directories, not files)
        total_groups = sum(1 for item in os.listdir(spikestuff_path) if os.path.isdir(os.path.join(spikestuff_path, item)))
        print(f"Total number of groups: {total_groups}")
        
        recording_results = {}  # Initialize an empty dictionary to store the results
   
        # Loop through all the groups in the SpikeStuff folder
        group_index = 0  # Add a group index counter here
        for group_name in os.listdir(spikestuff_path):
            group_path = os.path.join(spikestuff_path, group_name)

            # Ensure it's a directory and not a file , great job!
            if os.path.isdir(group_path):
                
            # Get the total number of recordings in the current group (only count directories, not files)
                total_recordings_in_group = sum(1 for item in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, item)))
                print(f"Group {group_index + 1}/{total_groups}: {group_name} has {total_recordings_in_group} recordings")

                # Initialize the dictionary for the current group
                recording_results[group_name] = {}

                # Loop through all recordings in the current group
                for recording_index, recording_name in enumerate(os.listdir(group_path)): #recording_index is the index of the recording in the group, recording_name is the name of the recording
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
                                    
                                    # Before starting the channel processing loop, define output_file_path
                                    output_file_path = os.path.join(recording_path, f"{file_name.split('.')[0]}_downsampled.npy")
                                    
                                    #check if the downsampled file already exists
                                    if os.path.exists(output_file_path):
                                        print(f"Downsampled file already exists at {output_file_path} ...skipping downsampling and recalculating RMS") #if the file already exists, skip the downsampling step
                                        
                                        #proceed to calculate the RMS values
                                        recording_metrics = self.process_downsampled_data(output_file_path)

                                        recording_info = {
                                            "group_name": group_name,
                                            "recording_name": recording_name,
                                            "downsampled_path": output_file_path,
                                            "rms_values": recording_metrics['rms_values'], 
                                            "iqr": recording_metrics['iqr'],
                                            "good_channels": recording_metrics['good_channels'],
                                            "noisy_channels": recording_metrics['noisy_channels']
                                        }
                                        
                                        recording_results[group_name][recording_name] = recording_info
                                        
                                        print(f"Processed RMS for recording {recording_index + 1}/{total_recordings_in_group} in group {group_index + 1}/{total_groups}")
                                        
                                    if not os.path.exists(output_file_path):
                                        print(f"Downsampled file does not exist at {output_file_path} ...proceeding with downsampling")
                                    
                                        # Now, dat_file_path is the path to a .dat file
                                        # Call your .dat file processing function here
                                        # Step 1: Read the .dat file
                                        data = np.fromfile(dat_file_path, dtype=self.dtype)
                                        #check the initial data loading 
                                        print(f"Initial data min: {np.min(data)}, max: {np.max(data)}")
                                        
                                        # Step 1.1: Reshape the data to create a 2D array where each row is a channel and each column is a time point
                                        reshaped_data = data.reshape((-1, self.n_channels))
                                        #check data after reshaping
                                        print(f"Reshaped data min: {np.min(reshaped_data)}, max: {np.max(reshaped_data)}")
                                        
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
                                            channel_data = reshaped_data[:, channel_idx].astype(np.float32)


                                            # Downsample the channel data using resample_poly
                                            downsampled_channel_data = resample_poly(channel_data, up=1, down=downsample_factor)
                                            
                                            print(f"Downsampled data min: {np.min(downsampled_channel_data)}, max: {np.max(downsampled_channel_data)}")

                                            # If the downsampled data is slightly longer or shorter than the expected length, trim or pad it
                                            if len(downsampled_channel_data) > expected_length:
                                                downsampled_channel_data = downsampled_channel_data[:expected_length]
                                            elif len(downsampled_channel_data) < expected_length:
                                                downsampled_channel_data = np.pad(downsampled_channel_data, (0, expected_length - len(downsampled_channel_data)))

                                            # Store the downsampled data in the appropriate column of the downsampled_data array
                                            downsampled_data[:, channel_idx] = downsampled_channel_data #what is the precision at this point?
                                        
                                        # Print the min and max of the downsampled data after processing all the channels
                                        print(f"Min and Max for all channels in a recording: {np.min(downsampled_data)} and {np.max(downsampled_data)}")
                                            
                                        # Save the downsampled data to a .npy file
                                        np.save(output_file_path, downsampled_data)
                                        end_time = time() # Stop the timer here
                                        
                                        # Clear the large variables to free up memory
                                        del downsampled_data
                                        del data
                                        del reshaped_data
                                        gc.collect()  # Call the garbage collector to free up memory
                                        
                                        elapsed_time = end_time - start_time # Calculate the elapsed time
                                        minutes, seconds = divmod(elapsed_time, 60) # Convert elapsed time to minutes and seconds
                                        
                                        print(f"Downsampled data saved to {output_file_path} in {int(minutes)} minutes and {seconds:.2f} seconds")
                                    
                                        recording_metrics = self.process_downsampled_data(output_file_path)

                                        recording_info = {
                                            "group_name": group_name,
                                            "recording_name": recording_name,
                                            "downsampled_path": output_file_path,
                                            "rms_values": recording_metrics['rms_values'], 
                                            "iqr": recording_metrics['iqr'],
                                            "good_channels": recording_metrics['good_channels'],
                                            "noisy_channels": recording_metrics['noisy_channels']
                                        }
                                        
                                        recording_results[group_name][recording_name] = recording_info
                                        
                                        print(f"Processed recording {recording_index + 1}/{total_recordings_in_group} in group {group_index + 1}/{total_groups}")
                        else:
                            print(f"No SUA directory found in {recording_path}")  # For testing purposes
            group_index += 1  # Increment the group index counter here

        recording_results_flat = []
        for group_name, group_data in recording_results.items():
            for recording_name, recording_data in group_data.items():
                recording_data['group_name'] = group_name
                recording_data['recording_name'] = recording_name
                recording_results_flat.append(recording_data)
        self.recording_results_df = pd.DataFrame(recording_results_flat)
        self.recording_results_df.to_csv(os.path.join(spikestuff_path, 'recording_results.csv'), index=False)
        
        # Return the recording results dictionary: the keys are group names, the values are dictionaries of recording names and results
        return recording_results
        
    def process_downsampled_data(self, downsampled_file_path):
        """
        Process the downsampled data to compute RMS values and identify good and noisy channels.

        Parameters:
        - downsampled_file_path: str, the path to the downsampled data file
        """
        
        # Construct the correct path to the downsampled data file
    
        # Load the downsampled data from the correct path
        downsampled_data = np.load(downsampled_file_path)
        
        # Step 2: Compute the RMS value for each channel
        rms_values = np.sqrt(np.mean(np.square(downsampled_data), axis=0)) #rows are time points, columns are channels, so axis=0 is the channels
        print(f"RMS values from proccess_downsampled_data method: {rms_values}")
        
        # Step 3: Identify the 1st and 3rd quartiles of the RMS values
        q1 = np.percentile(rms_values, 25) #percentile returns the qth percentile(s) of the array elements
        q3 = np.percentile(rms_values, 75) #percentile returns the qth percentile(s) of the array elements 
        iqr = q3 - q1 #interquartile range is the difference between the 3rd and 1st quartiles
        
        # Step 4: Mark channels as excessively noisy
        lower_bound = q1 - 3 * iqr #lower bound is the 1st quartile minus 3 times the interquartile range 
        upper_bound = q3 + 3 * iqr #upper bound is the 3rd quartile plus 3 times the interquartile range 
        
        # Step 5: Identify good and noisy channels based on the RMS values and bounds
        # to get the indices of the noisy channels, we need to use np.where to get the indices of the values that are less than the lower bound or greater than the upper bound
        noisy_channels = np.where((rms_values < lower_bound) | (rms_values > upper_bound))[0] #where returns a tuple, so we need to index it to get the array
        good_channels = np.setdiff1d(np.arange(self.n_channels), noisy_channels) #setdiff1d returns the set difference of two arrays, so we get the good channels by subtracting the noisy channels from the total number of channels
        
        # Store the results in a dictionary
        recording_results = {
            "downsampled_path": downsampled_file_path,  # Include the file path in your results dictionary
            "rms_values": list(rms_values),
            "iqr": iqr,
            "good_channels": list(good_channels),
            "noisy_channels": list(noisy_channels),
        }
        
        # Print the identified good and noisy channels
        print(f"Good channels: {good_channels}")
        print(f"Noisy channels: {noisy_channels}")
        print(f"Min and Max of downsampled data: {np.min(downsampled_data)} and {np.max(downsampled_data)}")
        
        
        # You can return the results to use them later
        del downsampled_data # Clear the large variables to free up memory
        gc.collect() # Call the garbage collector to free up memory
        
        return recording_results 
    
    def process_MUA(self, bandpass_low=500, bandpass_high=5000, order=3):
        """
        Process the re-referenced signals to isolate MUA and save the downsampled MUA activity.

        Parameters:
        - bandpass_low: int, the lower frequency for the bandpass filter (default: 500 Hz)
        - bandpass_high: int, the higher frequency for the bandpass filter (default: 5000 Hz)
        - order: int, the order of the Butterworth filter (default: 3)
        """
        
        total_recordings = len(self.recording_results_df) #the number of rows in the DataFrame which is the number of recordings used for printing progress
        
        # Define the bandpass filter using butter
        nyq = 0.5 * (self.sampling_rate / 3)  # Nyquist frequency based on the downsampled rate. Original sampling rate is 30kHz, so we divide by 3 to get the downsampled rate, then divide by 2 to get the Nyquist frequency
        bandpass_high = 4999  # A value safely below the Nyquist frequency to avoid aliasing
        low = bandpass_low / nyq # Normalize the cutoff frequencies to the Nyquist frequency
        high = bandpass_high / nyq # Normalize the cutoff frequencies to the Nyquist frequency

        # Define highpass and lowpass filters separately
        b_high, a_high = butter(order, low, btype='high') #
        b_low, a_low = butter(order, high, btype='low')
        
        # New column to store the paths of the downsampled MUA activity files
        mua_paths = []

        # Loop through each recording in the DataFrame
        for idx, row in self.recording_results_df.iterrows():
            
            print(f"Processing recording {idx+1}/{total_recordings}...")
            
            # Load the downsampled data from the path in the current row
            downsampled_data = np.load(row['downsampled_path'])
            downsampled_data = downsampled_data.astype(np.float32)


            # Step 1: CAR Re-Referencing in place (subtract the mean across channels from each channel)
            car_reference = np.mean(downsampled_data[:, row['good_channels']], axis=1, keepdims=True)
            downsampled_data -= car_reference #this is in place, so it modifies the downsampled_data array directly to get the CAR-referenced data to save memory
            
            # After the CAR re-referencing step
            downsampled_data[:, row['noisy_channels']] = np.nan

            # Step 2: Bandpass Filtering for MUA Isolation
            for ch_idx in row['good_channels']:
                #print the channel index to check progress 
                print(f"Processing channel {ch_idx + 1}/{self.n_channels}")
                downsampled_data[:, ch_idx] = filtfilt(b_high, a_high, downsampled_data[:, ch_idx])
                downsampled_data[:, ch_idx] = filtfilt(b_low, a_low, downsampled_data[:, ch_idx])

            # Step 3: Save the Processed Data
            # Define the output file path and save the downsampled MUA data
            output_file_path = os.path.join(os.path.dirname(row['downsampled_path']), f"{os.path.basename(row['downsampled_path']).replace('_downsampled', '_MUA')}")
            
            np.save(output_file_path, downsampled_data)
            
            print(f"Recording {idx+1}/{total_recordings} processed and MUA data saved at {output_file_path}")
            
            del downsampled_data # Clear the large variables to free up memory
            gc.collect() # Call the garbage collector to free up memory
            
            # Append the new path to the mua_paths list
            mua_paths.append(output_file_path)
        
        # Update the DataFrame with the new column
        self.recording_results_df['mua_data_path'] = mua_paths

        # Return the updated DataFrame
        return self.recording_results_df  
    
    
    def extract_stimulation_data(self):
        # List to store individual dataframes for each recording
        df_list = []

        for idx, row in self.recording_results_df.iterrows():
            # Construct the path to the .mat file
            mat_file_path = os.path.join(
                self.file_path, 
                'SpikeStuff',
                row['group_name'], 
                row['recording_name'], 
                'MUA', 
                'allData', 
                'timestamp_s.mat'
            )
            
            # Load the .mat file with scipy.io.loadmat
            mat_data = loadmat(mat_file_path)
            
            # Access the data using the 'timestamp_s' key and create a DataFrame
            timestamp_data = mat_data['timestamp_s']
            df = pd.DataFrame(timestamp_data, columns=['onset_times', 'offset_times', 'stimulation_ids'])
            
            # Add group_name and recording_name columns
            df['group_name'] = row['group_name']
            df['recording_name'] = row['recording_name']
            
            # Convert 'stimulation_ids' to integers
            df['stimulation_ids'] = df['stimulation_ids'].astype(int)
            
            # Remove rows with negative 'onset_times' and 'offset_times'
            df = df[(df['onset_times'] >= 0) & (df['offset_times'] >= 0)]
                
            # Append the DataFrame to the list
            df_list.append(df)
        
        # Concatenate all individual dataframes to create a single DataFrame
        self.stimulation_data_df = pd.concat(df_list, ignore_index=True)
        
    def extract_spike_times(self):
        """
        Extracts spike times from the MUA data and saves them in structured arrays.
        """

        # Iterating through each recording
        for idx, row in self.recording_results_df.iterrows():
            # Load the MUA data
            mua_data_path = row['mua_data_path']
            mua_data = np.load(mua_data_path)
            
            # Estimate the noise standard deviation
            noise_std_estimate = np.median(np.abs(mua_data), axis=0) / 0.6745
            
            # Find spikes: data points less than -3 times the noise standard deviation
            spike_indices = np.where(mua_data < -3 * noise_std_estimate) #returns a tuple of arrays, the first array is the time index, the second array is the channel index
            
            # Convert spike indices to times (in seconds) based on the downsampled rate (10 kHz)
            spike_times = spike_indices[0] / 10000 #the first index of spike_indices is the time index
            spike_channels = spike_indices[1] #this is the channel index, not the channel number 

            # Create a structured array to store the spike times and channels
            spike_data = np.zeros(spike_times.shape[0], dtype=[('time', 'f8'), ('channel', 'i4')]) #the first column is the time, the second column is the channel
            spike_data['time'] = spike_times #assign the spike times to the first column
            spike_data['channel'] = spike_channels #assign the spike channels to the second column

            # Define the output file path and save the spike data
            output_file_path = os.path.join(os.path.dirname(mua_data_path), f"{os.path.basename(mua_data_path).replace('_MUA.npy', '_spike_times.npy')}")
            np.save(output_file_path, spike_data)
            print(type(spike_data))  # Add this line to check the type of spike_data

            # Print a message to indicate progress
            print(f"Processed and saved spike times for recording {idx+1}/{len(self.recording_results_df)}")

        print("Spike time extraction completed.")
        
    def calculate_firingratesfor8HzLEDstim_and_plot_heatmap_for_specific_recording(self, recording_name, plot_heatmap=True):
        """
        Calculate firing rates for 8Hz LED stimulation and optionally plot a heatmap for a specific recording.

        Parameters
        ----------
        recording_name : str
            The name of the recording for which to calculate firing rates and plot the heatmap.
        plot_heatmap : bool, optional
            Whether to plot the heatmap. Default is True.

        Returns
        -------
        np.ndarray
            A 2D array containing the firing rates for each channel and stimulus event. The shape of the array is (n_channels, n_stimulus_events).
            Entire rows with value of NaN indicates that the corresponding channel at that index is a noisy channel.

        Notes
        -----
        This method assumes that `recording_results_df` and `stimulation_data_df` are attributes of the NeuralAnalysis class.
        """
        
        # Get the mua_data_path for the current recording to get the spike_data_path where the spike times are stored
        mua_data_path = self.recording_results_df.loc[
            self.recording_results_df['recording_name'] == recording_name, 
            'mua_data_path'
        ].values[0]
        
        # Construct the spike_data_path from the mua_data_path
        spike_data_path = mua_data_path.replace('_MUA.npy', '_spike_times.npy')
        
        # Step 1: Identify the time windows for stimulus_id = 8, which is an 8Hz signal 
        stim_data = self.stimulation_data_df[
            (self.stimulation_data_df['recording_name'] == recording_name) & 
            (self.stimulation_data_df['stimulation_ids'] == 8)
        ]
        
        # Step 2: Load the corresponding spike times
        spike_data = np.load(spike_data_path, allow_pickle=True)
        spike_times = spike_data['time'] # spike_times are in seconds
        spike_channels = spike_data['channel']
        
        # Get good and noisy channels for the current recording
        good_channels = self.recording_results_df.loc[
            self.recording_results_df['recording_name'] == recording_name, 
            'good_channels'
        ].values[0]
        
        # Step 3 & 4: Calculate and aggregate the firing rates
        # Initialize a 2D array with NaNs. The shape is (number of channels, number of stimulus events).
        firing_rates = np.full((self.n_channels, len(stim_data)), np.nan) 

        # Loop through each stimulus event to calculate the firing rates.
        for i, (onset, offset) in enumerate(zip(stim_data['onset_times'], stim_data['offset_times'])):
            # Only good channels are considered, and their firing rates are stored in the corresponding indices.
            # Loop through each good channel
            
            for ch in good_channels:
                # Find spikes in the current channel that fall within the current time window (between onset and offset)
                condition = (spike_channels == ch) & (spike_times >= onset) & (spike_times <= offset)
                spikes_in_window = spike_times[np.where(condition)]
                
                # Calculate the firing rate as the number of spikes divided by the duration of the time window
                firing_rate = len(spikes_in_window) / (offset - onset)
                firing_rates[ch, i] = firing_rate
                
        # Step 5: Plotting the heatmap (optional)
        if plot_heatmap:
            plt.imshow(firing_rates, aspect='auto', cmap='hot', interpolation='nearest')
            plt.colorbar(label='Firing Rate (Hz)')
            plt.ylabel('Channel')
            plt.xlabel('Trial')
            plt.title(f'Firing Rate Heatmap for 8 Hz LED, {recording_name}')
            plt.yticks(range(self.n_channels), range(1, self.n_channels+1))  # Label y-axis with channel numbers
            plt.show()
        
        return firing_rates