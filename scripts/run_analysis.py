from neural_analysis_pkg.core import NeuralAnalysis

# Define the path to your project folder
project_folder_path ='/home/cresp1el-local/Documents/MATLAB/Data/lmc_project_v2'

# Initialize the NeuralAnalysis class with the project folder path
analysis = NeuralAnalysis(project_folder_path)

# Print a message to confirm that the NeuralAnalysis object has been created
print("NeuralAnalysis object created successfully.")

#call the process_dat_file method and pass the project folder path
analysis.process_dat_file(project_folder_path)

# You can now use the methods of your class to perform analyses
# For example, assuming you have methods like `process_dat_file` and `perform_analysis` in your class:
# analysis.process_dat_file()
# analysis.perform_analysis()

# You might want to print some outputs or save results to files to verify that the analyses are working correctly
