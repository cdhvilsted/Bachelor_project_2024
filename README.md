# Detection of arousal in clinically unresponsive patients with brain injury from EEG data

## Bachelor project 2024
### DTU Compute, BSc Artificial Intelligence and Data
#### In collaboration with Rigshospitalet Department of Neurology 

A project analysing EEG data using Machine Learning models to predict covert consciousness in patients with brain injury.

Due to accidently publishing sensitive data to a previous private repository, this new resporitory is created and used for further work. The initial repository is deleted and cannot be found.

The commit history at initial repository began at February 28th 2024.

The folder 'old_scripts' holds previous scripts which are no longer used in the project.

A description of the files is found below:
- 'manual_inspection': Loads the .edf files as a mne Raw object, then performs all preprocessing needed to ensure signal quality. The function 'LoadRaw' lastly saves the preprocessed data for each patient in a .fif file.
- 'split_based_on_resting': The function 'LoadRawP' reads the raw file for each patient and divide it into the respective 3 events, namely 'Resting', 'Familiar voice' and 'Medical staff' (also mentioned as 'Unfamiliar voice').
- 'ECG_features': Calculates all ECG related feature values from individual epochs.
- 'Feature_calculation': The function 'FeatureMatrix2' calculates all feature values for each patient and saving them in a matrix.
- 'csv_file_writer_final': The function 'allToCsv' creates a CSV file for each patient holding all feature values in all events. The function 'fifToCsvP' can calculate the feature values from the .fif file of the patient.
- 'cross_val_pipeline_group': A notebook holding the entire group-level analysis, as well as classification results and clustering results. Additionally, the classification results are analysed using a corrected t-test.
- 'cross_val_pipeline_individual': A notebook holding the entire individual-level analysis. Both classification results and corrected t-test results are calculated.
- 'feature_matrix_plot': Visualizing a matrix of scatter plots for all feature combinations. Hereby correlation between features can be found as well as distributions of feature values.
- 'feature_stats': Creating a box plot for all feature values across all patients.
- 'fooof_slope': Plotting the aperiodic curve fitted to data.
- 'channels_across_files': Investigating the channels included in the data collection for all patients.
- 'ICA_scalpmaps_plot': A notebook for plotting ICA components as scalp maps.
- 'Montage_Plot': Visualizing the montage used for data collection of this project (according to the 10-20 International System) as well as the montage used in the nice toolbox.

The project was ended June 14th 2024.

Katharina Strauss Søgaard,
Cecilie Dahl Hvilsted
