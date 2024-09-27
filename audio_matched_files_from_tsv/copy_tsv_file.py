import os
import shutil

# Path to the TSV file
tsv_file_path = '/data1/manual_qc_references/utils/ref_audiomatch/audio_match_allAudios_megdap_2024-06-03_spkrSize.tsv'

# Specify the destination directory
destination_dir = '/data1/manual_qc_references/utils/ref_audiomatch/audio_matched_files/files'

# Path to the error log file
error_log_path = '/data1/manual_qc_references/utils/ref_audiomatch/audio_matched_files/error_log.txt'

# Open the error log file for writing
with open(error_log_path, 'w') as error_log:

    # Open the TSV file and read it line by line
    with open(tsv_file_path, 'r') as file:
        # Skip the header line if there's one
        next(file)
        
        for line in file:
            # Split the line into columns
            columns = line.strip().split('\t')
            
            # Get the file paths from the first and second columns
            file1 = columns[0]
            file2 = columns[1]
            
            # Try to copy the files to the destination directory
            try:
                shutil.copy(file1, destination_dir)
            except IOError:
                # Extract and store only the filename
                error_log.write(os.path.basename(file1) + '\n')
            
            try:
                shutil.copy(file2, destination_dir)
            except IOError:
                # Extract and store only the filename
                error_log.write(os.path.basename(file2) + '\n')

print("Files copied with error handling! Error filenames logged to error_log.txt.")
