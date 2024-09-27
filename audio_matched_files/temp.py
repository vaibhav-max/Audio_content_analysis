import os
import shutil

# Path to your text file
txt_file_path = '/data1/manual_qc_references/utils/ref_audiomatch/audio_matched_files/files_that_are_not_at_right_path.txt'

# Destination folder where you want to copy the files
destination_folder = '/data1/manual_qc_references/utils/ref_audiomatch/audio_matched_files/files'

# Read the file paths from the text file
with open(txt_file_path, 'r') as file:
    file_paths = file.readlines()

# Copy each file to the destination folder
for file_path in file_paths:
    file_path = file_path.strip()  # Remove any extra whitespace/newlines
    if os.path.exists(file_path):
        shutil.copy(file_path, destination_folder)
        # print('Copied {} to {}'.format(file_path, destination_folder))
    else:
        print('File {} does not exist'.format(file_path))

print('File copying completed.')