import librosa
import soundfile as sf
import os
import csv

def get_audio_info(file_path):
    # Load the audio file using librosa
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

    # Get number of channels
    channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[0]

    # Get duration in seconds
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    # Load the file with soundfile to get bit depth
    with sf.SoundFile(file_path) as audio_file:
        subtype = audio_file.subtype  # e.g., 'PCM_16'
        # Extract bit depth from the subtype
        if 'PCM' in subtype:
            bit_depth = int(subtype.split('_')[-1])  # Extract number from 'PCM_16' -> 16
        else:
            # Handle other types or raise an error
            raise ValueError(f"Unsupported audio subtype: {subtype}")

    # Calculate the size of the audio data without the header
    audio_data_size = (sample_rate * bit_depth * channels * duration) / 8

    # Get the actual file size including header information
    file_size_with_header = os.path.getsize(file_path)

    # Calculate the header size
    header_size = file_size_with_header - audio_data_size

    return {
        "File Name": os.path.basename(file_path),
        "Sample Rate": sample_rate,
        "Channels": channels,
        "Duration (seconds)": duration,
        "Bit Depth": bit_depth,
        "File Size with Header (bytes)": file_size_with_header,
        "Audio Data Size without Header (bytes)": audio_data_size,
        "Header Size (bytes)": header_size
    }

def process_folder(folder_path, output_csv):
    # Prepare the CSV file to store results
    with open(output_csv, mode='a', newline='') as csv_file:
        fieldnames = ["File Name", "Sample Rate", "Channels", "Duration (seconds)", "Bit Depth", 
                      "File Size with Header (bytes)", "Audio Data Size without Header (bytes)", "Header Size (bytes)"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # writer.writeheader()

        for subfolder in os.listdir(folder_path):
            print(subfolder)
            subfolder_path = os.path.join(folder_path, subfolder)
            # Iterate over all .wav files in the folder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(subfolder_path, filename)
                    audio_info = get_audio_info(file_path)
                    writer.writerow(audio_info)

# Example usage:
folder_paths = ["/data/Vaani/Dataset/Audios_all_district_vaani_3", "/data/Vaani/Dataset/Audios_all_district_vaani_2"]  # Replace with your folder path
output_csv = "/data/Root_content/Vaani/audio_content_analysis/audio_info_results.csv"  # Specify your output CSV file name

for i in range(len(folder_paths)):
    process_folder(folder_paths[i], output_csv)

print(f"Results saved to {output_csv}")
