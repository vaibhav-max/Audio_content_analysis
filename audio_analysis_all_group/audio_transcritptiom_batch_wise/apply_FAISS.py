import pandas as pd
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import numpy as np
from tqdm import tqdm  # Progress bar

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Check for multiple GPUs
device_ids = list(range(torch.cuda.device_count()))  # List of GPU ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Wrap the model with DataParallel if using multiple GPUs
model = torch.nn.DataParallel(model, device_ids=device_ids)
model.to(device)

root_folders = ['/data/Vaani/Dataset/Audios_all_district_vaani_3']

# Function to get transcription for a single file
def get_transcription(file_path):
    if os.path.exists(file_path):
        audio, sr = librosa.load(file_path, sr=None)
        input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=sr).input_values.to(device)

        with torch.no_grad():  # Disable gradient calculations
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    
    return None

# Create lists to store file information
file_data = {
    "File Name": [],
    "Duration": [],
    "Byte Size": [],
    "Transcription": []
}

# Process each folder and display progress with tqdm
for root_folder in root_folders:
    for subdir, _, files in os.walk(root_folder):
        print(f"Processing directory: {subdir}")
        files.sort()
        
        # Use tqdm for the files loop to track progress
        for audio_file in tqdm(files, desc=f"Processing files in {subdir}"):
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(subdir, audio_file)

                # Get file information
                duration = librosa.get_duration(filename=audio_file_path)
                byte_size = os.path.getsize(audio_file_path)

                # Get transcription
                transcription = get_transcription(audio_file_path)

                # Append file info to the data
                file_data["File Name"].append(audio_file_path)
                file_data["Duration"].append(duration)
                file_data["Byte Size"].append(byte_size)
                file_data["Transcription"].append(transcription)

# Create DataFrame from collected data
df = pd.DataFrame(file_data)

# Save the DataFrame to a CSV file
output_csv_path = "/data/Root_content/Vaani/audio_content_analysis/audio_analysis_all_group/temp.csv"
df.to_csv(output_csv_path, index=False)

print(f"CSV file saved to {output_csv_path}")
