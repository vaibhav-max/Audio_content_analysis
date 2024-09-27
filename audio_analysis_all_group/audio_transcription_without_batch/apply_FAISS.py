import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import numpy as np
from tqdm import tqdm

# Initialize the Sentence Transformer model
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)

# Define the input and output folder paths
input_folder_path = "/data/Root_content/Vaani/audio_content_analysis/audio_analysis_all_group/audio_transcription_without_batch/csv_group_wise"
output_folder_path = "/data/Root_content/Vaani/audio_content_analysis/audio_analysis_all_group/audio_transcription_without_batch/csv_group_wise_similarity"

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Loop through each CSV file in the input folder
for csv_filename in tqdm(os.listdir(input_folder_path), desc= f"Processing CSVs"):
    if csv_filename.endswith(".csv"):
        # Construct the full path of the current CSV file 
        csv_file_path = os.path.join(input_folder_path, csv_filename)
        
        # Load the CSV file containing transcriptions
        df = pd.read_csv(csv_file_path)
        print("df length = ", len(df))
        
        # Check if 'Transcription' column exists in the CSV
        if 'Transcription' not in df.columns:
            print(f"Skipping {csv_filename} as it does not contain 'Transcription' column.")
            continue
        
        # Generate embeddings for each transcription
        df['Transcription'] = df['Transcription'].fillna('')
        transcriptions = df['Transcription'].tolist()
        filenames = df['File Name'].tolist()
        embedding_vectors = model.encode(transcriptions)
        
        # Normalize embeddings to use cosine similarity
        embedding_vectors = embedding_vectors / np.linalg.norm(embedding_vectors, axis=1, keepdims=True)
        
        # Initialize FAISS index for inner product (cosine similarity)
        dimension = embedding_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embedding_vectors)
        
        # Create a dictionary with filenames as keys and transcriptions as values
        transcription_dict = {filename: transcription for filename, transcription in zip(filenames, transcriptions)}
        
        # Create a reverse dictionary to map transcriptions back to filenames
        reverse_transcription_dict = {transcription: filename for filename, transcription in transcription_dict.items()}
        
        # Create a document store with filenames as keys
        docstore = InMemoryDocstore({filename: Document(page_content=transcription) for filename, transcription in transcription_dict.items()})
        index_to_docstore_id = {i: filename for i, filename in enumerate(filenames)}
        
        # Create FAISS vector store
        vector_store = FAISS(
            embedding_function=lambda x: model.encode(x),
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        # Prepare a list to store results
        results_list = []
        
        # Perform a similarity search for each transcription
        for filename, query in transcription_dict.items():
            try:
                search_results = vector_store.similarity_search_with_score(query, k=2)  # k=2 to get top 2 similar transcriptions
                
                for res, score in search_results:
                    # Extract the transcription from the result
                    result_transcription = res.page_content
                    
                    # Use the reverse dictionary to get the filename for the result transcription
                    result_filename = reverse_transcription_dict.get(result_transcription, None)
                    
                    if result_filename is None:
                        print(f"Result transcription not found in reverse dictionary: {result_transcription}")
                        continue
                    
                    if result_filename == filename:  # Skip the result of the query itself
                        continue
                    
                    result_row = {
                        "Filename_1": filename,
                        "Filename_2": result_filename,
                        "Similarity_score": round(score, 3)
                    }
                    results_list.append(result_row)
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")
        
        # Save the results for the current CSV file
        output_csv_filename = f"similarity_results_{csv_filename.split('.')[0]}_size_{len(df)}.csv"
        output_csv_file_path = os.path.join(output_folder_path, output_csv_filename)
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv_file_path, index=False)

        print(f"Similarity results for {csv_filename} saved to {output_csv_file_path}.")
