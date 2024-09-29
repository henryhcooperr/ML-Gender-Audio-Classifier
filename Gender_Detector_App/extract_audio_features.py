import librosa
import numpy as np
import os
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
from function_tracker import count_function_calls

@count_function_calls
def get_audio_files(audio_directory_path, test_size=5):
    audio_files = []
    labels = []
    for file in os.listdir(audio_directory_path):
        if file.endswith(".mp3"):
            audio_files.append(os.path.join(audio_directory_path, file))
            # Correctly assign labels based on the filename prefix
            label = 1 if file[0].lower() == "f" else 0  # 1 for female, 0 for male
            labels.append(label)
            #print(f"File: {file}, Label: {label}")  # Debug print
        else:
            raise "in get_audio_files found file without ending in .mp3"
    
    train_files, test_files, train_labels, test_labels = train_test_split(audio_files, labels, test_size=test_size, random_state=42)
    

    return train_files, train_labels, test_files, test_labels

@count_function_calls
def load_audio_file(file_path, sample_rate=None):
    """Load an audio file with the specified sample rate."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio, sr = librosa.load(file_path, sr=sample_rate)  # Load with the specified sample rate
        return audio, sr
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Failed to load audio file: {e}")
    return None, None


@count_function_calls
def extract_features(audio, sr, n_mfcc=13):
    """Extract MFCC features from audio with the given sample rate and number of coefficients."""
    if audio is None or len(audio) == 0:
        raise ValueError("Empty Audio Data Provided")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

@count_function_calls
def process_files(audio_files, labels, file_limit=None):
    """Process specified audio files to extract features along with their labels and filenames."""
    
    features_list = []
    filenames_list = []
    uniform_labels = []

    if file_limit is not None:
        audio_files = audio_files[:file_limit]
        labels = labels[:file_limit]
        #print("process_file: Audio Files: ", audio_files)
        #print("process_file: Labels: ", labels)

    progress_bar = tqdm(total=len(audio_files), unit="file", ncols=80,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]")

    #i = 0 # counter for debugign
    for file_path, label in zip(audio_files, labels):
        
        audio, sr = load_audio_file(file_path)
        if audio is not None:
            features = extract_features(audio, sr)

            features_list.append(features)
            filenames_list.append(os.path.basename(file_path))
            
            uniform_labels.append(label)  # Ensure labels are appended only when features are added
            
            #if i < 10:
                #print("filenames list ", filenames_list)
               # print("labels: ", uniform_labels)
                #print("features list: ", features_list)
                
            
          
        else:
            print(f"Warning: Audio loading failed for {file_path}")  # Optionally handle or log the failed load
        progress_bar.update(1)

    progress_bar.close()

    return features_list, filenames_list, uniform_labels