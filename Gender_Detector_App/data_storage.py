import numpy as np
from function_tracker import count_function_calls

@count_function_calls
def save_features_to_file(features, labels, filenames, file_path):
    """
    Save features, labels, and filenames to a file, ensuring labels are saved as integers.

    Args:
        features (numpy.ndarray): Array of feature vectors.
        labels (list): List of corresponding labels.
        filenames (list): List of corresponding filenames.
        file_path (str): Path to the file where the data will be saved.
    """
    with open(file_path, 'w') as f:
        for feature, label, filename in zip(features, labels, filenames):
            # Ensure label is stored as an integer
            f.write(f"{filename},{int(label)},{' '.join(map(str, feature))}\n")

@count_function_calls
def load_features_from_file(feature_file):
    """
    Load features, labels, and filenames from a file.

    Args:
        feature_file (str): Path to the file containing the feature data.

    Returns:
        tuple: A tuple containing three lists - filenames, labels, and features.
            - filenames (list): List of filenames.
            - labels (list): List of corresponding labels.
            - features (numpy.ndarray): Array of feature vectors.
    """
    filenames = []
    labels = []
    features = []

    with open(feature_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Assuming comma is the delimiter

            if len(parts) > 2:
                filenames.append(parts[0])
                labels.append(int(parts[1]))

                try:
                    feature_list = parts[2].split()  # Splitting the feature string into individual numbers
                    features.append([float(x) for x in feature_list])
                except ValueError as e:
                    print("Error converting string to float:", e)
                    print("Offending data:", parts[2])

    return filenames, labels, np.array(features)