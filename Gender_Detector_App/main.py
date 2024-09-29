import os
import numpy as np
from extract_audio_features import load_audio_file, extract_features
from model import train_decision_tree, train_random_forest, evaluate_model, save_model, load_model, compare_model_accuracies, visualize_decision_tree
from data_storage import load_features_from_file
from function_tracker import count_function_calls
from sklearn.model_selection import train_test_split
from record_audio import record_sample, convert_audio_to_wav
import simpleaudio as sa



@count_function_calls
def main():
    """
    Main function to run the gender detection application.
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    feature_file = os.path.join(current_directory, "features.csv")

    if os.path.exists(feature_file):
        print("Features file found, loading data...")
        filenames, labels, features = load_features_from_file(feature_file)
        # Split data into train and test sets
        features_train, features_test, labels_train, labels_test, filenames_train, filenames_test = train_test_split(
            features, labels, filenames, test_size=0.2, random_state=42)
    else:
        print("No features file found. Exiting...")
        return

    while True:
        print("\nMenu:")
        print("1. Train decision tree model")
        print("2. Train random forest model")
        print("3. Evaluate model")
        print("4. Record audio and evaluate")
        print("5. Compare accuracies of models")
        print("6. Visualize Decision Tree")
        print("7. Exit")
        choice = input("Enter your choice (1-7): ")

        if choice == "1":
            if len(features_train) > 0 and len(labels_train) > 0:
                print("Training decision tree model...")
                model = train_decision_tree(features_train, labels_train, filenames_train)
                save_model(model, "decision_tree_model.pkl")
                print("Model trained and saved successfully.")
            else:
                print("Insufficient data for training. Please check your dataset.")
        elif choice == "2":
            if len(features_train) > 0 and len(labels_train) > 0:
                print("Training random forest model...")
                model = train_random_forest(features_train, labels_train, filenames_train)
                save_model(model, "random_forest_model.pkl")
                print("Random forest model trained and saved.")
            else:
                print("Insufficient data for training. Please check your dataset.")
        elif choice == "3":
            if len(features_test) > 0 and len(labels_test) > 0:
                print("Evaluating model...")
                model_type = input("Enter model type (decision_tree/random_forest): ")
                model_file = "decision_tree_model.pkl" if model_type == "decision_tree" else "random_forest_model.pkl"
                model = load_model(model_file)
                evaluate_model(model, features_test, labels_test, filenames_test)
            else:
                print("Insufficient test data available. Please check your dataset.")
        elif choice == "4":
            file_path = record_sample()
            file_path = convert_audio_to_wav(file_path)
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            model = load_model("random_forest_model.pkl")  # Load the random forest by default or ask user
            audio, sr = load_audio_file(file_path)
            if audio is not None:
                features = extract_features(audio, sr)
                features = np.array([features])  # Ensuring it's in the correct shape (2D array)
                prediction = model.predict(features)[0]
                predicted_label = 'Female' if prediction == 1 else 'Male'
                print(f"Predicted Label for Recorded Audio: {predicted_label}")
            else:
                print("Failed to load or process audio file.")
        elif choice == "5":
            compare_model_accuracies(features_test, labels_test, filenames_test)
        elif choice == "6":
            model = load_model("decision_tree_model.pkl")
            #each feature represents the power of 
            feature_names = ['f1: Loudness Overall', 'f2: Spectral Shape 1', 'f3: Spectral Shape 2', 'f4: Spectral Shape 3', 'f5: Spectral Shape 4', 'f6: Spectral Shape 5', 'f7: Spectral Shape 6', 'f8: Spectral Shape 7', 'f9: Spectral Shape 8', 'f10: Spectral Shape 9', 'f11: Spectral Shape 10', 'f12: Spectral Shape 11', 'f13: Spectral Shape 12']
            class_names = ['Male', 'Female']  # Replace with actual class names
            visualize_decision_tree(model, feature_names, class_names)
        elif choice == "7":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()