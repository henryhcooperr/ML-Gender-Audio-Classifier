from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from function_tracker import count_function_calls
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image, display



@count_function_calls
def train_decision_tree(X, y, filenames):
    """
    Train a Decision Tree Classifier.

    Args:
        X (numpy.ndarray): Feature array.
        y (numpy.ndarray): Label array.
        filenames (list): List of filenames corresponding to X and y.

    Returns:
        DecisionTreeClassifier: Trained Decision Tree Classifier model.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42)
    
    # Create and train the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

@count_function_calls
def train_random_forest(X, y, filenames):
    """
    Train a Random Forest Classifier.

    Args:
        X (numpy.ndarray): Feature array.
        y (numpy.ndarray): Label array.
        filenames (list): List of filenames corresponding to X and y.

    Returns:
        RandomForestClassifier: Trained Random Forest Classifier model.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
        X, y, filenames, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    
    return model

@count_function_calls
def evaluate_model(model, X_test, y_test, filenames_test, print_results=True):
    """
    Evaluate the given model using test data.

    Args:
        model (model): Trained model to evaluate.
        X_test (numpy.ndarray): Test feature array.
        y_test (numpy.ndarray): Test label array.
        filenames_test (list): List of filenames corresponding to X_test and y_test.
        print_results (bool): Flag to control the printing of the evaluation results. Default is True.

    Returns:
        tuple: A tuple containing accuracy, precision, and recall of the model.
    """
    if print_results:
        print("Evaluating model...")
    
    # Make predictions using the trained model
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')

    if print_results:
        print("Accuracy:", accuracy)
        print("Precision per class:", precision)
        print("Recall per class:", recall)
        
        # Print individual file predictions
        for filename, actual, predicted in zip(filenames_test, y_test, predictions):
            actual_label = 'Female' if actual == 1 else 'Male'
            predicted_label = 'Female' if predicted == 1 else 'Male'
            print(f"File: {filename}, Actual: {actual_label}, Predicted: {predicted_label}")

    return accuracy, precision, recall

@count_function_calls
def save_model(model, filename):
    """
    Save the trained model to disk.

    Args:
        model (model): Trained model to save.
        filename (str): Filename to save the model.
    """
    joblib.dump(model, filename)

@count_function_calls
def load_model(filename):
    """
    Load a trained model from disk.

    Args:
        filename (str): Filename of the saved model.

    Returns:
        model: Loaded trained model.
    """
    return joblib.load(filename)

def compare_model_accuracies(features_test, labels_test, filenames_test):
    print("Comparing model accuracies...")
    dt_model = load_model("decision_tree_model.pkl")
    rf_model = load_model("random_forest_model.pkl")
    dt_accuracy, dt_precision, dt_recall = evaluate_model(dt_model, features_test, labels_test, filenames_test, print_results=False)
    rf_accuracy, rf_precision, rf_recall = evaluate_model(rf_model, features_test, labels_test, filenames_test, print_results=False)
    print(f"Decision Tree - Accuracy: {dt_accuracy}, Precision: {dt_precision}, Recall: {dt_recall}")
    print(f"Random Forest - Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}")

def visualize_decision_tree(model, feature_names, class_names):
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names,
                               class_names=class_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # Setting graph size (width, height in inches)
    graph.set_size('"16,16!"')
    graph.set_dpi(2000)
    graph.write_png('decision_tree.png')
    display(Image(graph.create_png()))


