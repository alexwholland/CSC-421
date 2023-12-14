import os
import string
import re
import numpy as np
from typing import Dict, List, Tuple

class BernoulliNaiveBayes:
    def __init__(self, pos_probs: Dict[str, float], neg_probs: Dict[str, float]):
        """
        Initialize the Bernoulli Naive Bayes classifier.

        Args:
            pos_probs (Dict[str, float]): Dictionary of positive probabilities for each word.
            neg_probs (Dict[str, float]): Dictionary of negative probabilities for each word.
        """
        self.pos_prob = pos_probs
        self.neg_prob = neg_probs

    def make_prediction(self, X_test: List[str]) -> int:
        """
        Make a prediction for a given test instance.

        Args:
            X_test (List[str]): List of words in the test instance.

        Returns:
            int: Predicted class (1 for positive, 0 for negative).
        """
        epsilon = 1e-10

        pos_probs = [np.log10(self.pos_prob.get(word, epsilon)) for word in X_test]
        neg_probs = [np.log10(self.neg_prob.get(word, epsilon)) for word in X_test]
        pos_given_doc = np.sum(pos_probs)
        neg_given_doc = np.sum(neg_probs)

        prediction = 1 if pos_given_doc > neg_given_doc else 0
        return prediction

    def assess_performance(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Assess the performance of the classifier on the given dataset.

        Args:
            X (np.ndarray): Array of input instances.
            y (np.ndarray): Array of true labels.

        Returns:
            Tuple[float, np.ndarray]: Accuracy and confusion matrix.
        """
        predictions = np.array([self.make_prediction(instance) for instance in X])

        true_positive = np.sum((y == 1) & (predictions == 1))
        false_positive = np.sum((y == 0) & (predictions == 1))
        true_negative = np.sum((y == 0) & (predictions == 0))
        false_negative = np.sum((y == 1) & (predictions == 0))

        accuracy = (true_positive + true_negative) / len(y) if len(y) != 0 else 0

        confusion_matrix = np.array([[true_positive, false_positive], [false_negative, true_negative]])
        return accuracy, confusion_matrix

def clean_text(input_text: str) -> str:
    """
    Clean the input text by removing punctuation, extra spaces, and leading/trailing whitespace.

    Args:
        input_text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    cleaned_text = input_text.translate(str.maketrans('', '', string.punctuation))
    output_text = re.sub(r'\s+', ' ', cleaned_text)
    return output_text.strip()

def create_word_dictionary(input_text: str) -> Dict[str, float]:
    """
    Create a word dictionary from the input text.

    Args:
        input_text (str): Input text to be converted into a dictionary.

    Returns:
        Dict[str, float]: Dictionary with words as keys and their normalized frequencies as values.
    """
    dictionary = {}
    words = input_text.split(" ")
    for word in words:
        dictionary[word] = dictionary.get(word, 0) + 1 / len(words)
    return dictionary

def read_text_file(file_path: str) -> str:
    """
    Read the content of text files in a given directory.

    Args:
        file_path (str): Path to the directory containing text files.

    Returns:
        str: Concatenated content of all text files.
    """
    all_text = ""
    for txt_file in os.listdir(file_path):
        with open(os.path.join(file_path, txt_file)) as file:
            all_text += file.read() + " "
    return all_text

def get_random_data_point() -> Tuple[List[str], int]:
    """
    Get a random data point from either the positive or negative class.

    Returns:
        Tuple[List[str], int]: Cleaned content and class label (1 for positive, 0 for negative).
    """
    pos_or_neg = np.random.choice([0, 1])
    path = pos_path if pos_or_neg == 1 else neg_path

    lib = os.listdir(path)
    random_choice = np.random.choice(lib)
    with open(os.path.join(path, random_choice)) as file:
        content = file.read()
        cleaned_content = clean_text(content).split()
    return cleaned_content, pos_or_neg

def generate_training_data() -> Tuple[List[List[str]], np.ndarray]:
    """
    Generate training data and labels from positive and negative sets.

    Returns:
        Tuple[List[List[str]], np.ndarray]: Training data and corresponding labels.
    """
    positive_set = []
    negative_set = []

    for pos_file, neg_file in zip(os.listdir(pos_path), os.listdir(neg_path)):
        with open(os.path.join(pos_path, pos_file)) as pos_file_content, open(os.path.join(neg_path, neg_file)) as neg_file_content:
            positive_set.append(clean_text(pos_file_content.read()).split())
            negative_set.append(clean_text(neg_file_content.read()).split())

    training_data = positive_set + negative_set
    labels = np.ones(len(training_data), dtype=int)
    labels[len(positive_set):] = 0
    return training_data, labels

def split_data(dataset: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset and labels into training and testing sets.

    Args:
        dataset (np.ndarray): Array of input instances.
        labels (np.ndarray): Array of true labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training data, testing data, training labels, and testing labels.
    """
    np.random.seed(42)

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    dataset = np.array(dataset, dtype=object)

    split_index = int(0.8 * num_samples)
    train_data, test_data = dataset[indices[:split_index]], dataset[indices[split_index:]]
    train_labels, test_labels = labels[indices[:split_index]], labels[indices[split_index:]]
    return train_data, test_data, train_labels, test_labels

pos_path = os.path.join(os.getcwd(), 'txt_sentoken', 'pos')
neg_path = os.path.join(os.getcwd(), 'txt_sentoken', 'neg')

# Load and clean text data
raw_positive, raw_negative = read_text_file(pos_path), read_text_file(neg_path)
cleaned_positive = clean_text(raw_positive)
cleaned_negative = clean_text(raw_negative)

# Create word dictionaries
positive_dict = create_word_dictionary(cleaned_positive)
negative_dict = create_word_dictionary(cleaned_negative)

# Get a random data point
X_test, y_test = get_random_data_point()

# Initialize the classifier
classifier = BernoulliNaiveBayes(positive_dict, negative_dict)

# Make prediction and print results
prediction = classifier.make_prediction(X_test)
actual_label = 'Positive' if y_test == 1 else 'Negative'
predicted_label = 'Positive' if prediction == 1 else 'Negative'
print(f"Actual: {actual_label}\nPredicted: {predicted_label}")

# Generate training data and labels
training_data, labels = generate_training_data()
X_train, X_test, y_train, y_test = split_data(training_data, labels)

# Evaluate and print training performance
train_accuracy, train_confusion_matrix = classifier.assess_performance(X_train, y_train)
print("\nTraining Accuracy:", train_accuracy)
print("Training Confusion Matrix:")
print(train_confusion_matrix)

# Evaluate and print testing performance
test_accuracy, test_confusion_matrix = classifier.assess_performance(X_test, y_test)
print("\nTesting Accuracy:", test_accuracy)
print("Testing Confusion Matrix:")
print(test_confusion_matrix)

