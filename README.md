# Binary Decision Tree Classifier for Breast Cancer Diagnosis

This project, done as part of the CS-540 Intro to AI course at UW Madison, implements a binary decision tree classifier for diagnosing breast cancer. The classifier is trained using the Wisconsin Breast Cancer dataset and provides predictions for unlabeled test data.

## Project Overview

The main goal of this project is to develop a decision tree classifier that can accurately predict whether a breast tumor is benign or malignant based on certain features. The classifier is implemented using the Python programming language and the NumPy library for efficient numerical computations.

The project consists of the following components:

1. `breast-cancer-wisconsin.data`: The training dataset used to train the decision tree classifier. It contains information about various features of breast tumors and their corresponding class labels.

2. `test.txt`: The test dataset containing unlabeled instances for which the decision tree classifier will make predictions.

3. `tree.txt`: A text file that stores the structure of the trained decision tree.

4. `labels.txt`: A text file that contains the predicted class labels for the test instances using the trained decision tree.

5. `pruned_tree.txt`: A text file that stores the pruned version of the decision tree based on the specified target depth.

6. `pruned_predictions.txt`: A text file that contains the predicted class labels for the test instances using the pruned decision tree.

## Usage

To use the decision tree classifier, follow these steps:

1. Ensure that you have the required dependencies installed. This project relies on Python 3 and the NumPy library.

2. Download the project files and place them in a directory.

3. Run the `binary_decision_tree.py` file to train the decision tree classifier and generate the predictions.

4. The predicted class labels for the test instances will be stored in the `labels.txt` file.

5. To visualize the structure of the trained decision tree, refer to the `tree.txt` file.

6. To prune the decision tree to a specified depth, modify the `target_depth` variable in the `binary_decision_tree.py` file and re-run the script. The pruned tree will be stored in the `pruned_tree.txt` file.

7. The predicted class labels using the pruned decision tree will be stored in the `pruned_predictions.txt` file.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project was completed as part of the CS-540 Intro to AI course at UW Madison. The Wisconsin Breast Cancer dataset used in this project is publicly available and can be found at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).
