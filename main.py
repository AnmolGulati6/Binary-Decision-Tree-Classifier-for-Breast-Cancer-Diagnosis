# Written by Anmol Gulati
import numpy as np


# Helper functions
def entropy(data):
    ''' Calculates the entropy of a given dataset'''
    entropy = 0
    count = len(data)  # total number of instances
    n2 = np.sum(data[:, -1] == 2)  # number of k1
    n4 = np.sum(data[:, -1] == 4)  # number of k2
    if n2 == 0 or n4 == 0:
        return 0
    else:
        for n in [n2, n4]:
            p = n / count
            entropy += - (p * np.log2(p))
        return entropy


def infogain(data, feature, threshold):
    ''' Calculates the information gain for a given feature and threshold'''
    count = len(data)
    a = 0
    b = 0
    c = 0
    d = 0
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    proportion_d1 = len(d1) / count
    proportion_d2 = len(d2) / count
    return entropy(data) - proportion_d1 * entropy(d1) - proportion_d2 * entropy(d2)


def get_best_split(data, feature_list, threshold_list):
    ''' Calculates Max Info Gain, computes the threshold and returns the feature, threshold, and predictions for left and right nodes'''
    c = len(data)

    # c0 is the number of instances with class label 2
    c0 = sum(b[-1] == 2 for b in data)

    # if all instances have class label 2, return 2
    # else if all instances have class label 4, return 4
    if c0 == c: return 2, None, None, None
    if c0 == 0: return 4, None, None, None

    # compute possible information gain for all features and thresholds
    # pairwise combinations
    ig = [[infogain(
        data, feature, threshold) for threshold in threshold_list] for feature in feature_list]

    # convert ig to numpy array
    ig = np.array(ig)

    # find the maximum information gain
    max_ig = max(max(i) for i in ig)

    # if max_ig is 0, return 2 if c0 >= c - c0, else return 4
    # remember c0 is the number of instances with class label 2
    # and c - c0 is the number of instances with class label 4
    if max_ig == 0:
        if c0 >= c - c0:
            return 2, None, None, None
        else:
            return 4, None, None, None

    # can also return max_ig in case you need it for debugging

    # find the index of the maximum information gain
    idx = np.unravel_index(np.argmax(ig, axis=None), ig.shape)

    # return the feature, threshold, and predictions for left and right nodes
    feature, threshold = feature_list[idx[0]], threshold_list[idx[1]]

    # binary split: split the data into two parts based on the threshold
    dl = data[data[:, feature - 1] <= threshold]
    dr = data[data[:, feature - 1] > threshold]

    # get the number of instances with class label 2 and 4 in the left node
    dl_n2 = np.sum(dl[:, -1] == 2)
    dl_n4 = np.sum(dl[:, -1] == 4)

    # if the number of instances with class label 2 is greater than or equal to 4, predict 2
    if dl_n2 >= dl_n4:
        dl_prediction = 2
    else:
        # else predict 4
        dl_prediction = 4

    # get the number of instances with class label 2 and 4 in the left node
    dr_n2 = np.sum(dr[:, -1] == 2)
    dr_n4 = np.sum(dr[:, -1] == 4)

    # if the number of instances with class label 2 is greater than or equal to 4, predict 2
    if dr_n2 >= dl_n4:
        dr_prediction = 2
    else:
        # else predict 4
        dr_prediction = 4
    return feature, threshold, dl_prediction, dr_prediction


class Node:
    def __init__(self, feature=None, threshold=None, l_prediction=None, r_prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.l_prediction = l_prediction  # prediction for left subtree
        self.r_prediction = r_prediction  # prediction for right subtree
        self.l = None  # left child or left subtree
        self.r = None  # right child or right subtree`
        self.correct = 0  # number of correct predictions


def split(data, node):
    # split the data into two parts based on the threshold
    feature, threshold = node.feature, node.threshold
    d1 = data[data[:, feature - 1] <= threshold]
    d2 = data[data[:, feature - 1] > threshold]
    return (d1, d2)


def create_tree(data, node, feature_list):
    ''' Recursively creates the tree'''
    d1, d2 = split(data, node)
    f1, t1, l1_prediction, r1_prediction = get_best_split(d1, feature_list, threshold_list)
    f2, t2, l2_prediction, r2_prediction = get_best_split(d2, feature_list, threshold_list)
    if t1 == None:
        node.l_pre = f1
    else:
        node.l = Node(f1, t1, l1_prediction, r1_prediction)
        create_tree(d1, node.l, feature_list)
    if t2 == None:
        node.r_pre = f2
    else:
        node.r = Node(f2, t2, l2_prediction, r2_prediction)
        create_tree(d2, node.r, feature_list)


def maxDepth(node):
    ''' Calculates the maximum depth of the tree'''
    if node is None:
        return 0;
    else:
        left_depth = maxDepth(node.l)
        right_depth = maxDepth(node.r)

        return max(left_depth, right_depth) + 1


def expand_root(data, feature_list, threshold_list):
    ''' Expands the root node'''
    feature, threshold, dl, dr = get_best_split(
        data, feature_list, threshold_list)
    root = Node(feature, threshold)
    # first split
    data1, data2 = split(data, root)
    create_tree(data, root, feature_list)
    return root


def tree_prediction(node, x):
    ''' Predicts the class label for a single instance (test data)'''
    feature = node.feature
    threshold = node.threshold
    l_prediction = node.l_prediction
    r_prediction = node.r_prediction
    l = node.l
    r = node.r

    # assume threshold is 0.5
    # and x[feature-1] is the value of the feature for the instance x
    # if this value is 0.7
    if x[feature - 1] <= threshold:
        if l_prediction == x[-1]:
            node.correct += 1
        if l == None:
            return l_prediction
        else:
            return tree_prediction(l, x)
    else:
        if r_prediction == x[-1]:
            node.correct += 1
        if r == None:
            return r_prediction
        else:
            return tree_prediction(r, x)


def print_tree(node, f, prefix=''):
    fea = node.feature
    t = node.threshold
    l_pre = node.l_prediction
    r_pre = node.r_prediction
    l = node.l
    r = node.r
    if l == None:
        f.write(prefix + 'if (x' + str(fea) + ') <= ' + str(t) + ') return ' + str(l_pre) + '\n')
    else:
        f.write(prefix + 'if (x' + str(fea) + ') <= ' + str(t) + ')\n')
        print_tree(l, f, prefix + ' ')
    if r == None:
        f.write(prefix + 'else return ' + str(r_pre) + '\n')
    else:
        f.write(prefix + 'else\n')
        print_tree(r, f, prefix + ' ')


def prune(node, depth):
    ''' Prunes the tree to the specified depth'''
    if depth == 1:
        node.l = None
        node.r = None
    else:
        if node.l != None:
            prune(node.l, depth - 1)
        if node.r != None:
            prune(node.r, depth - 1)


# Adjust the following parameters by yourself

target_depth = 6

# Load the training data
with open('breast-cancer-wisconsin.data', 'r') as f:
    data_raw = [l.strip('\n').split(',') for l in f if '?' not in l]
data = np.array(data_raw).astype(int)

threshold_list = range(1, 11)
part_one_feature = [2]
feature_list = [9, 3, 7, 2, 4, 8]

feature, threshold, dl, dr = get_best_split(
    data, feature_list, threshold_list)

n2 = np.sum(data[:, -1] == 2)
n4 = np.sum(data[:, -1] == 4)
print('Number of positive instances in training set (class label 2): ', n2)
print('Number of negative instances in training set (with class label 4): ', n4)

print('initial entropy at the root before the split: ', entropy(data))

most = 0
thresh = None
feature = part_one_feature[0]
for threshold in threshold_list:
    compare = infogain(data, feature, threshold)
    if compare > most:
        thresh = threshold
        most = compare

print('Number of positive instances in training set below threshold',
      np.sum(data[data[:, feature - 1] <= thresh][:, -1] == 2))
print('Number of positive instances in training set above threshold: ',
      np.sum(data[data[:, feature - 1] > thresh][:, -1] == 2))
print('Number of negative instances in training set below threshold',
      np.sum(data[data[:, feature - 1] <= thresh][:, -1] == 4))
print('Number of negative instances in training set above threshold',
      np.sum(data[data[:, feature - 1] > thresh][:, -1] == 4))

print('Information gain after split:', infogain(data, feature, thresh))

# Binary Decision Tree
root = expand_root(data, feature_list, threshold_list)
with open('tree.txt', 'w') as f:
    print_tree(root, f)

print('Maximum depth of the tree: ', maxDepth(root))

# Load test data
with open('test.txt', 'r') as f:
    unlabeled = [l.strip('\n').split(',') for l in f if '?' not in l]
arr = np.array(unlabeled).astype(int)
predictions = [tree_prediction(root, x) for x in arr]
with open('labels.txt', 'w') as f:  # import labels on test data
    f.write(','.join([str(x) for x in predictions]))

# prune the tree to the specified depth and store the pruned tree in a file
prune(root, target_depth)
with open('pruned_tree.txt', 'w') as f:
    print_tree(root, f)

# get all the predictions on the pruned tree
predictions = [tree_prediction(root, x) for x in arr]
# store the predictions in a file
with open('pruned_predictions.txt', 'w') as f:
    f.write(','.join([str(x) for x in predictions]))
