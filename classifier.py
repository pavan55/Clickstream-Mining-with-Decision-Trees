import pandas as pd
import time
import pickle as pkl
import argparse
import numpy as np
from scipy import stats
import math

# data structure as per autograder
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

def get_label_file(feature_file_name):
    # label file has same name as feature with the addition of _label
    return feature_file_name.split('.')[0] + '_label.csv'

def load_data(train_features_file, test_features_file):
    train_features = pd.read_csv(train_features_file, header=None, delim_whitespace=True)
    train_label_file = get_label_file(train_features_file)
    train_labels = pd.read_csv(train_label_file, header=None, delim_whitespace=True)
    test_features = pd.read_csv(test_features_file, header=None, delim_whitespace=True)
    return (train_features, train_labels[0], test_features)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True)
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=True)
    parser.add_argument('-t', help='output tree filename', required=True)

    args = vars(parser.parse_args())
    return (args['p'], args['f1'], args['f2'], args['o'], args['t'])

def create_leaf_node(train_labels):
    neg_count, pos_count = train_labels.value_counts()
    if neg_count > pos_count:
        leaf_data = 'F'
    else:
        leaf_data = 'T'
    return TreeNode(leaf_data, [])

def prob_log_prob(prob):
    return prob*np.log2(prob)

def calculate_entropy(train_labels):
    length = len(train_labels)
    if(length == 0):
        return 0
    val_count = train_labels.value_counts()
    # value count less than 2 indicates one of the two class labels is missing.
    # So entropy is zero
    if(len(val_count) < 2):
        return 0
    neg_count, pos_count = val_count
    neg_prob = neg_count / float(length)
    pos_prob = pos_count / float(length)
    return -(prob_log_prob(neg_prob) + prob_log_prob(pos_prob))

def calc_entropy_for_attribute(feature_values, train_labels):
    unique_vals = feature_values.unique()
    attribute_entropy = 0
    total_length = feature_values.shape[0]
    attribute_label_splits = []
    attribute_splits = []
    for feature_val in unique_vals:
        # finding all the index where current attribute value is present
        attribute_split = feature_values[feature_values == feature_val]
        attr_split_index = attribute_split.keys()
        attr_split_labels = train_labels[attr_split_index]
        entropy_for_val = calculate_entropy(attr_split_labels)
        attr_split_length = attr_split_labels.shape[0]
        attribute_entropy += (attr_split_length /float(total_length)) * entropy_for_val
        attribute_label_splits.append(attr_split_labels)
        attribute_splits.append(attribute_split)
    return (attribute_entropy, attribute_label_splits, attribute_splits)

def find_max_info_gain_feature(train_features, train_labels, current_entropy, attributes_added):
    maximum_info_gain = 0
    feat_with_max_info_gain = None
    max_attribute_splits = []
    max_attribute_label_splits = []
    # Info read through pandas has column data as first dimension and row data as second dimension.
    # As num of features correspond to num of columns, second dimension length needs to be used.
    total_features = train_features.shape[1]
    for feature_ind in range(total_features):
        if feature_ind not in attributes_added:
            # pandas has column index in the beginning
            attribute_entropy, attribute_label_splits, attribute_splits = calc_entropy_for_attribute(train_features[feature_ind], train_labels)
            info_gain = current_entropy - attribute_entropy
            if (info_gain > maximum_info_gain):
                maximum_info_gain = info_gain
                feat_with_max_info_gain = feature_ind
                max_attribute_label_splits = attribute_label_splits
                max_attribute_splits = attribute_splits
    return (feat_with_max_info_gain, max_attribute_label_splits, max_attribute_splits)

def check_chi_squared_criterion(max_attribute_label_splits, threshold):
    observed_frequencies = []
    total_positive = 0
    total_negative = 0
    for split_ind in range(len(max_attribute_label_splits)):
        attribute_split = max_attribute_label_splits[split_ind]
        neg_count, pos_count = attribute_split[0].value_counts()
        observed_frequencies.append(neg_count)
        observed_frequencies.append(pos_count)
        total_positive += pos_count
        total_negative += neg_count

    total_length = total_positive + total_negative
    expected_frequencies = []
    for i in range(len(max_attribute_label_splits)):
        attribute_split = max_attribute_label_splits[split_ind]
        expected_frequencies.append(total_negative * attribute_split.shape[0] / total_length)
        expected_frequencies.append(total_positive * attribute_split.shape[0] / total_length)
    chi, p = stats.chisquare(observed_frequencies, expected_frequencies)
    return p < threshold


def id3_decision_tree(train_features, train_labels, threshold, attributes_added = [], depth = 0):
    current_entropy = calculate_entropy(train_labels)
    # entropy of zero indicates no further information can be gained by splitting on attribute value
    if(current_entropy <= 0):
        return create_leaf_node(train_labels)
    feat_with_max_info_gain, max_attribute_label_splits, max_attribute_splits = find_max_info_gain_feature(train_features, train_labels, current_entropy, attributes_added)
    if (feat_with_max_info_gain is None):
        return create_leaf_node(train_labels)
    root_node_for_branch = TreeNode(feat_with_max_info_gain, [])
    attributes_added.append(feat_with_max_info_gain)
    if check_chi_squared_criterion(max_attribute_label_splits, threshold):
        return create_leaf_node(train_labels)
    for i in range(0, 5):
        child = id3_decision_tree(max_attribute_label_splits[i], max_attribute_splits[i], attributes_added, depth + 1)
        root_node_for_branch.nodes.append(child)
    return root_node_for_branch

if __name__ == "__main__":
    threshold, train_features_file, test_features_file, test_predictions_file, dtree_file = parse_arguments()
    train_features, train_labels, test_features = load_data(train_features_file, test_features_file)
    start_time = time.time()
    decision_tree = id3_decision_tree(train_features, train_labels, threshold)
    # print("Time taken to create decision tree %f"%(time.time() - start_time))
    # decision_tree.save_tree(dtree_file)
