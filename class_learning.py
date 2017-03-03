"""
A program to run a classification problem with fully connected neural network or convolutional neural network.

Usage:
python class_learning.py <datatarget_dir> <output_dir> <name> <delimiter> <target_size> <network_type> <architecture>,
where network_type is:
 - 'nn' (fully connected network)
 - 'cnn' (convolutional neural network),
where architecture is:
 - in case of nn_type='cnn':
    *  '3c2f': 3 conv layers and 2 fully connected layers (default)
    *  '2c1f': 2 conv layers and 1 fully connected layer
    *  '1c2f': 1 conv layer and 2 fully connected layers
 - in case of nn_type='nn':
    *  integer > 0 representing the number of hidden layers (default=3).

Examples of usage:
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_cnn_2c1f' , 14 cnn 2c1f
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_cnn_3c2f' , 14 cnn
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_nn_3' , 14 nn
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_nn_4' , 14 nn 4
"""

import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
import data_target, CNNLearner, NNLearner
import os
import time
import sys
import pandas as pd
from sklearn.metrics import accuracy_score


def learn_and_score(datatarget_file, delimiter, target_size, nn_type, architecture):
    """
    Dense connected or convolution Neural network learning and correlation scoring. Learning and predicting one or multiple targets.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :param nn_type: network type: 'cnn' if convolutional neural network or 'nn' if fully connected neural network.
    :param architecture: description of 'cnn' ('3c2f', '2c1f', '1c2f') or number of hidden layers in 'nn'.
    :return: rhos and p-values of relative Spearman correlation, predictions and ids of instances that were included
    in learning and testing.
    """

    class_ = 3  # negative expression, no expression, positive expression
    """ Get data and target tables. """
    data, target, target_class, ids, target_names = data_target.get_datatarget(datatarget_file, delimiter, target_size, "class", class_, nn_type)
    print(data.shape, target.shape)

    """ Neural network architecture initialisation. """
    class_size = class_ * target_size
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    if nn_type == 'cnn':
        if architecture == "0":
            architecture = "3c2f"
        net = CNNLearner.CNNLearner(class_size, architecture, n_hidden_n, "class")

    elif nn_type == 'nn':
        architecture = int(architecture)
        if architecture == 0:
            architecture = 3
        net = NNLearner.NNLearner(data.shape[1], class_size, architecture, n_hidden_n)

    nn_scores = []

    probs = np.zeros((target.shape[0], target_size*2))
    ids_end = np.zeros((target.shape[0], 1)).astype(str)

    """ Split to train and test 10-fold Cross-Validation """
    skf = KFold(target.shape[0], n_folds=10, shuffle=True)
    idx = 0
    for train_index, test_index in skf:
        trX, teX = data[train_index], data[test_index]
        trY, teY = target[train_index], target[test_index]

        """ Learning and predicting """
        net.fit(trX, trY)
        prY = net.predict(teX)

        maj = majority(trY, teY, class_)
        nn_score, true_p, pred_p = score_ca_and_prob(prY, teY, class_)
        print("Accuracy score of majority:", np.mean(maj), "|Accuracy score of cnn:", np.mean(nn_score))
        nn_scores.append(nn_score)

        """ Storing results... """
        probs[idx:idx + len(teY), :target_size] = true_p
        probs[idx:idx + len(teY), target_size:] = pred_p
        ids_end[idx:idx + len(teY), 0] = ids[test_index].flatten()
        idx += len(teY)

    rhos = []
    p_values = []
    for i in range(target_size):
        print(probs[:, i], probs[:, i + target_size])
        print(np.isnan(probs[:, i]).any(), np.isnan(probs[:, i+target_size]).any())
        rho, p_value = spearmanr(probs[:, i], probs[:, i + target_size])
        rhos.append(rho)
        p_values.append(p_value)
    print("Accuracy score of cnn:", np.mean(nn_scores))
    return rhos, p_values, np.around(probs, decimals=2), ids_end, target_names


def majority(tr_y, te_y, k):
    """
    Classification accuracy of majority classifier.
    :param tr_y: training set in one hotformat.
    :param te_y: test set in onehot format.
    :param k: number of classes.
    :return: majority score.
    """
    mc = []
    for i in range(int(tr_y.shape[1] / k)):
        col_train = tr_y[:, i * k:i * k + k]
        col_test = te_y[:, i * k:i * k + k]

        col_train = np.argmax(col_train, axis=1)
        col_test = np.argmax(col_test, axis=1)
        counts = np.bincount(col_train)
        predicted = np.argmax(counts)

        maj = np.mean(col_test == predicted)
        mc.append(maj)
    return mc


def score_ca_and_prob(y_predicted, y_true, k):
    """
    Multi-target scoring with classification accuracy.
    :param y_predicted: predicted values in onehot format.
    :param y_true: true values in onehot format.
    :param k: number of classes.
    :return: accuracy score, true values, predicted values.
    """
    true_prob = []
    pred_prob = []
    all_ca = []
    for i in range(int(y_true.shape[1] / k)):
        col_true = y_true[:, i * k:i * k + k]
        col_predicted = y_predicted[:, i * k:i * k + k]

        col_true = np.argmax(col_true, axis=1)
        col_predicted = np.argmax(col_predicted, axis=1)

        pred_prob.append(col_predicted)
        true_prob.append(col_true)

        ca = accuracy_score(col_true, col_predicted)
        all_ca.append(ca)
    return all_ca, np.array(true_prob).T, np.array(pred_prob).T


def main():
    start = time.time()
    architecture = "0"
    arguments = sys.argv[1:]

    if len(arguments) < 6:
        print("Error: Not enough arguments stated! Usage: \n"
              "python class_learning.py <datatarget_dir> <output_dir> <name> <delimiter> <target_size> <network_type>"
              " <architecture>,\nwhere network_type is:\n - 'nn' (fully connected network)\n - 'cnn' "
              "(convolutional neural network), \nwhere architecture is:\n - in case of nn_type='cnn':\n"
              "   *  '3c2f': 3 conv layers and 2 fully connected layers (default)\n"
              "   *  '2c1f': 2 conv layers and 1 fully connected layer\n"
              "   *  '1c2f': 1 conv layer and 2 fully connected layers."
              "\n - in case of nn_type='nn':\n"
              "   *  integer > 0 representing the number of hidden layers (default=3).")
        sys.exit(0)

    datatarget_dir = arguments[0]
    output_dir = arguments[1]
    name = arguments[2]
    delimiter = arguments[3]
    target_size = int(arguments[4])
    nn_type = arguments[5]
    if len(arguments) == 5:
        architecture = arguments[6]

    nn_scores = []
    col_names = []
    corr_scores = []
    target_names = []

    datatarget_list = os.listdir(datatarget_dir)
    print(datatarget_list)

    for row in datatarget_list:
        datatarget_file = datatarget_dir + row
        if os.path.isfile(datatarget_file):
            col_names.append(row[:-4])
            print(datatarget_file)

            rhos, p_values, probs, ids, target_names = learn_and_score(datatarget_file, delimiter, target_size, nn_type,
                                                                       architecture)
            corr_scores.append(rhos)
            corr_scores.append(p_values)
            nn_scores.append(np.hstack([probs, ids]))

    nn_one_file = output_dir + "/spearman_" + name + "_" + time.strftime("%d_%m_%Y") + ".csv"
    df = pd.DataFrame(data=np.array(corr_scores), index=(['rho', 'p-value'] * len(col_names)), columns=target_names)
    df = df.round(decimals=4)
    df.to_csv(nn_one_file, sep=delimiter)

    nn_probs_file = output_dir + "/prob_" + name + "_" + time.strftime("%d_%m_%Y") + ".csv"
    print(nn_probs_file)
    df = pd.DataFrame(data=np.vstack(nn_scores))
    df = df.round(decimals=2)
    df.to_csv(nn_probs_file, sep=delimiter)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()