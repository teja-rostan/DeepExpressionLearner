"""
Main program to run a regression problem with fully connected neural network or convolutional neural network.
"""
import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
import data_target, CNNLearner, NNLearner_reg
import os
import time
import sys
import pandas as pd


def learn_and_score(datatarget_file, delimiter, target_size, nn_type, architecture):
    """
    Dense connected or covnolutional Neural network learning and correlation scoring. Learning and predicting one target
    per time on balanced or unbalanced data.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :return: rhos and p-values of relative Spearman correlation, predictions and ids of instances that were included
    in learning and testing.
    """

    """ Get data and target tables. """
    data, target, target_class, ids, target_names = data_target.get_datatarget(datatarget_file, delimiter, target_size, "reg", 1)

    """ Neural network architecture initialisation. """
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    if nn_type == 'cnn':
        if architecture == "0":
            architecture = "3c2f"
        net = CNNLearner.CNNLearner(target_size, architecture, n_hidden_n, "class")

    elif nn_type == 'nn':
        architecture = int(architecture)
        if architecture == 0:
            architecture = 3
        net = NNLearner_reg.NNLearner_reg(data.shape[1], target_size, architecture, n_hidden_n)
    nn_scores = []

    probs = np.zeros((target.shape[0], target_size*2))
    ids_end = np.zeros((target.shape[0], 1)).astype(str)

    """ Split to train and test 10-fold Cross-Validation """
    skf = KFold(target.shape[0], n_folds=10, shuffle=True)
    idx = 0
    for train_index, test_index in skf:
        # trX, teX = data, data  # FOR OVERFITTING!!!
        trX, teX = data[train_index], data[test_index]
        # trY, teY = target, target  # FOR OVERFITTING!!!
        trY, teY = target[train_index], target[test_index]

        # print(trX.shape, trY.shape, teX.shape, teY.shape)
        """ Learning and predicting """
        net.fit(trX, trY)
        prY = net.predict(teX)

        ms = mean_score(trY, teY)
        nn_score = rmse_score(prY, teY)
        print("RMSE of mean score:", np.mean(ms), "|RMSE of cnn:", np.mean(nn_score))

        nn_scores.append(np.mean(nn_score))

        """ Storing results... """
        probs[idx:idx + len(teY), :target_size] = teY
        probs[idx:idx + len(teY), target_size:] = prY
        ids_end[idx:idx+len(teY), 0] = ids[test_index].flatten()
        # ids_end[idx:idx+len(teY), 0] = ids.flatten()  # FOR OVERFITTING!!!
        idx += len(teY)
        # break   # FOR OVERFITTING!!!
    rhos = []
    p_values = []
    for i in range(target_size):
        rho, p_value = spearmanr(probs[:, i], probs[:, i+target_size])
        rhos.append(rho)
        p_values.append(p_value)
    print("RMSE of cnn:", np.mean(nn_scores))
    return rhos, p_values, np.around(probs, decimals=2), ids_end, target_names


def rmse_score(y_predicted, y_true):
    """ Computes root mean squared error on every target variable separately. """

    results = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        col_predicted = y_predicted[:, i]
        results.append(np.sqrt(np.sum(np.square(col_true - col_predicted)) / col_true.size))
    return results


def mean_score(tr_y, te_y):
    """ Computes mean value (Mean learner) on every target variable separately and scores with RMSE."""

    predicted = np.ones(te_y.shape)
    for i in range(tr_y.shape[1]):
        predicted[:, i] *= np.median(tr_y[:, i])
    return rmse_score(predicted, te_y)


def main():
    start = time.time()
    architecture = "0"
    arguments = sys.argv[1:]

    if len(arguments) < 6:
        print("Error: Not enough arguments stated! Usage: \n"
              "python reg_learning.py <datatarget_dir> <output_dir> <name> <delimiter> <target_size> <network_type>"
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