"""
A program to fit all data in a convolutional neural network amd retrieve a file with motifs.

Usage:
python class_learning.py <datatarget_dir> <output_dir> <name> <delimiter> <target_size> <network_type> <architecture>,
where architecture is:
    *  '3c2f': 3 conv layers and 2 fully connected layers (default)
    *  '2c1f': 2 conv layers and 1 fully connected layer
    *  '1c2f': 1 conv layer and 2 fully connected layers

Examples of usage:
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/get_motifs.py datatarget/ results/ 'test_motifs_cnn_2c1f' , 14 2c1f
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/get_motifs.py datatarget/ results/ 'test_motifs_cnn_3c2f' , 14
"""

import data_target, CNNLearner
import os
import time
import sys
import pandas as pd


def learn_and_get_motives(datatarget_file, delimiter, target_size, architecture="3c2f"):
    """
    Dense connected or convolution Neural network learning and correlation scoring. Learning and predicting one or multiple targets.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :return: rhos and p-values of relative Spearman correlation, predictions and ids of instances that were included
    in learning and testing.
    """

    class_ = 3  # negative expression, no expression, positive expression
    """ Get data and target tables. """
    data, target, target_class, ids, target_names = data_target.get_datatarget(datatarget_file, delimiter, target_size, "class", class_)
    print(data.shape, target.shape)

    """ Neural network architecture initialisation. """
    class_size = class_ * target_size
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    net = CNNLearner.CNNLearner(class_size, architecture, n_hidden_n, "class")

    """ Split to train and test 10-fold Cross-Validation """

    trX = data
    trY = target

    """ Learning and predicting """
    net.fit(trX, trY)

    """ Get motifs """
    motifs = net.get_motifs()
    motifs = motifs.reshape((-1, motifs.shape[-1]))
    motifs_dataframe = pd.DataFrame(motifs)
    motifs_dataframe = motifs_dataframe.round(4)
    return motifs_dataframe


def main():
    start = time.time()
    architecture = "3c2f"
    arguments = sys.argv[1:]

    if len(arguments) < 5:
        print("Error: Not enough arguments stated! Usage: \n"
              "python get_motifs.py <datatarget_dir> <output_dir> <name> <delimiter> <target_size> "
              "<architecture>,\nwhere architecture is:\n"
              " - '3c2f': 3 conv layers and 2 fully connected layers (default)\n"
              " - '2c1f': 2 conv layers and 1 fully connected layer\n"
              " - '1c2f': 1 conv layer and 2 fully connected layers.")
        sys.exit(0)

    datatarget_dir = arguments[0]
    output_dir = arguments[1]
    name = arguments[2]
    delimiter = arguments[3]
    target_size = int(arguments[4])
    if len(arguments) == 6:
        architecture = arguments[5]

    datatarget_list = os.listdir(datatarget_dir)
    print(datatarget_list)

    for row in datatarget_list:
        datatarget_file = datatarget_dir + row
        if os.path.isfile(datatarget_file):
            print(datatarget_file)

            nn_one_file = output_dir + "/motifs_" + name + "_" + row[:-4] + "_" + time.strftime("%d_%m_%Y") + ".csv"

            motifs_dataframe = learn_and_get_motives(datatarget_file, delimiter, target_size, architecture=architecture)
            print(motifs_dataframe)
            motifs_dataframe.to_csv(nn_one_file, sep=delimiter,  float_format='%.4f')

    end = time.time() - start
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()