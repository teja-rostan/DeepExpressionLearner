"""
A program for data and target retrieving in right format used by class_learning.py and reg_learning.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_datatarget(scores_file, delimiter, target_size, problem_type, class_, nn_type):
    """
    Reads a complete file with pandas, splits to data and target and makes wished preprocessing of the target variables.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of columns at the end of scores_file that represent target variables.
    :param nn_type: The type of learning problem for neural networks {class, reg, ord}.
    :param class_: Number of classes (supported in classification problem).
    :return: data that represent the input of neural network, target that represent the output of neural network,
    raw target expressions and classified target.
    """

    """ get data """
    df = pd.read_csv(scores_file, sep=delimiter)

    input_matrix = df.select_dtypes(include=['float64', 'int']).as_matrix()

    """ split data and target """
    data = input_matrix[:, :-target_size]

    if nn_type == "cnn":
        data = data.reshape((-1, 1, 1, data.shape[1]))

    target = input_matrix[:, -target_size:]
    target_class = 0
    target_names = list(df)[-target_size:]

    if problem_type == "class":
        """ Classify raw expression with percentil thresholds or raw value thresholds"""

        # TODO: make without manual fix
        target_class = classification(target, 0.2, 0.8)  # raw value thresshold
        # target_class = classification(target, 0.1, 0.9)  # percentile threshold

        print(target_class.shape)
        target = one_hot_encoder_target(target_class, class_)

    ids = df['ID'].as_matrix()
    return data, target, target_class, ids, target_names


def classification(target, down_per, up_per):
    """
    Changes to a classification problem up to three classes.
    :param target: Raw target values that we want to classify.
    :param down_per: lower threshold as percentile.
    :param up_per: upper threshold as percentile.
    :return: new target with classes.
    """

    new_target = np.zeros(target.shape)
    for i, expression in enumerate(target.T):
        new_expression = np.ones(expression.shape)

        # IF USING PERCENTIL THRESHOLD FOR CLASSIFICATION  # TODO: make without manual fix
        # down_10 = np.percentile(expression, down_per * 100)
        # up_10 = np.percentile(expression, up_per * 100)
        # new_expression -= (expression <= down_10)
        # new_expression += (expression >= up_10)

        # IF USING RAW VALUE THRESHOLD [0, 1] FOR CLASSIFICATION  # TODO: make without manual fix
        new_expression -= (expression < down_per)
        new_expression += (expression > up_per)

        new_target[:, i] = new_expression
    return new_target


def one_hot_encoder_target(y, k):
    """
    One hot encoding of all target variables (can handle multiple targets with k classes).
    :param y: target variables.
    :param k: number of classes per target.
    :return: one hot per target.
    """

    new_y = np.zeros((y.shape[0], y.shape[1] * k))
    # for i in range(y.shape[1]):
    #     col = y[:, i]
    #     enc = OneHotEncoder(sparse=False)
    #     one_hot = enc.fit_transform(col.reshape(-1, 1))
    #     new_y[:, i * k:i * k + k] = one_hot

    for i, s in enumerate(y):
        new_row = np.array(list(s)).astype(np.int)
        b = np.zeros((new_row.size, 3))
        b[np.arange(new_row.size), new_row] = 1
        new_y[i] = b.flatten()
    return new_y
