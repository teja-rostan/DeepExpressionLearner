import pandas as pd
import time
import sys
from scipy.stats import spearmanr
import data_target
import os
import numpy as np


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 5:
        print("Error: Not enough arguments stated! Usage: \n"
              "python class_learning.py <probs_file> <datatarget_dir> <output_dir> <delimiter> <target_size>")
        sys.exit(0)

    probs_file = arguments[0]
    datatarget_dir = arguments[1]
    output_dir = arguments[2]
    delimiter = arguments[3]
    target_size = int(arguments[4])

    probs = pd.read_csv(probs_file, delimiter=delimiter)
    probs = probs.select_dtypes(include=['float64']).as_matrix()
    datatarget_list = os.listdir(datatarget_dir)
    data, target, target_class, ids, target_names = data_target.get_datatarget(datatarget_dir + datatarget_list[1], delimiter, target_size, "reg")
    step = len(data)
    col_len = len(datatarget_list) - 1

    probs_dir, probs_file = os.path.split(probs_file)

    corr_scores = []
    for i in range(col_len):
        probs_one = probs[i*step:i*step + step]

        rhos = []
        p_values = []

        for j in range(target_size):
            rho, p_value = spearmanr(probs_one[:, j], probs_one[:, j + target_size])
            rhos.append(rho)
            p_values.append(p_value)
        corr_scores.append(rhos)
        corr_scores.append(p_values)

    nn_one_file = output_dir + "/spearman_" + probs_file[5:-15] + "_" + time.strftime("%d_%m_%Y") + ".csv"
    df = pd.DataFrame(data=np.array(corr_scores), index=(['rho', 'p-value'] * col_len), columns=target_names)
    df = df.round(decimals=4)
    df.to_csv(nn_one_file, sep=delimiter)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()