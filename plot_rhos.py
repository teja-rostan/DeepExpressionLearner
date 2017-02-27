import time
import sys
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

help = \
    """
    Program for plotting correlation results at input_directory_path.

    Usage:
        python plot_rhos.py <input_directory_path> <delimiter>.
"""


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 2:
        print("Not enough arguments stated! Usage: \n"
              "python plot_rhos.py <input_directory_path> <delimiter>.")
        sys.exit(0)

    input_dir = arguments[0]
    delimiter = arguments[1]

    plots_names = []
    plots_time = []
    vals = []
    rows = os.listdir(input_dir)
    print(rows)
    for row in rows:
        if row[:8] == "spearman":
            scores_file = input_dir + row
            plots_names.append(row[9:-15])
            print(scores_file)
            df = pd.read_csv(scores_file, sep=delimiter)
            vals = list(df)[1:]
            matrix = df.as_matrix()
            matrix = matrix[np.arange(0, len(matrix), step=2)][:, 1:]  # get only rhos without name rho
            print("mean matrix", np.nanmean(matrix))
            print(matrix.shape)
            plots_time.append(matrix.flatten())

    plots_time = np.array(plots_time).astype(float)

    plt.figure("Correlation of expressions")

    for i in range(len(plots_time)):
        if i > 3:
            print(i)
            plt.plot(plots_time[i], '--')
        else:
            plt.plot(plots_time[i])

    plots_names = np.array(plots_names)
    plt.legend(np.array(plots_names), loc='lower left')
    plt.xticks(np.arange(len(vals)), vals)
    plt.axis([0, 13, 0, 0.62])
    plt.xlabel('expressions')
    plt.ylabel('spearman correlation')
    plt.grid()

    plt.show()
    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()