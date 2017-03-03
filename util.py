import numpy as np


def generate_sequence(priors, L):
    """
    Generate a random sequence.

    :param priors:
        Prior probabilities of keys.
    :param L:
        Length of a sequence.
    :return:
    """
    keys = list(priors.keys())
    csum = np.cumsum([priors[ky] for ky in keys])
    seq = ""
    for l in range(L):
        seq += keys[np.argmax(np.random.rand() < csum)]
    return seq


def insert_motif(seq, motif, mean=0, var=3):
    """
     Insert a motif to sequences.

    :param seq:
        Initial sequence.
    :param motif:
        Motif to insert.
    :param mean:
        Mean position around center.
    :param var:
        Positional variance.
    :return:
        String with inserted motif.
    """
    j = int(len(seq)/2 + np.random.normal(mean, var))
    return (seq[:j] + motif + seq[j+len(motif):])[:len(seq)]


def generate_data(N, L, p, motif1, mean1, var1, motif2, mean2, var2, priors, seed=None):
    """
    Generate random sequences with motif.

    :param N:
        Number of sequences.
    :param L:
        Length of sequences.
    :param p:
        Probability of motif insertion (positive class).
    :param motif:
        Motif to insert.
    :param mean:
        Mean potisioning.
    :param var:
        Positional variance
    :param priors:
        Prior probability of insertion.
    :param seed:
        Random seed.
    :return:
        data
            List of sequences.
        y
            Classes.
    """

    if seed is not None:
        np.random.seed(seed)
    data, y = list(), list()
    for n in range(N):
        seq = generate_sequence(priors, L)
        r = np.random.rand()
        if r < p:
            seq = insert_motif(seq, motif1, mean=mean1, var=var1)
        if r > 1 - p:
            seq = insert_motif(seq, motif2, mean=mean2, var=var2)
        data.append(seq)
        y.append(r)
    y = np.array(y)
    return data, y


def generate_signal_data(M, N, L, p, motif1, mean1, var1, motif2, mean2, var2, priors, seed=None):
    """
    Generate multi-output (vector-output) data.

    :param M:
        Signal length.

    :param N: See generate_data.
    :param L: See generate_data.
    :param p: See generate_data.
    :param motif1: See generate_data.
    :param mean1: See generate_data.
    :param var1: See generate_data.
    :param motif2: See generate_data.
    :param mean2: See generate_data.
    :param var2: See generate_data.
    :param priors: See generate_data.
    :param seed: See generate_data.
    :return:
        data
            List of sequences.
        Y
            Multi-output regression matrix.
        cls
            Class from which the line was drawn {-1, 0, 1}.
    """
    Y = np.zeros((N, M))
    data, r = generate_data(N, L, p, motif1, mean1, var1, motif2, mean2, var2, priors, seed)
    cls = np.zeros((len(r), ))  # Line coefficients
    cls[r < p] = -1
    cls[r > 1 - p] = 1

    # Generate random lines as signals
    for i in range(N):
        k = cls[i] * np.random.rand()
        n = np.random.rand() * 0.2
        noise = np.random.rand(M).ravel() * 0.1
        Y[i, :] = k * np.linspace(-1, 1, M) + n + noise

    return data, Y, cls


if __name__ == "__main__":

    # data, y = generate_data(N=5000, L=200, p=0.2, motif1="GGGGGG", mean1=0, var1=1, motif2="AAAAAA", mean2=0.5,
    #                         var2=2, priors={"A": 0.25, "C": 0.25, "G": 0.25, "T":0.25})
    #
    # print(len(data))
    # print(y)
    # f = open('data.csv', 'w')
    # for i, d in enumerate(data):
    #     f.write(">DDB_G" + str(i) + '\n')  # python will convert \n to os.linesep
    #     f.write(d + '\n')
    #     f.write('\n')
    # f.close()
    # f = open('target.csv', 'w')
    # f.write('ID, val\n')
    # for i, d in enumerate(y):
    #     f.write("DDB_G" + str(i) + "," + str(d) + '\n')
    # f.close()

    # Just for show; delete plotting
    import matplotlib.pyplot as plt

    motif1 = "GGGGGG"
    motif2 = "AAAAAA"
    data, Y, cls = generate_signal_data(M=14, N=5000, L=200, p=0.2, motif1=motif1, mean1=0, var1=1, motif2=motif2, mean2=0.5,
                            var2=2, priors={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})

    print(Y.shape)
    print(len(data))
    f = open('data.csv', 'w')
    for i, d in enumerate(data):
        f.write(">DDB_G" + str(i) + '\n')  # python will convert \n to os.linesep
        f.write(d + '\n')
        f.write('\n')
    f.close()
    f = open('target.csv', 'w')
    f.write('ID,val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14\n')  # columns of an ID and targets
    for i, d in enumerate(Y):
        f.write("DDB_G" + str(i))
        for j in d:
            f.write("," + str(np.around(j, decimals=2)))
        f.write("\n")
    f.close()

    plt.figure()
    for y, c in zip(Y, cls):
        color = {-1: "red", 0: "gray", 1: "green"}[c]
        plt.plot(y, color=color)
    plt.text(12, 0.5, motif1, color="green")
    plt.text(12, -0.5, motif2, color="red")
    plt.xlabel("Time point")
    plt.ylabel("Value")
    plt.savefig("random_signals.png", bbox_inches="tight")
    plt.show()


