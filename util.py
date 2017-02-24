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

data, y = generate_data(N=5000, L=200, p=0.2, motif1="GGGGGG", mean1=0, var1=1, motif2="AAAAAA", mean2=0.5, var2=2, priors={"A": 0.25, "C": 0.25, "G": 0.25, "T":0.25})
f = open('data.csv', 'w')
for i, d in enumerate(data):
    f.write(">DDB_G" + str(i) + '\n')  # python will convert \n to os.linesep
    f.write(d + '\n')
    f.write('\n')
f.close()
f = open('target.csv', 'w')
f.write('ID, val\n')
for i, d in enumerate(y):
    f.write("DDB_G" + str(i) + "," + str(d) + '\n')
f.close()