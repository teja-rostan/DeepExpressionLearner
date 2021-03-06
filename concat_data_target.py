"""
A program for changing the data file and the target file to a new file that is an input file for neural network learning.

Usage:
python concat_data_target.py <input_fasta_sequences> <input_expressions> <output_data_target_file> <expression_size> <delimiter> <max_seq_len>.

Example of usage:
python concat_data_target.py my_data/data.csv my_data/target.csv my_new_data/ 14 , 200
"""

import time
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
import os


def get_seq_and_id(input_fasta_sequences, max_seq_len):
    """
    Extracts raw sequence strings from fasta input file and their ids to separate dataframes.
    :param input_fasta_sequences: fasta file of sequences for learning.
    :param max_seq_len: minimal length of sequences (longer are cut on the left to that length).
    :return: dataframe of ids of sequences and dataframe of sequences.
    """

    sequences = []
    record_ids = []
    for record in SeqIO.parse(input_fasta_sequences, "fasta"):
        seq = str(record.seq)
        if len(seq) >= max_seq_len:
            sequences.append(seq[-max_seq_len:])
            record_id = str(record.id)
            end = record_id.find('|')
            if end != -1:
                record_id = record_id[:end]
            record_ids.append(record_id)
    data_record_ids = pd.DataFrame({"record_id": record_ids})
    data_sequences = pd.DataFrame({"record_sequence": sequences})
    return data_record_ids, data_sequences


def convert_ids(old_ids, map_txt, conv_type):
    """
    Performs conversion of IDs with the help of DDB-GeneID-UniProt.txt file.
    The function supports conversion from DDB to DDB_G (conv_type=0) or from DDB_G to DDB (conv_type=1).
    :param old_ids: dataframe of the id names before conversion.
    :param map_txt: DDB-GeneID-UniProt.txt.
    :param conv_type: "0" converts from DDB to DDB_G, "1" converts from DDB_G to DDB.
    :return: dataframe of the id names after conversion.
    """

    df = pd.read_csv(map_txt, sep="\t")
    ddb_id = df['DDBDDB ID'].as_matrix()
    ddb_g_id = df['DDB_G ID'].as_matrix()
    map_names = df['Name'].as_matrix()
    old_ids = old_ids.as_matrix()[:, 0]
    new_ids = np.copy(old_ids)
    names = np.empty(old_ids.shape, dtype=object)
    if conv_type == '0':
        for i, old_id in enumerate(old_ids):
            occurrence = list(np.where(ddb_id == old_id)[0])
            if len(occurrence) > 0:
                new_ids[i] = ddb_g_id[occurrence[0]]
                if map_names[occurrence[0]][:12] == ddb_g_id[occurrence[0]]:
                    names[i] = ""
                else:
                    names[i] = str(map_names[occurrence[0]])
    else:
        for i, old_id in enumerate(old_ids):
            occurrence = list(np.where(ddb_g_id == old_id)[0])
            if len(occurrence) > 0:
                new_ids[i] = ddb_id[occurrence[0]]
    data_record_ids = pd.DataFrame({"record_id": new_ids})
    return data_record_ids, names


def seq2one_hot(data_sequences, data_record_ids, max_len):
    """
    Creates a new matrix where N,A,C,G,T get 0,1,2,3,4 integer values. Sequences are padded on the left with 0 as nan.
    After that the method performs OneHot Encoding.
    :param data_sequences: dataframes of raw strings of sequences.
    :param data_record_ids: dataframe of sequences ids.
    :param max_len: minimum length of a sequence (longer are cut on the left to the maxim_len).
    :return: dataframe of sequences that are converted to onehot format.
    """

    # print(data_sequences)
    data_sequences.record_sequence = data_sequences.record_sequence.str.replace('A', '0')
    data_sequences.record_sequence = data_sequences.record_sequence.str.replace('C', '1')
    data_sequences.record_sequence = data_sequences.record_sequence.str.replace('G', '2')
    data_sequences.record_sequence = data_sequences.record_sequence.str.replace('T', '3')

    data = data_sequences.record_sequence.as_matrix()
    new_data = np.zeros((len(data_sequences), max_len * 4))
    bad = []
    for i, s in enumerate(data):
        try:
            new_row = np.array(list(s)).astype(np.int)
            b = np.zeros((new_row.size, 4))
            b[np.arange(new_row.size), new_row] = 1
            new_data[i] = b.flatten()
        except ValueError:
            bad.append(i)
    data_record_ids = data_record_ids.drop(data_record_ids.index[bad])
    new_data = new_data[~np.all(new_data == 0, axis=1)]  # removes zero rows
    data_sequences = pd.DataFrame(new_data, dtype=np.int)
    return data_sequences, data_record_ids


def sort_data_and_get_expressions(input_expressions, delimiter, data_record_ids, data_sequences, output_data_target_file, exps_size, times):
    """
    The program matches and concatenates sequences with their expressions.
    Afterward, the program writes final dataframes in datatarget csv files.
    :param input_expressions: a file with expressions or a directory with files of expressions (multiple expressions per sequence).
    :param delimiter: delimiter of of expression files and a delimiter for a final datatarget file.
    :param data_record_ids: dataframe of ids of sequences.
    :param data_sequences: dataframe of sequences.
    :param output_data_target_file: the path name of a file to write a new datatarget file.
    :param exps_size: number of expressions in target data.
    :param times: if there are multiple times of expressions records (starts with 0h and ends with 24h, step is 2h)
    """

    if os.path.isfile(input_expressions):
        bind_exps = ["", ""]
    else:
        bind_exps = os.listdir(input_expressions)

    for j, bind_exp in enumerate(bind_exps[1:]):
        print(bind_exp)
        df = pd.read_csv(input_expressions + bind_exp, sep=delimiter)
        print(df.columns, exps_size)

        exps = df[df.columns[-exps_size:]]
        col_names = list(np.arange(data_sequences.shape[1]).astype(np.str)) + list(exps)
        exps_m = exps.as_matrix()
        bind_ids = df['ID'].as_matrix()

        sorted_seq_with_exp = []
        new_ids = []
        idx = 0
        for i, bind_id in enumerate(bind_ids):
            occurrence = list(np.where(bind_id == data_record_ids)[0])
            if len(occurrence) > 0:
                sorted_seq_with_exp.append(np.hstack((data_sequences[occurrence[0]], exps_m[i])))
                new_ids.append(bind_id)
                idx += 1
        data_target = pd.DataFrame(sorted_seq_with_exp, index=new_ids)
        output_path = output_data_target_file + "datatarget_" + times[j] + "_" + time.strftime("%d_%m_%Y") + ".csv"
        print(output_path)
        data_target.to_csv(output_path, index_label='ID', header=col_names, delimiter=delimiter)


def main():
    start = time.time()
    arguments = sys.argv[1:]
    map_txt = "DDB2DDB_G/DDB-GeneID-UniProt.txt"
    conv_type = "0"
    times = ['0h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h', '2h', '4h', '6h', '8h']

    if len(arguments) < 6:
        print("Not enough arguments stated! Usage: \n"
              "python concat_data_target.py <input_fasta_sequences> <input_expressions> <output_data_target_file> "
              "<expression_size> <delimiter> <max_seq_len>.")
        return

    input_fasta_sequences = arguments[0]
    input_expressions = arguments[1]
    output_data_target_file = arguments[2]
    exps_size = int(arguments[3])
    delimiter = arguments[4]
    max_seq_len = int(arguments[5])

    data_record_ids, data_sequences = get_seq_and_id(input_fasta_sequences, max_seq_len)
    data_record_ids, names = convert_ids(data_record_ids, map_txt, conv_type)

    data_sequences, data_record_ids = seq2one_hot(data_sequences, data_record_ids, max_seq_len)

    sort_data_and_get_expressions(input_expressions, delimiter, data_record_ids, data_sequences.as_matrix(), output_data_target_file, exps_size, times)

    end = time.time() - start
    print("Program has successfully written scores at " + output_data_target_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
