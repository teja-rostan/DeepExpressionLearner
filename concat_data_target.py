"""
[Prequel program]
get sequences and their ids
    convert ids from DDB to DDB_G where possible
run seq2num
get expressions
join sequences with their expressions
"""
import time
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO


def get_seq_and_id(fasta_file, max_seq_len):
    """ Extracts raw sequence strings and ids to separate files."""

    sequences = []
    record_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq)[-max_seq_len:])
        record_id = str(record.id)
        end = record_id.find('|')
        if end != -1:
            record_id = record_id[:end]
        record_ids.append(record_id)
    data_record_ids = pd.Series({"record_id": record_ids})
    data_sequences = pd.Series({"record_sequence": sequences})
    return data_record_ids, data_sequences


#def convert_ids(old_ids, map_txt, conv_type):
#    """ Performs conversion of IDs with the help of DDB-GeneID-UniProt.txt file."""


#def seq2num():

def main():
    start = time.time()
    arguments = sys.argv[1:]
    max_seq_len = 900 # if shorter, add Nan values at beginning.

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python concat_data_target.py <input_fasta_sequences> <input_expressions> <output_data_target_file> <delimiter>.")
        return

    input_fasta_sequences = arguments[0]
    input_expressions = arguments[1]
    output_data_target_file = arguments[2]
    delimiter = arguments[3]

    data_record_ids, data_sequences = get_seq_and_id(fasta_file, max_seq_len)


    end = time.time() - start
    print("Program has successfully written scores at " + output_data_target_file + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
