# DeepExpressionLearner

todo...

## Example in practice
The example represents the whole procedure, from generating the sequences and expressions to learning the neural network that predicts these expressions 
 (negative expression, no expression or positive expression). In the example we also plot the results of multiple neural network models. 
 Finally, with the architectures of the convoluional neural network that we are satisfied we fit the model with the whole set of data 
 (no splitting for learning and testing) and retrieve the motifs.

### Generate an example of data and target files for learning
Returns two generated files, one is a data file of sequences in fasta format and the second is a target file with expressions.

#### Usage:
python util.py

#### Parameters setting in a code:
generate_signal_data(M=14, N=5000, L=200, p=0.2, motif1=motif1, mean1=0, var1=1, motif2=motif2, mean2=0.5, var2=2, priors={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
M: number of expression targets
N: number of examples
L: length of a sequence
p: thresholds: negative expression if r < p, positive expression if r > (1 - p), no expression if else. 
mean1, var1, motif1: in case of negative expression, the subsequence at the mean mean1 and variance var1 is replaced with motif1.
mean2, var2, motif2: in case of positive expression, the subsequence at the mean mean2 and variance var2 is replaced with motif2. 


### Get data and target in right format for learning
The files input_fasta_sequences and input_expressions generated with util.py can be the input files for concat_data_target.py (or your own fasta file and expressions file).
The program returns a single file where the sequences are one-hot encoded and 
the file contains all the examples that have the matching ids in expression file and in fasta file. These examples are concatenated.
The output file is meant as an input file to a class_learning.py program.

#### Usage: 
python concat_data_target.py <input_fasta_sequences> <input_expressions> <output_data_target_file> <expression_size> <delimiter> <max_seq_len>.

#### Example of usage:
python concat_data_target.py my_data/data.fasta my_data/target.csv my_new_data/ 14 , 200

### Classification learning with neural network
The program needs a datatarget_dir with files of datatargets to get the data columns and target columns (target columns are classified to three classes). 
The program then learns a neural network model with 10-fold cross validation (during learning and predicting the program prints out the classification accuracy
of a neural network and of a majority classifier for a reference.
Afterward, the relative spearman correlation is calculated. The results of correlation are written at the output_dir that starts with spearman_*. 
Another file probs_* is written at output_dir that contains the true classes along side with the predicted classes.

#### Usage:
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
    
#### Examples of usage:

THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_class_cnn_2c1f' , 14 cnn 2c1f
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_class_cnn_3c2f' , 14 cnn
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_class_nn_3' , 14 nn
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/class_learning.py datatarget/ results/ 'test_class_nn_4' , 14 nn 4


### Plot all correlation results at input_directory_path.
Plots all the spearman_* files in the input_directory_path directory.

Usage:
python plot_rhos.py <input_directory_path> <delimiter>.
 

### Get motifs by fitting with all data (no splitting for learning and testing purposes)
For the plotted results that we are satisfied, we retrieve the motifs with this program. The motifs are written in a file motifs_*.

#### Usage:
python class_learning.py <datatarget_dir> <output_dir> <name> <delimiter> <target_size> <network_type> <architecture>,
where architecture is:
    *  '3c2f': 3 conv layers and 2 fully connected layers (default)
    *  '2c1f': 2 conv layers and 1 fully connected layer
    *  '1c2f': 1 conv layer and 2 fully connected layers

#### Examples of usage:

THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/get_motifs.py datatarget/ results/ 'test_class_cnn_2c1f' , 14 2c1f
THEANO_FLAGS='floatX=float32,device=gpu2,lib.cnmem=1' python code/get_motifs.py datatarget/ results/ 'test_class_cnn_3c2f' , 14

