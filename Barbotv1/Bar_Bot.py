from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

# Prefining the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='Dataset.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
# Change weights value to have the bot just generate lyrics instead of train.
# Example : ap.add_argument('-weights', default='checkpoint_layer_2_hidden_500_epoch_70.hdf5') -- will load the weights at epoch 70
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# method for preparing the training data
def load_data(data_dir, seq_length):
	data = open("Dataset.txt").read()
	chars = list(set(data))
	VOCAB_SIZE = len(chars)

	print('Data length: {} characters'.format(len(data)))
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

	ix_to_char = {ix:char for ix, char in enumerate(chars)}
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

    #sets the x and y to the shape of the data
	X = np.zeros((len(data)//seq_length, seq_length, VOCAB_SIZE))
	y = np.zeros((len(data)//seq_length, seq_length, VOCAB_SIZE))
    
    
	for i in range(0, len(data)//seq_length):
		X_sequence = data[i*seq_length:(i+1)*seq_length]
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			input_sequence[j][X_sequence_ix[j]] = 1.
			X[i] = input_sequence

		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
	return X, y, VOCAB_SIZE, ix_to_char

# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network, ammount of layers can be changed with the layernum param
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
# output layer
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

  # method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
	# starting with random character from the dataset
	ix = [np.random.randint(vocab_size)]
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocab_size))
	for i in range(length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		print(ix_to_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

if not WEIGHTS == '':
  model.load_weights(WEIGHTS)
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
  nb_epoch = 0

# perform training if no weights are specified
if args['mode'] == 'train' or WEIGHTS == '':
  while True:
    print('\n\nEpoch: {}\n'.format(nb_epoch))
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    if nb_epoch % 10 == 0:
      model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
  # Loading the trained weights
  model.load_weights(WEIGHTS)
  generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  print('\n\n')
else:
  print('\n\nNothing to do!')
  
