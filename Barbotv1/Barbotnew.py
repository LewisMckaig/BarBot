import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open("Dataset.txt").read()

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

processed_inputs = tokenize_words(file)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:", input_len)
print ("total vocab:", vocab_len)

seq_length = 150
x_data = []
y_data = []

# loop through each input, start at the beginning of the file
# until it reaches the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    out_seq = processed_inputs[i + seq_length]

    # convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

#convert the sequences into a numpy array so the network can use it
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))

start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
