from pickle import load
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model


all_data = load(open('all_data.pkl' , 'rb'))

train_data = load(open('train_data.pkl' , 'rb'))

test_data = load(open('test_data.pkl' , 'rb'))

all_data = array(all_data)
train_data = array(train_data)
test_data = array(test_data)



def tokenize(sentences):

    t = Tokenizer()

    t.fit_on_texts(sentences)

    return t

def max_len(sentences):

    return max(len(sentence.split()) for sentence in sentences )


def encoding(sentences , max):

    pad_sequences(sentences , maxlen = max , padding = 'post')

eng_tokenizer = tokenize(all_data[:,0])
eng_vocab = len(eng_tokenizer.word_index)
eng_max_sentence = max_len(train_data[:,0])
german_tokenizer = tokenize(all_data[:,1])
german_vocab = len(german_tokenizer.word_index)
german_max_sentence = max_len(train_data[:,1])

def encoding(sentences , max):

    return pad_sequences(sentences , maxlen = max , padding = 'post')

def encoding_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size+1)
		ylist.append(encoded)
	y = array(ylist)

	return y


trainX = encoding(german_tokenizer.texts_to_sequences(train_data[:,1]) , german_max_sentence)
trainY = encoding(eng_tokenizer.texts_to_sequences(train_data[:,0]) , eng_max_sentence)
trainY = encoding_output(trainY , eng_vocab)

testX = encoding(german_tokenizer.texts_to_sequences(test_data[:,1]) , german_max_sentence)
testY = encoding(eng_tokenizer.texts_to_sequences(test_data[:,0]) , eng_max_sentence)
testY = encoding_output(testY , eng_vocab)
