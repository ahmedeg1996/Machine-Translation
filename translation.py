from pickle import load
from the_model import tokenize
from the_model import max_len
from the_model import encoding
from keras.models import load_model

from numpy import array
# load datasets
dataset = array(load(open('all_data.pkl' , 'rb')))

train = array(load(open('train_data.pkl' , 'rb')))

test = array(load(open('test_data.pkl' , 'rb')))



# prepare english tokenizer
eng_tokenizer = tokenize(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_len(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = tokenize(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_len(dataset[:, 1])
# prepare data
trainX = encoding(ger_tokenizer.texts_to_sequences(train[:,1]) , ger_length)
testX = encoding(ger_tokenizer.texts_to_sequences(test[:,1]) , ger_length)

model = load_model('machine_translation_model.h5')
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

source = testX[0]
source = source.reshape((1, source.shape[0]))
print(model.predict(source,verbose=0))[0]
