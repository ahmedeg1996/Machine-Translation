import string
import re
import unicodedata
from pickle import dump
from numpy.random import shuffle


text = open('translation.txt' , 'r' , encoding = 'utf-8').read()


lines = text.strip().split('\n')

pairs = [line.split('\t') for line in lines]


re_print = re.compile('[^%s]' % string.printable)

table = str.maketrans(" " , " " , string.punctuation)
dataset = []
for pair in pairs:

    list_small = []

    for element in pair:

        element = unicodedata.normalize('NFD' , element).encode('ascii' , 'ignore').decode() #this is to normalize the special characters in the german languagethat are uni codded and transform them to ascii to make them english readable by the computer

        element = element.split()

        element = [word.lower() for word in element]

        element = [word.translate(table) for word in element] #remove punctuation from each word in each pair

        element = [re_print.sub("" , word) for word in element] #remove non-printable characters from eachword

        element = [word for word in element if word.isalpha()] #returns false if a word contains any number

        list_small.append(" ".join(element))

    dataset.append(list_small)


dataset = dataset[:10000]

shuffle(dataset)

all_data = open('all_data.pkl' , 'wb')

train_data = open('train_data.pkl' , 'wb')

test_data = open('test_data.pkl' , 'wb')

dump(dataset , all_data)

dump(dataset[:9000] , train_data)

dump(dataset[9000:] , test_data)

all_data.close()
train_data.close()
test_data.close()
