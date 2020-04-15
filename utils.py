# utils.py

import torch
from torchtext import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    def load_data(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

    def load_data_cls(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        # train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_examples = []
        for i in train_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            train_examples.append(example)


        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        # test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_examples = []
        for i in test_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label],datafields)
            test_examples.append(example)

        test_data = data.Dataset(test_examples, datafields)

        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            # val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_examples = []
            for i in val_df.values.tolist():
                label = i[1]
                text = i[0]
                text = text.split(' ')
                category = text[-5::]
                text_str = text[0:-5]
                text_str = (' ').join(text_str)
                text_str = clean_str(text_str)
                text_str = text_str.split(' ')
                text = ['<cls>'] + text_str + ['<sep>'] + category
                text = (' ').join(text)
                example = data.Example.fromlist([text, label],datafields)
                val_examples.append(example)
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        # self.train_iterator = data.BucketIterator(
        #     (train_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False,
        #     shuffle=True)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))

    def load_data_cls_wo_valid(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        # train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_test_examples = []
        train_examples = []
        for i in train_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            train_examples.append(example)
            train_test_examples.append(example)

        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        # test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_examples = []
        for i in test_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label],datafields)
            test_examples.append(example)
            train_test_examples.append(example)

        val_data = data.Dataset(test_examples, datafields)

        train_test_data = data.Dataset(train_test_examples, datafields)

        # # If validation file exists, load it. Otherwise get validation data from training data
        # if val_file:
        #     val_df = self.get_pandas_df(val_file)
        #     # val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        #     val_examples = []
        #     for i in val_df.values.tolist():
        #         label = i[1]
        #         text = i[0]
        #         text = text.split(' ')
        #         text = ['<cls>'] + text[0:-5] + ['<sep>'] + text[-5::]
        #         text = (' ').join(text)
        #         example = data.Example.fromlist([text, label],datafields)
        #         val_examples.append(example)
        #     val_data = data.Dataset(val_examples, datafields)
        # else:
        #     train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_test_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator = data.BucketIterator(
            (val_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        # self.val_iterator, self.test_iterator = data.BucketIterator.splits(
        #     (val_data, test_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False,
        #     shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} validation examples".format(len(val_data)))


class Dataset_adaptive_category(object):
    def __init__(self, config, num_category):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        self.num_category = num_category

    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text": data_text, "label": data_label})
        return full_df

    def load_data(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)

        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))

    def load_data_cls(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        # train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_examples = []
        for i in train_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            train_examples.append(example)

        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        # test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_examples = []
        for i in test_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            test_examples.append(example)

        test_data = data.Dataset(test_examples, datafields)

        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            # val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_examples = []
            for i in val_df.values.tolist():
                label = i[1]
                text = i[0]
                text = text.split(' ')
                category = text[-5::]
                text_str = text[0:-5]
                text_str = (' ').join(text_str)
                text_str = clean_str(text_str)
                text_str = text_str.split(' ')
                text = ['<cls>'] + text_str + ['<sep>'] + category
                text = (' ').join(text)
                example = data.Example.fromlist([text, label], datafields)
                val_examples.append(example)
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        # self.train_iterator = data.BucketIterator(
        #     (train_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False,
        #     shuffle=True)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))
        print("Loaded {} validation examples".format(len(val_data)))

    def load_data_cls_wo_valid_category(self, train_file, test_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        # train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_test_examples = []
        train_examples = []
        for i in train_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            if self.num_category != 0:
                category = text[-5::]
            if self.num_category == 0:
                category = []
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            train_examples.append(example)
            train_test_examples.append(example)

        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        # test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_examples = []
        for i in test_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            if self.num_category != 0:
                category = text[-5::]
            if self.num_category == 0:
                category = []
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            test_examples.append(example)
            train_test_examples.append(example)

        val_data = data.Dataset(test_examples, datafields)

        train_test_data = data.Dataset(train_test_examples, datafields)

        # # If validation file exists, load it. Otherwise get validation data from training data
        # if val_file:
        #     val_df = self.get_pandas_df(val_file)
        #     # val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        #     val_examples = []
        #     for i in val_df.values.tolist():
        #         label = i[1]
        #         text = i[0]
        #         text = text.split(' ')
        #         text = ['<cls>'] + text[0:-5] + ['<sep>'] + text[-5::]
        #         text = (' ').join(text)
        #         example = data.Example.fromlist([text, label],datafields)
        #         val_examples.append(example)
        #     val_data = data.Dataset(val_examples, datafields)
        # else:
        #     train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_test_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator = data.BucketIterator(
            (val_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        # self.val_iterator, self.test_iterator = data.BucketIterator.splits(
        #     (val_data, test_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False,
        #     shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} validation examples".format(len(val_data)))

    def load_data_cls_wo_valid(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data

        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        # TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)

        # train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_test_examples = []
        train_examples = []
        for i in train_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            train_examples.append(example)
            train_test_examples.append(example)

        train_data = data.Dataset(train_examples, datafields)

        test_df = self.get_pandas_df(test_file)
        # test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_examples = []
        for i in test_df.values.tolist():
            label = i[1]
            text = i[0]
            text = text.split(' ')
            category = text[-5::]
            text_str = text[0:-5]
            text_str = (' ').join(text_str)
            text_str = clean_str(text_str)
            text_str = text_str.split(' ')
            text = ['<cls>'] + text_str + ['<sep>'] + category
            text = (' ').join(text)
            example = data.Example.fromlist([text, label], datafields)
            test_examples.append(example)
            train_test_examples.append(example)

        val_data = data.Dataset(test_examples, datafields)

        train_test_data = data.Dataset(train_test_examples, datafields)

        # # If validation file exists, load it. Otherwise get validation data from training data
        # if val_file:
        #     val_df = self.get_pandas_df(val_file)
        #     # val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        #     val_examples = []
        #     for i in val_df.values.tolist():
        #         label = i[1]
        #         text = i[0]
        #         text = text.split(' ')
        #         text = ['<cls>'] + text[0:-5] + ['<sep>'] + text[-5::]
        #         text = (' ').join(text)
        #         example = data.Example.fromlist([text, label],datafields)
        #         val_examples.append(example)
        #     val_data = data.Dataset(val_examples, datafields)
        # else:
        #     train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_test_data)
        self.vocab = TEXT.vocab
        # print('toprecreationclimbing' in TEXT.vocab.itos)

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator = data.BucketIterator(
            (val_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)

        # self.val_iterator, self.test_iterator = data.BucketIterator.splits(
        #     (val_data, test_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False,
        #     shuffle=False)

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

def evaluate_model_seg(model, iterator, seg_lookup):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
            t = torch.tensor(np.take(seg_lookup, batch.text.numpy())).cuda()
        else:
            x = batch.text
        y_pred = model(x,t)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score



def evaluate_model_seg_mask(model, iterator, seg_lookup, pad_lookup):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
            t = torch.tensor(np.take(seg_lookup, batch.text.numpy())).cuda()
            pad = torch.tensor(np.take(pad_lookup, batch.text.numpy())).cuda()
        else:
            x = batch.text
        y_pred = model(x,t,pad)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score


