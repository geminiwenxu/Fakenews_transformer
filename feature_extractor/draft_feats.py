import numpy as np
import pandas as pd
import spacy
import torch
import re
import pickle

class article():
    def __init__(self, raw_text, dict_positive, dict_negative, keywords):
        self.raw_text = raw_text
        self.dict_positive = dict_positive
        self.dict_negative = dict_negative
        self.keywords = keywords
        #self.dict_arousal = dict_arousal
        self.text = self.preproces()
        self.doc = self.to_spacy()
        self.num_chars = self.get_word_len()
        self.num_words = self.get_sent_len()
        self.num_clauses = None
        self.positive = self.get_positive_words()
        self.negative = self.get_negative_words()
        self.nouns = self.get_no_nouns()
        self.adjectives = self.get_no_adj()
        #self.adverbs = None
        self.arousal = None
        self.count_keywords = self.get_count_keywords()
        self.numbers = None
        #self.certainity = None
        #self.general = None
        self.ner = self.get_count_ner()
        self.hate = None
        self.negations = None


    def to_spacy(self):
        nlp = spacy.load('de_core_news_sm')
        doc = nlp(self.text)
        return doc


    def preproces(self):
        text = re.sub(r'http\S+|{link}|\[video\]|@\S*|#\S*', '',
                           self.raw_text.replace('\t', '').replace("\xad", ''))
        return text

    def get_positive_words(self):
        positive = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                positive+=self.dict_positive.get(t.text.lower(), 0)
        return positive

    def get_negative_words(self):
        negative = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                negative+=self.dict_negative.get(t.text.lower(), 0)
        return negative

    def get_count_keywords(self):
        k_count = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                if (t.text.lower() in self.keywords):k_count+=1
        return k_count

    def get_word_len(self):
        tokens = [str(t) for t in self.doc]
        tl = []
        for t in tokens:
            tl.append(len(list(t)))
        return np.average(tl)

    def get_sent_len(self):
        sentences_len = [len(s) for s in self.doc.sents]
        return np.average(sentences_len)

    def get_no_nouns(self):
        nouns = [t for t in self.doc if t.pos_ == 'NOUN']
        return len(nouns)

    def get_no_adj(self):
        adj = [t for t in self.doc if t.pos_ == 'ADJ']
        return len(adj)

    def get_count_ner(self):
        return len(list(self.doc.ents))

    def return_results(self):
        return self.num_chars, \
               self.num_words, \
               self.positive,   \
               self.count_keywords


if __name__ == "__main__":
    dict_positive = pickle.load(open("dictPositive.p", "rb"))
    dict_negative = pickle.load(open("dictNegative.p", "rb"))
    keywords = pd.read_csv('Keywords_fake.txt')
    keywords = keywords.keyword.str.lower().tolist()
    a = article(raw_text='Hier ist ein schön TExt. Und hier noch ein Satz armutsmigrant',
                dict_positive=dict_positive, dict_negative=dict_negative, keywords=keywords
                )
    print(a.return_results())