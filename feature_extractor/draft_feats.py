import numpy as np
import pandas as pd
import spacy
import re
import pickle
from transformers import pipeline



class article():
    def __init__(self, raw_text, dict_positive, dict_negative, keywords, dict_arousal, toxicity_pipeline):
        self.raw_text = raw_text
        self.toxicity_pipeline = toxicity_pipeline
        self.dict_positive = dict_positive
        self.dict_negative = dict_negative
        self.dict_arousal = dict_arousal
        self.keywords = keywords
        self.text = self.preproces()
        self.doc = self.to_spacy()
        self.num_chars = self.get_word_len()
        self.num_words = self.get_sent_len()
        #self.num_clauses = None
        self.positive = self.get_positive_words()
        self.negative = self.get_negative_words()
        self.nouns = self.get_no_nouns()
        self.adjectives = self.get_no_adj()
        self.arousal = self.get_arousive_words()
        self.count_keywords = self.get_count_keywords()
        self.numbers = self.get_no_num()
        self.ner = self.get_count_ner()
        self.oov = self.get_no_oov()
        self.question = self.get_questions()
        self.exclamation = self.get_exclamation()
        self.personal = self.get_personal()
        self.hate = self.get_hate_score()
        self.negations = None   #ToDo this is important, bc fake used more negations than news -> BERT maybe?


    def to_spacy(self):
        nlp = spacy.load('de_core_news_sm')
        doc = nlp(self.text)
        return doc


    def preproces(self):
        text = re.sub(r'http\S+|{link}|\[video\]|@\S*|#\S*', '',
                           self.raw_text.replace('\t', '').replace("\xad", ''))
        return text

    def get_positive_words(self):
        """
        :return: cumulated score of words which are in positive dict (SentiWs)
        """
        positive = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                positive+=self.dict_positive.get(t.text.lower(), 0)
        return positive

    def get_negative_words(self):
        """
        :return: cumulated score of words which are in negative dict (SentiWs)
        """
        negative = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                negative+=self.dict_negative.get(t.text.lower(), 0)
        return negative

    def get_arousive_words(self):
        """
        :return: cumulated score of words which are in arousal dict (BAWL)
        """
        arousal = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                arousal+=self.dict_arousal.get(t.lemma_.lower(), 0)
        return arousal

    def get_count_keywords(self):
        """
        :return: count words which are in keywords list (words which are likely to be used in populism,
        conspiracy theories and fake news)
        """
        k_count = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                if (t.text.lower() in self.keywords):k_count+=1
        return k_count

    def get_word_len(self):
        """
        :return: returns average word length
        """
        tokens = [str(t) for t in self.doc]
        tl = []
        for t in tokens:
            tl.append(len(list(t)))
        return np.average(tl)

    def get_sent_len(self):
        """
        :return: returns average sentence length
        """
        sentences_len = [len(s) for s in self.doc.sents]
        return np.average(sentences_len)

    def get_no_nouns(self):
        """
        :return: returns number of nouns
        """
        nouns = [t for t in self.doc if t.pos_ == 'NOUN']
        return len(nouns)

    def get_no_adj(self):
        """
        :return: returns number of adjectives
        """
        adj = [t for t in self.doc if t.pos_ == 'ADJ']
        return len(adj)

    def get_no_num(self):
        """
        :return: returns number of numbers
        """
        num = [t for t in self.doc if t.pos_ == 'NUM']
        return len(num)

    def get_questions(self):
        """
        :return: returns number of question marks
        """
        questions = [t for t in self.doc if t.text == '?']
        return len(questions)

    def get_exclamation(self):
        """
        :return: returns number of exclamation marks
        """
        exclamation_marks = [t for t in self.doc if t.text == '!']
        return len(exclamation_marks)

    def get_no_oov(self):
        """
        :return: returns number of out-of-vocabulary words
        """
        num = [t for t in self.doc if t.is_oov is True]
        return len(num)

    def get_count_ner(self):
        """
        :return: returns number of NER
        """
        return len(list(self.doc.ents))

    def get_personal(self):
        """
        :return: returns number of times personal pronouns were used
        """
        personal = [t for t in self.doc if t.tag_ in ['PPOSAT', 'PPER'] and
               t.lemma_.lower() in ['mein', 'dein', 'deine', 'unser', 'unseren', 'ich', 'du', 'wir', 'ihr', 'mir',
                                    'dir', 'euch', 'uns', 'dich', 'mich', 'euch', 'uns']]
        print(len(personal))

    def get_hate_score(self):
        """
        :return: returns score of hate/toxicity
        """
        hate_score = 0
        result = toxicity_pipeline(self.text)[0]
        if result['label'] == 'toxic':
            hate_score = result['score']
        return hate_score


    def return_results(self):
        results = [self.num_chars, self.num_words, self.positive, self.negative, self.nouns,
                   self.adjectives, self.arousal, self.count_keywords, self.numbers, self.ner,
                   self.oov,self.personal,self.hate, self.question, self.exclamation]
        return [0 if i is None else i for i in results]

class headline():
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.text = self.preproces()
        self.doc = self.to_spacy()
        self.num_words = self.get_word_len()
        self.question = self.get_questions()
        self.exclamation = self.get_exclamation()
        self.count_cap = self.get_no_capital()

    def to_spacy(self):
        nlp = spacy.load('de_core_news_sm')
        doc = nlp(self.text)
        return doc

    def preproces(self):
        text = re.sub(r'http\S+|{link}|\[video\]|@\S*|#\S*', '',
                           self.raw_text.replace('\t', '').replace("\xad", ''))
        return text

    def return_results(self):
        results = [self.num_chars, self.num_words,  self.question, self.exclamation]
        return [0 if i is None else i for i in results]

    def get_word_len(self):
        """
        :return: returns sum of words
        """
        tokens = [str(t) for t in self.doc]
        return len(tokens)

    def get_questions(self):
        """
        :return: returns number of question marks
        """
        questions = [t for t in self.doc if t.text == '?']
        return len(questions)

    def get_exclamation(self):
        """
        :return: returns number of exclamation marks
        """
        exclamation_marks = [t for t in self.doc if t.text == '!']
        return len(exclamation_marks)

    def get_no_capital(self):
        """
        :return: returns number of capital letters in the header
        """
        count_cap=0
        for token in self.doc:
            count_cap += sum(1 for c in token.text if c.isupper())
        return count_cap

    def return_results(self):
        results = [self.num_words,  self.question, self.exclamation, self.count_cap]
        return [0 if i is None else i for i in results]


if __name__ == "__main__":

    # get dictionaries for positive, negative and arousal words
    # load keyword list
    dict_positive = pickle.load(open("dictPositive.p", "rb"))
    dict_negative = pickle.load(open("dictNegative.p", "rb"))
    keywords = pd.read_csv('Keywords_fake.txt')
    keywords = keywords.keyword.str.lower().tolist()
    pd_arousal = pd.read_csv(filepath_or_buffer='list_arousal.csv', sep=';')
    dict_arousal = pd_arousal.set_index('WORD_LOWER').to_dict()['AROUSAL_MEAN']

    # for hatespeech detection
    model_name = 'ml6team/distilbert-base-german-cased-toxic-comments'
    toxicity_pipeline = pipeline('text-classification', model=model_name, tokenizer=model_name)

    feats_article = article(raw_text='Hier ist ein schön TExt. Und hier noch ein Satz. Das ist wütender Junge.',
                dict_positive=dict_positive, dict_negative=dict_negative, keywords=keywords,
                dict_arousal=dict_arousal, toxicity_pipeline=toxicity_pipeline
                )
    print(feats_article.return_results())

    feats_headline = headline('hier ist eine Überschrift?!!!')
    print(feats_headline.return_results())

    feats_all = feats_article.return_results()+feats_headline.return_results()
    print(feats_all)