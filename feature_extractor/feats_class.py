import pickle
import re

import numpy as np
import pandas as pd
import spacy
from transformers import pipeline


class Article():
    def __init__(self, raw_text, dict_positive, dict_negative, lex_style, lex_anto, lex_topo,
                 dict_arousal,
                 toxicity_pipeline, nlp):
        self.raw_text = raw_text
        self.nlp = nlp
        self.toxicity_pipeline = toxicity_pipeline
        self.dict_positive = dict_positive
        self.dict_negative = dict_negative
        self.dict_arousal = dict_arousal
        self.lex_style = lex_style
        self.lex_anto = lex_anto
        self.lex_topo = lex_topo
        self.text = self.preproces()
        self.doc = self.to_spacy()
        self.num_chars = self.get_word_len()
        self.num_words = self.get_sent_len()
        self.positive = self.get_positive_words() / len(self.doc)
        self.negative = self.get_negative_words() / len(self.doc)
        self.nouns = self.get_no_nouns() / len(self.doc)
        self.modal = self.get_no_noun_modal() / len(self.doc)
        self.adjectives = self.get_no_adj() / len(self.doc)
        self.arousal = self.get_arousive_words() / len(self.doc)
        self.count_style, self.count_anto, self.count_topoi = self.get_scores()
        self.numbers = self.get_no_num() / len(self.doc)
        self.ner = self.get_count_ner() / len(self.doc)
        self.oov = self.get_no_oov() / len(self.doc)
        self.question = self.get_questions() / len(self.doc)
        self.exag = self.get_exag() / len(self.doc)
        self.quatation = self.get_quatation() / len(self.doc)
        self.exclamation = self.get_exclamation() / len(self.doc)
        self.personal = self.get_personal() / len(self.doc)
        self.hate = self.get_hate_score()
        self.negations = self.count_negations() / len(self.doc)

    def to_spacy(self):
        doc = self.nlp(self.text)
        return doc

    def preproces(self):
        text = re.sub(r'http\S+|{link}|\[video\]|@\S*|#\S*', '',
                      self.raw_text.replace('\t', '').replace("\xad", '').replace("\n", ''))
        return text

    def get_positive_words(self):
        """
        :return: cumulated score of words which are in positive dict (SentiWs)
        """
        positive = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                positive += self.dict_positive.get(t.lemma_.lower(), 0)
        return positive

    def get_negative_words(self):
        """
        :return: cumulated score of words which are in negative dict (SentiWs)
        """
        negative = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                negative += self.dict_negative.get(t.lemma_.lower(), 0)
        return negative

    def get_arousive_words(self):
        """
        :return: cumulated score of words which are in arousal dict (BAWL)
        """
        arousal = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                arousal += self.dict_arousal.get(t.lemma_.lower(), 0)
        return arousal

    def get_count_keywords(self):
        """
        :return: count words which are in keywords list (words which are likely to be used in populism,
        conspiracy theories and fake news)
        """
        k_count = 0
        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                if (t.text.lower() in self.keywords): k_count += 1
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

    def get_no_noun_modal(self):
        """
        :return: returns number of nouns
        """
        nouns = [t for t in self.doc if
                 t.lemma_ in ['dürfen', "können", 'wollen', 'mögen', 'müssen', 'sollen']]
        # print(len(nouns))
        return len(nouns)

    def get_no_nouns(self):
        """
        :return: returns number of numbers
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

    def get_quatation(self):
        """
        :return: returns number of quatation marks  -> sogenannte 'Demokratie'
        """
        quotation_marks = [t for t in self.doc if t.text in ['"', "'", '„', '‚', '“', '‘', '«', '`']]
        return len(quotation_marks)

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

        personal = [t for t in self.doc if
                    t.lemma_.lower() in ['mein', 'unser', 'unsere', 'unseren', 'ich', 'wir', 'mir',
                                         'dir', 'euch', 'ihr', 'eure', 'eurer', 'uns', 'dich', 'mich']]
        return len(personal)

    def get_hate_score(self):
        """
        :return: returns score of hate/toxicity
        """
        hate_score = 0
        result = self.toxicity_pipeline(self.text, truncation=True)[0]
        if result['label'] == 'toxic':
            hate_score = result['score']
        return hate_score

    def count_negations(self):
        neg_pattern = r"(kein|nie|weder|ohne|nicht|nein)."
        return len(re.findall(neg_pattern, self.text))

    # wird nicht gebraucht, das Lexicon Scandal enthält diese Wörter schon
    def get_exag(self):
        # dramatisch, steigend, massenhaft  # Schaefer S.177 , schwerwiegend, äußerst, massiv, enorm, extrem, hochgradig Breil et. al S.250 , absolut, total, völlig Scharloth s.7
        exag = [t for t in self.doc if
                t.lemma_.lower() in ['dramatisch', 'steigend', 'massenhaft', 'schwerwiegend', 'äußerst', 'massiv',
                                     'enorm', 'hochgradig', 'extrem', 'absolut', 'total', 'völlig']]
        return len(exag)

    def get_scores(self):
        self.count_style = 0
        self.count_anto = 0
        self.count_topoi = 0

        for t in self.doc:
            if t.is_stop is False and t.is_space is False:
                if (t.text.lower() in self.lex_style): self.count_style += 1
                if (t.text.lower() in self.lex_anto): self.count_anto += 1
                if (t.text.lower() in self.lex_topo): self.count_topoi += 1
        return self.count_style / len(self.doc), self.count_anto / len(self.doc), self.count_topoi / len(self.doc)

    def return_results(self):
        results = [self.num_chars,
                   self.num_words,
                   self.positive,
                   self.negative,
                   self.nouns,
                   self.modal,
                   self.adjectives,
                   self.arousal,
                   # self.fear,
                   # self.pop,
                   # self.manip,
                   # self.scandal,
                   self.count_style,
                   self.count_anto,
                   self.count_topoi,
                   self.numbers,
                   self.ner,
                   self.oov,
                   self.question,
                   # self.exag,
                   self.quatation,
                   self.exclamation,
                   self.personal,
                   self.hate,
                   self.negations]
        return [0 if i is None else i for i in results]


class Headline():
    def __init__(self, raw_text, nlp):
        self.raw_text = raw_text
        self.nlp = nlp
        self.text = self.preproces()
        self.doc = self.to_spacy()
        self.num_words = self.get_word_len()
        self.question = self.get_questions()
        self.exclamation = self.get_exclamation()
        self.count_cap = self.get_no_capital()

    def to_spacy(self):
        # nlp = spacy.load('de_core_news_sm')
        doc = self.nlp(self.text)
        return doc

    def preproces(self):
        text = re.sub(r'http\S+|{link}|\[video\]|@\S*|#\S*', '',
                      self.raw_text.replace('\t', '').replace("\xad", ''))
        return text

    def return_results(self):
        results = [self.num_chars, self.num_words, self.question, self.exclamation]
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
        count_cap = 0
        for token in self.doc:
            count_cap += sum(1 for c in token.text if c.isupper())
        return count_cap

    def return_results(self):
        results = [self.num_words, self.question, self.exclamation, self.count_cap]
        return [0 if i is None else i for i in results]


if __name__ == "__main__":
    nlp = spacy.load('de_core_news_lg')
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

    feats_article = Article(raw_text='Hier ist ein schön TExt. Und hier noch ein Satz. Das ist wütender Junge.',
                            dict_positive=dict_positive, dict_negative=dict_negative, keywords=keywords,
                            dict_arousal=dict_arousal, toxicity_pipeline=toxicity_pipeline, nlp=nlp
                            )
    print(feats_article.return_results())

    feats_headline = Headline('hier ist eine Überschrift?!!!', nlp)
    print(feats_headline.return_results())

    feats_all = feats_article.return_results() + feats_headline.return_results()

    print(feats_all)
