import pandas as pd
import numpy as np
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc,
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pymorphy2 import MorphAnalyzer
lemmatizer = WordNetLemmatizer()

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
morph = MorphAnalyzer()
forb_el = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def extract_adjective_noun(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    adjectives_nouns = []
    for i in range(len(doc.tokens)-1):
        if doc.tokens[i].pos == 'ADJ' and doc.tokens[i+1].pos == 'NOUN':
            f_flag = True
            for el in doc.tokens[i].text:
                if el in forb_el:

                    f_flag = False
                    break
            if f_flag:
                if [doc.tokens[i].text.lower(), doc.tokens[i+1].text.lower()] not in adjectives_nouns:
                    adjectives_nouns.append([doc.tokens[i].text.lower(), doc.tokens[i+1].text.lower()])
    return adjectives_nouns


def create_phrase_lists(text):
    doc = Doc(text)
    doc.segment(segmenter)
    sentences = []
    for s in doc.sents:
        sentences.append(s.text)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    tokens = []
    for t in doc.tokens:
        tokens.append(t.lemma)

    unique_stops = set(stopwords.words('russian'))
    no_stops = []
    for token in tokens:
        if token not in unique_stops and token.isalpha():
            no_stops.append(token)

    tfIdfVectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
    tfIdf = tfIdfVectorizer.fit_transform(tokens)
    mean_weights = np.asarray(tfIdf.mean(axis = 0)).ravel().tolist()
    mean_weights_df = pd.DataFrame({'mean_weights': mean_weights},
                                   index=tfIdfVectorizer.get_feature_names_out())

    adj_nouns = extract_adjective_noun(text1)

    a_and_n_phrases = []
    ap_and_n_phrases = []
    a_and_np_phrases = []
    for p in adj_nouns:
        if morph.normal_forms(p[0])[0] in mean_weights_df.index and morph.normal_forms(p[1])[0] in mean_weights_df.index:
            t_i1 = mean_weights_df.loc[morph.normal_forms(p[0])[0]]['mean_weights']
            t_i2 = mean_weights_df.loc[morph.normal_forms(p[1])[0]]['mean_weights']
            if t_i2 >= 0.01:
                a_and_n_phrases.append(p)
            elif t_i2 < 0.01 and t_i1 >= 0.003:
                ap_and_n_phrases.append(p)
            else:
                a_and_np_phrases.append(p)

    res1 = [a_and_n_phrases, ap_and_n_phrases, a_and_np_phrases]
    print('Существительное в прямом значении + прилагательное:')
    print(res1[0])
    print('Существительное + прилагательное в переносном значении:')
    print(res1[1])
    print('Существительное в переносном значении + прилагательное:')
    print(res1[2])
    print('-----------------------------------------------------------')
    word = input('Введите слово для поиска: ')

    print('Существительное в прямом значении + прилагательное:')

    for p in res1[0]:
        if morph.normal_forms(p[1])[0] == word:
            print(p)
    print('Существительное + прилагательное в переносном значении:')

    for p in res1[1]:
        if morph.normal_forms(p[1])[0] == word:
            print(p)
    print('Существительное в переносном значении + прилагательное:')

    for p in res1[2]:
        if morph.normal_forms(p[1])[0] == word:
            print(p)


filename = input('Введите название файла, с которым будет работать модуль: ')
data1 = open(filename, 'r', encoding='utf-8')
text1 = ''
for t in data1:
    text1 += t
create_phrase_lists(text1)
