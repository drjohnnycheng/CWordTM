# ta.py
#    
# Extractive text summarization of a prescribed range of Scripture
#
# Copyright (c) 2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Updated: 5 June 2024 (0.6.4), 31 Oct 2024
#
# URL: https://github.com/drjohnnycheng/cwordtm.git
# For license information, see LICENSE.TXT

import re
import string
import numpy as np
import pandas as pd
from importlib_resources import files
from collections import OrderedDict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from . import util


def split_chi_sentences(text):
    # Define sentence-ending punctuation, including those followed by quotations
    # sentence_endings = r'[。！？](『.*?』)?'
    # sentence_endings = r'(?<=[。！？])(?=([^』]*$))|(?<=』)'
    sentence_endings = r'[。！？]|(?<=』)'
    
    # Use regex to split by sentence endings
    sentences = re.split(sentence_endings, text)

    # Remove any empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s is not None]
    sentences = list(filter(None, sentences))  # Remove empty strings
    sentences = list(OrderedDict.fromkeys(sentences))  # Remove duplicates
    
    return sentences


def get_sentences(docs, lang='en'):
    """Returns the list of sentences tokenized from the collection of documents (df).

    :param docs: The input documents storing the Scripture, default to None
    :type docs: pandas.DataFrame
    :param lang: If the value is 'chi' , the processed language is assumed to be Chinese
        otherwise, it is English, default to 'en'
    :type lang: str, optional
    :return: The list of sentences tokenized from the collection of document
    :rtype: list
    """

    join_str = '' if lang == 'chi' else ' '
    if isinstance(docs, pd.DataFrame):
        text = join_str.join(list(docs.text.astype(str)))
    elif isinstance(docs, pd.Series):
        text = join_str.join(list(docs.astype(str)))
    elif isinstance(docs, list) or isinstance(docs, np.ndarray):
        text = join_str.join(str(doc) for doc in docs)
    else:
        text = docs

    if lang == 'chi':
        text = text.replace('\u3000', '')
        # sentences = text.split('。')
        sentences = split_chi_sentences(text)
    else:
        text = text.replace('\n', '')
        text = re.sub(r'[0-9]', '', text)
        sentences = sent_tokenize(text)
        sentences = [re.sub(r'[.:•]', '', sent) for sent in sentences]
        sentences = [sent for sent in sentences if len(sent) > 10]

    return sentences


def get_sent_scores(sentences, diction, sent_len) -> dict:   
    """Returns the dictionary of a list of sentences with their scores 
    computed by their words.

    :param sentences: The list of sentences for computing their scores,
        default to None
    :type sentences: list
    :param diction: The dictionary storing the collection of tokenized words
        with their frequencies
    :type diction: collections.Counter object
    :param sent_len: The upper bound of the number of sentences to be processed,
        default to None
    :type sent_len: int
    :return: The list of sentences tokenized from the collection of document
    :rtype: pandas.DataFrame
    """

    sent_weight = dict()

    for sentence in sentences:
        sent_wordcount_net = 0
        for word_weight in diction:
            if word_weight in sentence.lower():
                sent_wordcount_net += 1
                if sentence[:sent_len] in sent_weight:
                    sent_weight[sentence[:sent_len]] += diction[word_weight]
                else:
                    sent_weight[sentence[:sent_len]] = diction[word_weight]

        if sent_weight != dict() and sent_weight.get(sentence[:sent_len], '') != '' \
            and sent_wordcount_net > 0:
            sent_weight[sentence[:sent_len]] = sent_weight[sentence[:sent_len]] / \
                                                sent_wordcount_net

    return sent_weight


def get_summary(sentences, sent_weight, threshold, sent_len):
    """Returns the summary of the collection of sentences.

    :param sentences: The list of target sentences for summarization, default to None
    :type sentences: list
    :param sent_weight: The dictionary of a list of sentences with their scores 
        computed by their words
    :type sent_weight: collections.Counter object
    :param threshold: The minimum value of sentence weight for extracting that sentence
        as part of the final summary, default to None
    :type threshold: float
    :param sent_len: The upper bound of the number of sentences to be processed,
        default to None
    :type sent_len: int
    :return: The list of sentences of the extractive summary
    :rtype: list
    """

    sent_counter = 0
    summary = []

    for sentence in sentences:
        if sentence[:sent_len] in sent_weight and \
            sent_weight[sentence[:sent_len]] >= (threshold):
            summary.append(sentence)

    return summary


def summary_chi(docs, weight=1.5, sent_len=8):
    """Returns an extractive summary of a collection of Chinese sentences.

    :param docs: The collection of target documents for summarization,
        default to None
    :type docs: pandas.DataFrame or pandas.Series or numpy.ndarray or list
    :param weight: The factor to be multiplied to the threshold, which 
        determines the sentences as the summary, default to 1.5
    :type weight: float, optional
    :param sent_len: The upper bound of the number of sentences to be processed,
        default to 8
    :type sent_len: int, optional
    :return: The list of sentences of the extractive summary
    :rtype: list
    """

    lang = 'chi'
    util.set_lang(lang)
    diction = util.get_diction(docs)
    sentences = get_sentences(docs, lang)

    sent_scores = get_sent_scores(sentences, diction, sent_len)
    threshold = np.mean(list(sent_scores.values()))
    return get_summary(sentences, sent_scores, weight * threshold, sent_len)


def preprocess_sent(text):
    """Preprocesses English text by tokenizing text into sentences of words,
    converting text to lower case, removing stopwords, lemmatize text, and
    tagging text with Part-of-Speech (POS).

    :param text: The text to be preprocessed, default to None
    :type text: str
    :return: The list of preprocessed and tagged sentences (word, pos)
    :rtype: list of tuples (str, str)
    """

    if isinstance(text, list) or isinstance(text, np.ndarray):
        text = ' '.join(text)

    # print("Preprocessing text ...")

    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Convert text to lowercase and tokenize text into words
    sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    sentences = [[word for word in sentence if word not in stop_words] for sentence in sentences]
    
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    sentences = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in sentences]
    
    # Tag text with POS
    sentences = [pos_tag(sentence) for sentence in sentences]
    
    return sentences


def summary_en(docs, sent_len=8):
    """Returns an extractive summary of a collection of English sentences.

    :param docs: The collection of target documents for summarization,
        default to None
    :type docs: pandas.DataFrame or pandas.Series or numpy.ndarray or list or text
    :param sent_len: The upper bound of the number of sentences to be processed,
        default to 8
    :type sent_len: int, optional
    :return: The list of sentences of the extractive summary
    :rtype: list
    """

    join_str = ' '
    if isinstance(docs, pd.DataFrame):
        text = join_str.join(list(docs.text.astype(str)))
    elif isinstance(docs, pd.Series):
        text = join_str.join(list(docs.astype(str)))
    elif isinstance(docs, list) or isinstance(docs, np.ndarray):
        text = join_str.join(str(doc) for doc in docs)
    else:
        text = docs

    tagged_sentences = preprocess_sent(text)

    # Compute sentence scores
    sentence_scores = []
    for sentence in tagged_sentences:
        score = 0
        for word, pos in sentence[:sent_len]:
            # Filter with nouns and verbs
            if pos.startswith('NN') or pos.startswith('VB'):
                score += 1
        sentence_scores.append(score)

    # Extract top scoring sentences
    top_sentences = sorted(range(len(sentence_scores)), \
                           key=lambda i: sentence_scores[i], \
                           reverse=True)

    # Build a summary
    sentences = sent_tokenize(text)
    # summary = ' '.join([sentences[i] for i in top_sentences])
    summary = [sentences[i] for i in top_sentences]

    return summary
