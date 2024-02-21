# ta.py
#    
# Extractive text summarization of a prescribed range of Scripture
#
# Copyright (c) 2022-2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Updated: 21 February 2024
#
# URL: https://github.com/drjohnnycheng/cwordtm.git
# For license information, see LICENSE.TXT

import numpy as np
from importlib_resources import files
from nltk.tokenize import sent_tokenize
from . import util

def get_sentences(df, lang):
    """Returns the list of sentences tokenized from the collection of documents (df).

    :param df: The input DataFrame storing the Scripture, default to None
    :type df: pandas.DataFrame
    :param lang: If the value is 'chi' , the processed language is assumed to be Chinese
        otherwise, it is English, default to None
    :type lang: str
    :return: The list of sentences tokenized from the collection of document
    :rtype: pandas.DataFrame
    """

    text = util.get_text(df)
    if lang == 'chi':
        sentences = text.split('。')
        if sentences[-1] == '':
            sentences = np.delete(sentences, -1)
    else:
        sentences = sent_tokenize(text)

    return sentences


# Score a sentence by its words
def get_sent_scores(sentences, diction, sent_len) -> dict:   
    """Returns the dictionary of a list of sentences with their scores 
    computed by their words

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
        # sent_wordcount = (len(util.get_sent_terms(sentence)))
        sent_wordcount_net = 0
        for word_weight in diction:
            if word_weight in sentence.lower():
                sent_wordcount_net += 1
                if sentence[:sent_len] in sent_weight:
                    sent_weight[sentence[:sent_len]] += diction[word_weight]
                else:
                    sent_weight[sentence[:sent_len]] = diction[word_weight]

        if sent_weight != dict() and sent_weight.get(sentence[:sent_len], '') != '':
            sent_weight[sentence[:sent_len]] = sent_weight[sentence[:sent_len]] / \
                                                sent_wordcount_net

    return sent_weight


def get_summary(sentences, sent_weight, threshold, sent_len):
    """Returns the summary of the collection of sentences

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
    :return: The summary of the collection of sentences
    :rtype: str
    """

    sent_counter = 0
    summary = ''
    sep ='~'

    for sentence in sentences:
        if sentence[:sent_len] in sent_weight and \
           sent_weight[sentence[:sent_len]] >= (threshold):
            summary += sentence + sep
            sent_counter += 1

    return summary


def summary(df, lang='en', weight=1.5, sent_len=8):
    """Returns the summary of the collection of sentences stored in a DataFrame (df)

    :param df: The collection of target sentences for summarization,
        default to None
    :type df: pandas.DataFrame
    :param lang: The language, either English ('en') or Chinese ('chi') 
        of the target text to be processed, default to 'en'
    :type lang: str, optional
    :param weight: The factor to be multiplied to the threshold, which 
        determines the sentences as the summary, default to 1.5
    :type weight: float, optional
    :param sent_len: The upper bound of the number of sentences to be processed,
        default to 8
    :type sent_len: int, optional
    :return: The summary of the collection of sentences
    :rtype: str
    """

    if type(df) == str: return

    util.set_lang(lang)
    diction = util.get_diction(df)
    sentences = get_sentences(df, lang)

    sent_scores = get_sent_scores(sentences, diction, sent_len)
    threshold = np.mean(list(sent_scores.values()))
    return get_summary(sentences, sent_scores, weight * threshold, sent_len)
