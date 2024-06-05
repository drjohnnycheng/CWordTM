# tm.py
#    
# Topic modeling with LDA, NMF, and BERTopic for a prescribed range of
#   Scripture or other text
#
# Copyright (c) 2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Updated: 4 June 2024
#
# URL: https://github.com/drjohnnycheng/cwordtm.git
# For license information, see LICENSE.TXT


# Dependencies

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import string
import re
import math

import time
from pprint import pprint
from IPython.display import IFrame
from importlib_resources import files

import jieba
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import nltk

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

import torch
from bertopic import BERTopic
from transformers import BertTokenizer, BertModel

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

from . import util


def load_text(textfile, text_col='text'):
    """Loads and returns the list of documents from the prescribed file ('textfile').

    :param textfile: The prescribed text file from which the text is loaded,
        default to None
    :type textfile: str
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :return: The list of documents loaded
    :rtype: list
    """

    # docs = pd.read_csv(textfile)
    docs = util.load_text(textfile)
    return list(docs[text_col])


def load_bible(textfile, cat=0, group=True):
    """Loads and returns the Bible Scripture from the prescribed internal
    file ('textfile').

    :param textfile: The package's internal Bible text from which the text is loaded,
        either World English Bible ('web.csv') or Chinese Union Version (Traditional)
        ('cuv.csv'), default to None
    :type textfile: str
    :param cat: The category indicating a subset of the Scripture to be loaded, where
        0 stands for the whole Bible, 1 for OT, 2 for NT, or one of the ten categories
        ['tor', 'oth', 'ket', 'map', 'mip', 'gos', 'nth', 'pau', 'epi', 'apo'] (See 
        the package's internal file 'data/book_cat.csv'), default to 0
    :type cat: int or str, optional
    :param group: The flag indicating whether the loaded text is grouped by chapter,
        default to True
    :type group: bool, optional
    :return: The collection of Scripture loaded
    :rtype: pandas.DataFrame
    """

    # textfile = "web.csv"
    scfile = files('cwordtm.data').joinpath(textfile)
    print("Loading Bible '%s' ..." %scfile)
    df = pd.read_csv(scfile)

    cat_list = ['tor', 'oth', 'ket', 'map', 'mip',\
                'gos', 'nth', 'pau', 'epi', 'apo']	
    cat = str(cat)
    if cat == '1' or cat == 'ot':
        df = util.extract(df, testament=0)
    elif cat == '2' or cat == 'nt':
        df = util.extract(df, testament=1)               
    elif cat in cat_list:
        df = util.extract(df, category=cat)

    if group:
        # Group verses into chapters
        df = df.groupby(['book_no', 'chapter'])\
                        .agg({'text': lambda x: ' '.join(x)})\
                .reset_index()

    df.text = df.text.str.replace('　', '')
    return list(df.text)

   
def process_text(doc):
    """Processes the English text through tokenization, converting to lower case,
    removing all digits, stemming, and removing punctuations and stopwords.

    :param doc: The prescribed text, in form of a string, to be processed,
        default to None
    :type doc: str
    :return: The list of the processed strings
    :rtype: list
    """

    # List of punctuation
    punc = list(set(string.punctuation))

    # List of stop words
    add_stop = []
    stop_words = ENGLISH_STOP_WORDS.union(add_stop)

    doc = TweetTokenizer().tokenize(doc)
    doc = [each.lower() for each in doc]
    doc = [re.sub('[0-9]+', '', each) for each in doc]
    doc = [SnowballStemmer('english').stem(each) for each in doc]
    doc = [w for w in doc if w not in punc]
    doc = [w for w in doc if w not in stop_words]
    return doc


class LDA:
    """The LDA object for Latent Dirichlet Allocation (LDA) modeling.
    
    :cvar num_topics: The number of topics to be built from the modeling,
        default to 10.
    :vartype num_topics: int
    :ivar textfile: The filename of the text file to be processed
    :vartype textfile: str
    :ivar chi: The flag indicating whether the processed text is in Chinese or not,
        True stands for Traditional Chinese or False for English
    :vartype chi: bool
    :ivar num_topics: The number of topics set for the topic model
    :vartype num_topics: int
    :ivar docs: The collection of the original documents to be processed
    :vartype docs: pandas.DataFrame or list
    :ivar pro_docs: The collection of documents, in form of list of lists of words
        after text preprocessing
    :vartype pro_docs: list
    :ivar dictionary: The dictionary of word ids with their tokenized words
        from preprocessed documents ('pro_docs')
    :vartype dictionary: gensim.corpora.Dictionary
    :ivar corpus: The list of documents, where each document is a list of tuples
        (word id, word frequency in the particular document)
    :vartype corpus: list
    :ivar model: The LDA model object
    :vartype model: gensim.models.LdaModel
    :ivar vis_data: The LDA model's prepared data for visualization
    :vartype vis_data: pyLDAvis.PreparedData
    """

    def __init__(self, textfile, chi=False, num_topics=15):
        """Constructor method.
        """

        self.textfile = textfile
        self.chi = chi
        self.num_topics = num_topics
        self.docs = None
        self.pro_docs = None
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.vis_data = None

    
    def preprocess(self):
        """Process the original English documents (cwordtm.tm.LDA.docs)
        by invoking cwordtm.tm.process_text, and build a dictionary and
        a corpus from the preprocessed documents for the LDA model.
        """

        self.pro_docs = [process_text(doc) for doc in self.docs]

        # Create a dictionary and corpus for the LDA model
        self.dictionary = corpora.Dictionary(self.pro_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.pro_docs]

    def preprocess_chi(self):
        """Process the original Chinese documents (cwordtm.tm.LDA.docs) 
        by tokenizing text, removing stopwords, and building a dictionary
        and a corpus from the preprocessed documents for the LDA model.
        """

        # Build stop words
        stop_file = files('cwordtm.data').joinpath("tc_stopwords_2.txt")
        stopwords = [k[:-1] for k in open(stop_file, encoding='utf-8')\
                     .readlines() if k != '']

        # Tokenize"the text using Jieba
        dict_file = files('cwordtm.data').joinpath("user_dict_4.txt")
        jieba.load_userdict(str(dict_file))
        docs = [jieba.cut(doc) for doc in self.docs]

        # Replace special characters
        docs = [[word.replace('\u3000', ' ') for word in doc] \
                                     for doc in docs]

        # Remove stop words
        self.pro_docs = [' '.join([word for word in doc if word not in stopwords]) \
                                        for doc in docs]

        self.pro_docs = [doc.split() for doc in self.pro_docs]

        # Create a dictionary and corpus
        self.dictionary = corpora.Dictionary(self.pro_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.pro_docs]


    def fit(self):
        """Build the LDA model with the created corpus and dictionary.
        """

        self.model = models.LdaModel(self.corpus, 
                            num_topics=self.num_topics, 
                            id2word=self.dictionary, 
                            passes=10)
    
    def viz(self):
        """Shows the Intertopic Distance Map for the built LDA model.
        """

        self.vis_data = gensimvis.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.enable_notebook()
        pyLDAvis.display(self.vis_data)
        print("If no visualization is shown,")
        print("  you may execute the following commands to show the visualization:")
        print("    > import pyLDAvis")
        print("    > pyLDAvis.display(lda.vis_data)")

    def show_topics(self):
        """Shows the topics with their keywords from the built LDA model.
        """

        print("\nTopics from LDA Model:")
        pprint(self.model.print_topics())
    
    def evaluate(self):
        """Computes and outputs the coherence score, perplexity, topic diversity,
            and topic size distribution.
        """

        # Compute coherence score
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.pro_docs,
                                         dictionary=self.dictionary,
                                         coherence='c_v')
        print(f"  Coherence: {coherence_model.get_coherence()}")
        
        # Compute perplexity
        perplexity = self.model.log_perplexity(self.corpus)
        print(f"  Perplexity: {perplexity}")
        
        # Compute topic diversity
        topic_sizes = [len(self.model[self.corpus[i]]) for i in range(len(self.corpus))]
        total_docs = sum(topic_sizes)
        topic_diversity = sum([(size/total_docs)**2 for size in topic_sizes])
        print(f"  Topic diversity: {topic_diversity}")
        
        # Compute topic size distribution
        # topic_sizes = [len(self.model[self.corpus[i]]) for i in range(len(self.corpus))]
        topic_size_distribution = max(topic_sizes) / sum(topic_sizes)
        print(f"  Topic size distribution: {topic_size_distribution}\n")

# End of LDA Class


def lda_process(doc_file, source=0, text_col='text', cat=0, chi=False, group=True, eval=False):
    """Pipelines the LDA modeling.

    :param doc_file: The filename of the prescribed text file to be loaded,
        default to None
    :type doc_file: str
    :param source: The source of the prescribed document file ('doc_file'),
        where 0 refers to internal store of the package and 1 to external file,
        default to 0
    :type source: int, optional
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :param cat: The category indicating a subset of the Scripture to be loaded, where
        0 stands for the whole Bible, 1 for OT, 2 for NT, or one of the ten categories
        ['tor', 'oth', 'ket', 'map', 'mip', 'gos', 'nth', 'pau', 'epi', 'apo'] (See 
        the package's internal file 'data/book_cat.csv'), default to 0
    :type cat: int or str, optional
    :param chi: The flag indicating whether the text is processed as Chinese (True)
        or English (False), default to False
    :type chi: bool, optional
    :param group: The flag indicating whether the loaded text is grouped by chapter,
        default to True
    :type group: bool, optional
    :param eval: The flag indicating whether the model evaluation results will be shown,
        default to False
    :type eval: bool, optional
    :return: The pipelined LDA
    :rtype: cwordtm.tm.LDA object
    """

    lda = LDA(doc_file, chi)
    if source == 0:
        lda.docs = load_bible(lda.textfile, cat=cat, group=group)
    else:
        lda.docs = load_text(lda.textfile, text_col=text_col)

    print("Corpus loaded!")

    if chi:
        lda.preprocess_chi()
    else:
        lda.preprocess()
    print("Text preprocessed!")

    lda.fit()
    print("Text trained!")
    lda.viz()
    print("Visualization prepared!")
    lda.show_topics()

    if eval:
        print("\nModel Evaluation Scores:")
        lda.evaluate()

    return lda


class NMF:
    """The NMF object for Non-negative Matrix Factorization (NMF) modeling.

    :cvar num_topics: The number of topics to be built from the modeling,
        default to 10.
    :vartype num_topics: int
    :ivar textfile: The filename of the text file to be processed
    :vartype textfile: str
    :ivar chi: The flag indicating whether the processed text is in Chinese or not,
        True stands for Traditional Chinese or False for English
    :vartype chi: bool
    :ivar num_topics: The number of topics set for the topic model
    :vartype num_topics: int
    :ivar docs: The collection of the original documents to be processed
    :vartype docs: pandas.DataFrame or list
    :ivar pro_docs: The collection of documents, in form of list of lists of words
        after text preprocessing
    :vartype pro_docs: list
    :ivar dictionary: The dictionary of word ids with their tokenized words
        from preprocessed documents ('pro_docs')
    :vartype dictionary: gensim.corpora.Dictionary
    :ivar corpus: The list of documents, where each document is a list of tuples
        (word id, word frequency in the particular document)
    :vartype corpus: list
    :ivar model: The NMF model object
    :vartype model: gensim.models.Nmf
    """

    def __init__(self, textfile, chi=False, num_topics=15):
        """Constructor method.
        """

        self.textfile = textfile
        self.chi = chi
        self.num_topics = num_topics
        self.docs = None
        self.pro_docs = None
        self.dictionary = None
        self.corpus = None
        self.model = None

    
    def preprocess(self):
        """Process the original English documents (cwordtm.tm.NMF.docs)
        by invoking cwordtm.tm.process_text, and build a dictionary
        and a corpus from the preprocessed documents for the NMF model.
        """

        self.pro_docs = [process_text(doc) for doc in self.docs]

        # Create a dictionary and corpus for the NMF model
        self.dictionary = corpora.Dictionary(self.pro_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.pro_docs]

    def preprocess_chi(self):
        """Process the original Chinese documents (cwordtm.tm.NMF.docs) 
        by tokenizing text, removing stopwords, and building a dictionary
        and a corpus from the preprocessed documents for the NMF model.
        """

        # Build stop words
        stop_file = files('cwordtm.data').joinpath("tc_stopwords_2.txt")
        stopwords = [k[:-1] for k in open(stop_file, encoding='utf-8')\
                     .readlines() if k != '']

        # Tokenize"the text using Jieba
        dict_file = files('cwordtm.data').joinpath("user_dict_4.txt")
        jieba.load_userdict(str(dict_file))
        docs = [jieba.cut(doc) for doc in self.docs]

        # Replace special characters
        docs = [[word.replace('\u3000', ' ') for word in doc] \
                                     for doc in docs]

        # Remove stop words
        self.pro_docs = [' '.join([word for word in doc if word not in stopwords]) \
                                        for doc in docs]

        self.pro_docs = [doc.split() for doc in self.pro_docs]

        # Create a dictionary and corpus
        self.dictionary = corpora.Dictionary(self.pro_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.pro_docs]


    def fit(self):
        """Build the NMF model with the created corpus and dictionary.
        """

        self.model = models.Nmf(self.corpus,
                                num_topics=self.num_topics)

    def show_topics_words(self):
        """Shows the topics with their keywords from the built NMF model.
        """

        print("\nTopics-Words from NMF Model:")
        for topic_id in range(self.model.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=10)
            print(f"Topic {topic_id+1}:")
            for word_id, prob in topic_words:
                # word = self.dictionary.id2token[int(word_id)]
                word = self.dictionary[int(word_id)]
                print("%s (%.6f)" %(word, prob))
            print()

    def evaluate(self):
        """Computes and outputs the coherence score, topic diversity,
        and topic size distribution.
        """

        # Compute coherence score
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.pro_docs,
                                         dictionary=self.dictionary,
                                         coherence='c_v')
        print(f"  Coherence: {coherence_model.get_coherence()}")
        
        # Compute topic diversity
        topic_sizes = [len(self.model[self.corpus[i]]) for i in range(len(self.corpus))]
        total_docs = sum(topic_sizes)
        topic_diversity = sum([(size/total_docs)**2 for size in topic_sizes])
        print(f"  Topic diversity: {topic_diversity}")
        
        # Compute topic size distribution
        # topic_sizes = [len(self.model[self.corpus[i]]) for i in range(len(self.corpus))]
        topic_size_distribution = max(topic_sizes) / sum(topic_sizes)
        print(f"  Topic size distribution: {topic_size_distribution}\n")

# End of NMF Class


def nmf_process(doc_file, source=0, text_col='text', cat=0, chi=False, group=True, eval=False):
    """Pipelines the NMF modeling.

    :param doc_file: The filename of the prescribed text file to be loaded,
        default to None
    :type doc_file: str
    :param source: The source of the prescribed document file ('doc_file'),
        where 0 refers to internal store of the package and 1 to external file,
        default to 0
    :type source: int, optional
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :param cat: The category indicating a subset of the Scripture to be loaded, where
        0 stands for the whole Bible, 1 for OT, 2 for NT, or one of the ten categories
        ['tor', 'oth', 'ket', 'map', 'mip', 'gos', 'nth', 'pau', 'epi', 'apo'] (See 
        the package's internal file 'data/book_cat.csv'), default to 0
    :type cat: int or str, optional
    :param chi: The flag indicating whether the text is processed as Chinese (True)
        or English (False), default to False
    :type chi: bool, optional
    :param group: The flag indicating whether the loaded text is grouped by chapter,
        default to True
    :type group: bool, optional
    :param eval: The flag indicating whether the model evaluation results will be shown,
        default to False
    :type eval: bool, optional
    :return: The pipelined NMF
    :rtype: cwordtm.tm.NMF object
    """

    nmf = NMF(doc_file, chi)
    if source == 0:
        nmf.docs = load_bible(nmf.textfile, cat=cat, group=group)
    else:
        nmf.docs = load_text(nmf.textfile, text_col=text_col)

    print("Corpus loaded!")

    if chi:
        nmf.preprocess_chi()
    else:
        nmf.preprocess()
    print("Text preprocessed!")

    nmf.fit()
    print("Text trained!")
    nmf.show_topics_words()

    if eval:
        print("\nModel Evaluation Scores:")
        nmf.evaluate()

    return nmf


class BTM:
    """The BTM object for BERTopic modeling.

    :cvar num_topics: The number of topics to be built from the modeling,
        default to 10
    :vartype num_topics: int
    :ivar textfile: The filename of the text file to be processed
    :vartype textfile: str
    :ivar chi: The flag indicating whether the processed text is in Chinese or not,
        True stands for Traditional Chinese or False for English
    :vartype chi: bool
    :ivar num_topics: The number of topics set for the topic model
    :vartype num_topics: int
    :ivar docs: The collection of the original documents to be processed
    :vartype docs: pandas.DataFrame or list
    :ivar pro_docs: The collection of documents, in form of list of lists of words
        after text preprocessing
    :vartype pro_docs: list
    :ivar dictionary: The dictionary of word ids with their tokenized words
        from preprocessed documents ('pro_docs')
    :vartype dictionary: gensim.corpora.Dictionary
    :ivar corpus: The list of documents, where each document is a list of tuples
        (word id, word frequency in the particular document)
    :vartype corpus: list
    :ivar model: The BERTopic model object
    :vartype model: bertopic.BERTopic
    :ivar embed: The flag indicating whether the BERTopic model is trained
        with the BERT pretrained model
    :vartype embed: bool
    :ivar bmodel: The BERT pretrained model
    :vartype bmodel: transformers.BertModel
    :ivar bt_vectorizer: The vectorizer extracted from the BERTopic model
        for model evaluation
    :vartype bt_vectorizer: sklearn.feature_extraction.text.CountVectorizer
    :ivar bt_analyzer: The analyzer extracted from the BERTopic model
        for model evaluation
    :vartype bt_analyzer: functools.partial
    :ivar cleaned_docs: The list of documents (string) built by grouping
        the original documents by the topics created from the BERTopic model
    :vartype cleaned_docs: list
    """

    def __init__(self, textfile, chi=False, num_topics=15, embed=True):
        """Constructor method.
        """

        self.textfile = textfile
        self.chi = chi
        self.num_topics = num_topics
        self.docs = None
        self.pro_docs = None
        self.dictionary = None
        self.corpus = None
        self.model = None

        self.embed = embed
        self.bmodel = None
        self.bt_vectorizer = None
        self.bt_analyzer = None
        self.cleaned_docs = None

    
    def preprocess(self):
        """Process the original English documents (cwordtm.tm.BTM.docs)
        by invoking cwordtm.tm.process_text, and build a dictionary and
        a corpus from the preprocessed documents for the BERTopic model.
        """

        self.pro_docs = [process_text(doc) for doc in self.docs]

        # Create a dictionary and corpus for the BERTopic model
        self.dictionary = corpora.Dictionary(self.pro_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.pro_docs]

    def preprocess_chi(self):
        """Process the original Chinese documents (cwordtm.tm.BTM.docs) 
        by tokenizing text, removing stopwords, and building a dictionary
        and a corpus from the preprocessed documents for the BERTopic model.
        """

        # Build stop words
        stop_file = files('cwordtm.data').joinpath("tc_stopwords_2.txt")
        stopwords = [k[:-1] for k in open(stop_file, encoding='utf-8')\
                     .readlines() if k != '']

        # Tokenize"the text using Jieba
        dict_file = files('cwordtm.data').joinpath("user_dict_4.txt")
        jieba.load_userdict(str(dict_file))
        docs = [jieba.cut(doc) for doc in self.docs]

        # Replace special characters
        docs = [[word.replace('\u3000', ' ') for word in doc] \
                                     for doc in docs]

        # Remove stop words
        self.pro_docs = [' '.join([word for word in doc if word not in stopwords]) \
                                        for doc in docs]

        self.pro_docs = [doc.split() for doc in self.pro_docs]

        # Create a dictionary and corpus
        self.dictionary = corpora.Dictionary(self.pro_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.pro_docs]


    def fit(self):
        """Build the BERTopic model for English text with the created corpus
        and dictionary.
        """

        j_pro_docs = [" ".join(doc) for doc in self.pro_docs]

        if self.embed:
            self.bmodel = BertModel.from_pretrained('bert-base-uncased')
            self.model = BERTopic(language='english', 
                                  calculate_probabilities=True,
                                  embedding_model=self.bmodel,
                                  nr_topics=self.num_topics)
        else:
            self.model = BERTopic(language='english', 
                                  calculate_probabilities=True,
                                  nr_topics=self.num_topics)

        _, _ = self.model.fit_transform(j_pro_docs)


    def fit_chi(self):
        """Build the BERTopic model for Chinese text with the created corpus
        and dictionary.
        """

        j_pro_docs = [" ".join(doc) for doc in self.pro_docs]

        if self.embed:
            self.bmodel = BertModel.from_pretrained('bert-base-chinese')
            self.model = BERTopic(language='chinese (traditional)', 
                                  calculate_probabilities=True,
                                  embedding_model=self.bmodel,
                                  nr_topics=self.num_topics)
        else:
            self.model = BERTopic(language='chinese (traditional)', 
                                  calculate_probabilities=True,
                                  nr_topics=self.num_topics)

        _, _ = self.model.fit_transform(j_pro_docs)


    def show_topics(self):
        """Shows the topics with their keywords from the built BERTopic model.
        """

        print("\nTopics from BERTopic Model:")
        for topic in self.model.get_topic_freq().Topic:
            if topic == -1: continue
            twords = [word for (word, _) in self.model.get_topic(topic)]
            print(f"Topic {topic}: {' | '.join(twords)}")


    def pre_evaluate(self):
        """Prepare the original documents per built topic for model evaluation.
        """

        doc_df = pd.DataFrame({"Document": self.docs,
                       "ID": range(len(self.docs)),
                       "Topic": self.model.topics_})
        documents_per_topic = doc_df.groupby(['Topic'], \
                             as_index=False).agg({'Document': ' '.join})
        self.cleaned_docs = self.model._preprocess_text(\
                              documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        self.bt_vectorizer = self.model.vectorizer_model
        self.bt_analyzer = self.bt_vectorizer.build_analyzer()

    def evaluate(self):
        """Computes and outputs the coherence score.
        """

        try:
            self.pre_evaluate()

            # Extract features for Topic Coherence evaluation
            # words = self.bt_vectorizer.get_feature_names_out()
            tokens = [self.bt_analyzer(doc) for doc in self.cleaned_docs]

            self.dictionary = corpora.Dictionary(tokens)
            self.corpus = [self.dictionary.doc2bow(doc) for doc in tokens]

            topic_words = [[words for words, _ in self.model.get_topic(topic)] 
                            for topic in range(len(set(self.model.topics_))-1)]

            coherence = CoherenceModel(topics=topic_words, texts=tokens, corpus=self.corpus, 
                                                            dictionary=self.dictionary, coherence='c_v')\
                        .get_coherence()

            if math.isnan(coherence):
                print("** No coherence score computed!")
            else:
                print(f"  Coherence: {coherence}")
        except:
            print("** No coherence score computed!")

    def viz(self):
        """Visualize the built BERTopic model through Intertopic Distance Map,
        Topic Word Score Charts, and Topic Similarity Matrix.
        """

        print("\nBERTopic Model Visualization:")

        # Intertopic Distance Map
        try:
            self.model.visualize_topics().show()
        except:
            print("** No Intertopic Distance Map shown for your text!")

        # Visualize Terms (Topic Word Scores)
        try:
            self.model.visualize_barchart().show()
        except:
            print("** No chart of Topic Word Scores shown for your text!")

        # Visualize Topic Similarity
        try:
           self.model.visualize_heatmap().show()
        except:
            print("** No heatmap of Topic Similarity shown for your text!")

        print("  If no visualization is shown,")
        print("    you may execute the following commands one-by-one:")
        print("      btm.model.visualize_topics()")
        print("      btm.model.visualize_barchart()")
        print("      btm.model.visualize_heatmap()")
        print()

# End of BTM Class


def btm_process(doc_file, source=0, text_col='text', cat=0, chi=False, group=True, eval=False):
    """Pipelines the BERTopic modeling.

    :param doc_file: The filename of the prescribed text file to be loaded,
        default to None
    :type doc_file: str
    :param source: The source of the prescribed document file ('doc_file'),
        where 0 refers to internal store of the package and 1 to external file,
        default to 0
    :type source: int, optional
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :param cat: The category indicating a subset of the Scripture to be loaded, where
        0 stands for the whole Bible, 1 for OT, 2 for NT, or one of the ten categories
        ['tor', 'oth', 'ket', 'map', 'mip', 'gos', 'nth', 'pau', 'epi', 'apo'] (See 
        the package's internal file 'data/book_cat.csv'), default to 0
    :type cat: int or str, optional
    :param chi: The flag indicating whether the text is processed as Chinese (True)
        or English (False), default to False
    :type chi: bool, optional
    :param group: The flag indicating whether the loaded text is grouped by chapter,
        default to True
    :type group: bool, optional
    :param eval: The flag indicating whether the model evaluation results will be shown,
        default to False
    :type eval: bool, optional
    :return: The pipelined BTM
    :rtype: cwordtm.tm.BTM object
    """

    btm = BTM(doc_file, chi)
    if source == 0:
        btm.docs = load_bible(btm.textfile, cat=cat, group=group)
    else:
        btm.docs = load_text(btm.textfile, text_col=text_col)

    print("Corpus loaded!")

    if chi:
        btm.preprocess_chi()
        print("Chinese text preprocessed!")
        btm.fit_chi()
    else:
        btm.preprocess()
        print("Text preprocessed!")
        btm.fit()

    print("Text trained!")

    btm.show_topics()

    if eval:
        print("\nModel Evaluation Scores:")
        btm.evaluate()

    btm.viz()

    return btm

# End of cwordtm.tm Module
