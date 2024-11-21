# util.py
#    
# Some utility functions including loading Scriptue, setting Scripture language,
#   extracting a specifc range of Scripture
#
# Copyright (c) 2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Updated: 16 June 2024 (0.6.4), 1 Nov 2024
#
# URL: https://github.com/drjohnnycheng/cwordtm.git
# For license information, see LICENSE.TXT


import re
import string
from io import BytesIO
import numpy as np
import pandas as pd
from importlib_resources import files

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import jieba
from collections import Counter


chi_flag = False
glang = 'en'
stops = set()


def is_chi():
    """Checks whether the Chinese language flag is set.

    :return: True if the Chinese language flag (chi_flag) is set,
        False otherwise
    :rtype: bool
    """

    return chi_flag


def bible_cat_info(lang='en'):
    """Prints a table of Bible book categories with their books.

    :param lang: The language of the information to be shown, default to "en"
    :type lang: str, optional
    :return: The table of Bible book categories
    :rtype: pandas.DataFrame
    """

    if lang not in ['en',  'chi']:
        return  "The language should be either English ('en') or Chinese ('chi')"

    cat_file = 'category_chi.csv' if lang=='chi' else 'category.csv'
    cdf = pd.read_csv(files('cwordtm.data').joinpath(cat_file))
    return cdf


def remove_noise(text, noise_list):
    """Removes a list of substrings in noise_list from the input text.

    :param text: The input text, default to None
    :type text: str
    :param noise_list: The list of substrings to be removed, default to ""
    :type noise_list: list, optional
    :return: The text with the prescribed substrings removed
    :rtype: str
    """

    text = text.rstrip()
    for noise in noise_list:
        text = text.replace(noise, '')
    return text


def load_csv(file_obj, nr=0, info=False):
    """Loads a CSV file with a "Text" column.

    :param file_obj: The prescribed filepath from which the text is loaded,
        or a BytesIO object from Streamlit's file_uploader, default to None
    :type file_obj: str or io.BytesIO
    :param nr: The number of rows of text to be loaded; 0 represents all rows,
        default to 0
    :type nr: int, optional
    :param info: The flag whether the dataset information is shown,
        default to False
    :type info: bool, optional
    :return: The collection of text with the prescribed number of rows loaded
    :rtype: pandas.DataFrame
    """

    # print("Loading file '%s' ..." %filepath)
    if isinstance(file_obj, BytesIO) or \
       file_obj.lower().endswith('csv'):
        nrows = None
        if nr > 0: nrows = nr
        df = pd.read_csv(file_obj, nrows=nrows, encoding='utf-8')
    else:
        noise_list = ['\u3000', '─ ', '•']
        tf =  open(file_obj, encoding='utf-8')
        lines = [remove_noise(line, noise_list) for line in tf.readlines()]
        lines = list(filter(None, lines))

        df = pd.DataFrame({'text': lines})
        if nr > 0: df = df.iloc[:nr]

    if info:
        print("\nDataset Information:")
        df.info()

    return df


def load_text(filepath, nr=0, info=False):
    """Loads and returns the text from the prescribed file path ('filepath').

    :param filepath: The prescribed filepath from which the text is loaded,
        default to None
    :type filepath: str
    :param nr: The number of rows of text to be loaded; 0 represents all rows,
        default to 0
    :type nr: int, optional
    :param info: The flag whether the dataset information is shown,
        default to False
    :type info: bool, optional
    :return: The collection of text with the prescribed number of rows loaded
    :rtype: pandas.DataFrame
    """

    # print("Loading file '%s' ..." %filepath)
    if filepath.lower().endswith('csv'):
        nrows = None
        if nr > 0: nrows = nr
        df = pd.read_csv(filepath, nrows=nrows, encoding='utf-8')
    else:
        noise_list = ['\u3000', '─ ', '•']
        tf =  open(filepath, encoding='utf-8')
        lines = [remove_noise(line, noise_list) for line in tf.readlines()]
        lines = list(filter(None, lines))

        df = pd.DataFrame({'text': lines})
        if nr > 0: df = df.iloc[:nr]

    if info:
        print("\nDataset Information:")
        df.info()

    return df


def load_word(ver='web.csv', nr=0, info=False):
    """Loads and returns the text from the prescribed internal file ('ver').

    :param ver: The package's internal Bible text from which the text is loaded,
        either World English Bible ('web.csv') or Chinese Union Version
        (Traditional)('cuv.csv'), default to 'web.csv'
    :type ver: str, optional
    :param nr: The number of rows of Scripture to be loaded; 0 represents all rows,
        default to 0
    :type nr: int, optional
    :param info: The flag whether the dataset information is shown,
        default to False
    :type info: bool, optional
    :return: The collection of Scripture with the prescribed number of rows loaded
    :rtype: pandas.DataFrame
    """

    scfile = files('cwordtm.data').joinpath(ver)
    print("Loading file '%s' ..." %scfile)
    df = pd.read_csv(scfile)
    if nr > 0:
       print("Initial Records:")
       df.head(int(nr))
    if info:
        print("\nDataset Information:")
        df.info()
    return df


def group_text(df, column='chapter'):
    """Groups the Bible Scripture in the DataFrame 'df' by the prescribed column, and
    'df' should include columns 'book', 'book_no', 'chapter', 'verse', 'text',
    'testament', 'category', 'cat', and 'cat_no'.

    :param df: The input DataFrame storing the Scripture, default to None
    :type df: pandas.DataFrame
    :param column: The column by which the Scriture is grouped, default to 'chapter'
    :type column: str, optional
    :return: The grouped Scripture
    :rtype: pandas.DataFrame
    """

    gdf = df.groupby(['book_no', column])\
                        .agg({'text': lambda x: ''.join(x)})\
                .reset_index()
    return gdf


def get_list(df, column='book'):
    """Extracts and returns the prescribed column from the Scripture
    stored in the DataFrame 'df'.

    :param df: The input DataFrame storing the Scripture, default to None
    :type df: pandas.DataFrame
    :param column: The column by which the Scriture is grouped, default to 'book'
    :type column: str, optional
    :return: The grouped Scripture
    :rtype: pandas.DataFrame
    """

    if column in list(df.columns):
        return list(df[column].unique())
    else:
        return "No such column!"


def get_text(df, text_col='text'):
    """Extracts and returns the text from the Scripture
    stored in the DataFrame 'df' after joining the list of text into a string
    and removing all the ideographic spaces ('\u3000') from the text.

    :param df: The input DataFrame storing the Scripture, default to None
    :type df: pandas.DataFrame
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :return: The extracted text
    :rtype: str
    """

    # return ' '.join(list(df[text_col])).replace('\u3000', '')
    return ' '.join(list(df[text_col].astype(str))).replace('\u3000', '')


def get_text_list(df, text_col='text'):
    """Extracts and returns the list of text from the Scripture
    stored in the DataFrame 'df' after removing all the ideographic spaces
    ('\u3000') from the text.

    :param df: The input DataFrame storing the Scripture, default to None
    :type df: pandas.DataFrame
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :return: The extracted text
    :rtype: list
    """

    return df[text_col].apply(lambda x: x.replace('\u3000', '')).tolist()


def clean_text(df, text_col='text'):
    """Cleans the text from the Scripture stored in the DataFrame 'df',
    by removing all digits, replacing newline by a space, removing
    English stopwords, converting all characters to lower case, and
    removing all characters except alphanumeric and whitespace.

    :param df: The input DataFrame storing the Scripture, default to None
    :type df: pandas.DataFrame
    :param text_col: The name of the text column to be extracted, default to 'text'
    :type text_col: str, optional
    :return: The cleaned text in a DataFrame
    :rtype: pandas.DataFrame
    """

    df[text_col] = [re.sub(r'\d+', '', str(v).replace('\n', ' ')) for v in df[text_col]]
    for sw in stopwords.words('english'):
        df[text_col] = [v.replace(' ' + sw + ' ', ' ') for v in df[text_col]]

    df[text_col] = df[text_col].apply(lambda v: " ".join(w.lower() for w in v.split()))
    df[text_col] = df[text_col].str.replace('[^\w\s]', '', regex=True)
    return df


def clean_sentences(sentences):
    """Cleans the list of sentences by invoking the function preprocess_text.

    :param sentences: The list of sentences to be cleaned, default to None
    :type sentences: list
    :return: The list of cleaned sentences
    :rtype: list
    """

    cleaned = []
    for sentence in sentences:
        cleaned_sent = preprocess_text(sentence)
        if len(cleaned_sent) > 0:
            cleaned.append(cleaned_sent)

    return cleaned


def preprocess_text(text):
    """Preprocesses English text by converting text to lower case, removing 
    special characters and digits, removing punctuations, removing stopwords,
    removing short words, and Lemmatize text.

    :param text: The text to be preprocessed, default to None
    :type text: str
    :return: The preprocessed text
    :rtype: str
    """

    if isinstance(text, list) or isinstance(text, np.ndarray):
        text = ' '.join(str(item) for item in text)
    elif isinstance(text, pd.Series):
        text = ' '.join(list(text.astype(str)))

    # print("Preprocessing text ...")

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    text = " ".join([word for word in nltk.word_tokenize(text) \
                    if word.lower() not in stopwords.words('english')])

    # Remove short words (length < 3)
    text = " ".join([word for word in nltk.word_tokenize(text) if len(word) >= 3])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])

    return text


def add_chi_vocab():
    """Loads the Chinese Bible vocabulary from the internal file 'bible_vocab.txt',
    and adds to the Jieba word list for future tokenization
    """

    vocab_file = files('cwordtm.data').joinpath('bible_vocab.txt')
    print("Loading Chinese vocabulary '%s' ..." %vocab_file)
    with open(vocab_file, 'r', encoding='utf8') as f:
        vocab_list = f.readlines()
        for vocab in vocab_list:
            jieba.add_word(vocab.replace('\n', ''), freq=1000)


def chi_stops():
    """Loads the common Chinese (Traditional) vocabulary to Jieba for
    future tokenization, and the Chinese stopwords for future
    wordcloud plotting.

    :return: The list of stopwords for wordcloud plotting
    :rtype: list
    """

    dict_file = files('cwordtm.dictionary').joinpath('dict.txt.big.txt')
    cloud_file = files('cwordtm.dictionary').joinpath('stopWord_cloudmod.txt')
    jieba.set_dictionary(dict_file)
    with open(cloud_file, 'r', encoding='utf-8-sig') as f:
        return f.read().split('\n')


def set_lang(lang='en'):
    """Sets the prescribed language (English or Chinese (Traditional)) 
    for further text processing.

    :param lang: The prescribed language for text processing, where
        'en' stands for English or 'chi' for Traditonal Chinese,
        default to 'en'
    :type lang: str, optional
    """

    global glang, stops
    glang = lang
    if glang == 'en':  # English
        stops = set(stopwords.words("english"))
    else:  # Chinese (Traditional)
        add_chi_vocab()
        stops = chi_stops()
        chi_flag = True


def get_diction_en(docs):
    """Tokenizes the collection of English documents and builds a dictionary
    of words with their frequencies.

    :param docs: The collection of text, default to None
    :type docs: pandas.DataFrame or list
    :return: The dictionary of words with their frequencies
    :rtype: dict
    """

    if isinstance(docs, pd.DataFrame):
        docs = ' '.join(list(docs.text.astype(str)))
    elif isinstance(docs, pd.Series):
        docs = ' '.join(list(docs.astype(str)))
    elif isinstance(docs, list) or isinstance(docs, np.ndarray):
        docs = ' '.join(str(doc) for doc in docs)

    words = word_tokenize(docs)
    stem = PorterStemmer()
    
    terms = []
    for t in words:
        t = stem.stem(t)
        if t not in stops:
            terms.append(t)

    diction = Counter(terms)
    return diction


def get_diction_chi(docs):
    """Tokenizes the collection of Chinese documents and builds a dictionary
    of words with their frequencies.

    :param docs: The collection of documents, default to None
    :type docs: pandas.DataFrame or list
    :return: The dictionary of words with their frequencies
    :rtype: dict
    """

    if isinstance(docs, pd.DataFrame):
        docs = ''.join(list(docs.text.astype(str)))
    elif isinstance(docs, pd.Series):
        docs = ''.join(list(docs.astype(str)))
    elif isinstance(docs, list) or isinstance(docs, np.ndarray):
        docs = ''.join(str(doc) for doc in docs)

    text = docs.replace('\u3000', '')
    text = re.sub("[、．。，！？『』「」〔〕]", "", text)

    terms = []
    for t in jieba.cut(text, cut_all=False):
        if t not in stops:
            terms.append(t)

    diction = Counter(terms)
    return diction


def get_diction(docs):
    """Determines which is the target language, English or Chinese,
    in order to build a dictionary of words with their frequencies.

    :param docs: The collection of documents, default to None
    :type docs: pandas.DataFrame or list
    :return: The dictionary of words with their frequencies
    :rtype: dict
    """

    if glang == 'en':
        return get_diction_en(docs)
    else:
        return get_diction_chi(docs)


def chi_sent_terms(text):
    """Returns the list of Chinese words tokenized from the input text.

    :param text: The input Chinese text to be tokenized, default to None
    :type text: str
    :return: The list of Chinese words
    :rtype: list
    """

    text = re.sub("[、．。，：！？『』「」〔〕]", "", text)
    terms = []
    for t in jieba.cut(text, cut_all=False):
        if t not in stops:
            terms.append(t)
    return terms


def get_sent_terms(text):
    """Determines how to tokenize the input text, based on the global language
    setting, either English ('en') or Traditional Chinese ('chi').

    :param text: The input text to be tokenized, default to None
    :type text: str
    :return: The list of tokenized words
    :rtype: list
    """

    if glang == 'en':
        return word_tokenize(text)
    else:
        return chi_sent_terms(text)


def extract(df, testament=-1, category='', book=0, chapter=0, verse=0):
    """Extracts a subset of the Scripture stored in a DataFrame by testament,
    category, or book/chapter/verse.

    :param df: The collection of the Bible Scripture with columns 'book',
        'book_no', 'chapter', 'verse', 'text', 'testament', 'category',
        'cat', and 'cat_no', default to None
    :type df: pandas.DataFrame
    :param testament: The prescribed testament to be extracted,
        -1 stands for no prescription, 0 for OT, or 1 for NT,
        default to -1
    :type testament: int, optional
    :param category: The prescribed category to be extracted, and
        it should be either a full category name or a short name with
        3 lower-case letters from a list of 10 categories, default to ''
    :type category: str, optional
    :param book: The prescribed Bible book to be extracted, and
        it should be either a 3-letter short book name or a book number
        from 1 to 66, default to 0
    :type book: str, int, optional
    :param chapter: The prescribed chapter or a tuple indicating the range of
        chapters of a Bible book to be extracted, default to 0
    :type chapter: int or tuple, optional
    :param verse: The prescribed verse or a tuple indicating the range of verses
        from a chapter of a Bible book to be extracted, default to 0
    :type verse: int or tuple, optional
    :return: The subset of the input Scripture, if any, otherwise,
        the message 'No scripture is extracted!'
    :rtype: pandas.DataFrame or str
    """

    no_ret = "No scripture is extracted!"
    sub_df = pd.DataFrame()  # Empty DataFrame
    isbook = ischapter = False

    if (testament > -1) & (testament < 2):
        sub_df = df[df.testament==int(testament)]
    elif category != '':
        if category in get_list(df, column='category'):
            sub_df = df[df.category==category]
        elif category in get_list(df, column='cat'):
            sub_df = df[df.cat==category]
    elif book in get_list(df, column='book'):
        sub_df = df[df.book==book]
        isbook = True
    elif isinstance(book, int):
        if book > 0 & book < 67:
            sub_df = df[df.book_no==book]
            isbook = True
    elif isinstance(book, tuple):
        if (book[0] <= book[1]) & (book[0] > 0) & (book[1] < 67):
            sub_df = df[(df.book_no >= book[0]) & (df.book_no <= book[1])]
            isbook = True

    if isbook & (len(sub_df) > 0) & (chapter != 0):
        if isinstance(chapter, int):
            sub_df = sub_df[sub_df.chapter==chapter]
            ischapter = True
        elif isinstance(chapter, tuple):
            if chapter[0] <= chapter[1]:
                sub_df = sub_df[(sub_df.chapter >= chapter[0]) & (sub_df.chapter <= chapter[1])]
                ischapter = True

        if ischapter & (len(sub_df) > 0) & (verse != 0):
            if isinstance(verse, int):
                sub_df = sub_df[sub_df.verse==verse]
            elif isinstance(verse, tuple):
                if verse[0] <= verse[1]:
                    sub_df = sub_df[(sub_df.verse >= verse[0]) & (sub_df.verse <= verse[1])]

    if len(sub_df) > 0:
        return sub_df.copy()
    else:
        return no_ret


def extract2(df, filter=''):
    """Extracts a subset of the Scripture through a specific filter string by
    invoking the function 'util.extract'.

    :param df: The collection of the Bible Scripture, default to None
    :type df: pandas.DataFrame
    :param filter: The prescribed filter string with the format
        '<book> <chapter>:<verse>[-<verse2>]' for extracting a range of verses
        in the Scripture, default to ''
    :type filter: str, optional
    :return: The prescribed range of verses from the input Scripture, or
        the whole Scripture if the filter string is empty
    :rtype: pandas.DataFrame
    """

    chapter = verse = 0

    if filter == '':
        return df
    else:
        parts = filter.split()
        book = parts[0]
        if len(parts) > 1:
            parts = parts[1].split(':')
            if parts[0] == '':
                chapter = 0
            else:
                chapter = int(parts[0])

            if (len(parts) > 1):
                if (parts[1] != ''):
                    parts = parts[1].split('-')
                    if parts[0] == '':
                        verse = 1
                    else:
                        verse = int(parts[0])

                    if (len(parts) > 1):
                        if (parts[1] == ''):
                            verse = (verse, 999)
                        else:
                            verse = (verse, int(parts[1]))

        return extract(df, book=book, chapter=chapter, verse=verse)


def set_rows(n=None):
    """Set the maximum no. of rows of DataFrames to be displayed.

    :param n: The maximum no. of rows to be set, value None denotes that
        all rows are to be displayed, default to None
    :type n: int, optional
    """

    pd.options.display.max_rows = n


def reset_rows():
    """Reset the maximum no. of rows of DataFrames to be displayed to its default value.
    """

    pd.reset_option("display.max_rows")
