# viz.py
#
# Show a wordcloud for a precribed range of Scripture
#
# Copyright (c) 2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Updated: 4 June 2024 (0.6.4), 17 Nov 2024
#
# URL: https://github.com/drjohnnycheng/cwordtm.git
# For license information, see LICENSE.TXT

import numpy as np
import pandas as pd
from importlib_resources import files
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

from . import util


def plot_cloud(wordcloud, figsize, web_app=False):
    """Plot the prepared 'wordcloud'

    :param wordcloud: The WordCloud object for plotting, default to None
    :type wordcloud: WordCloud object
    :param figsize: Size (width, height) of word cloud, default to None
    :type figsize: tuple
    :param web_app: The flag indicating the function is initiated from a web
        application, default to False
    :type web_app: bool
    :return: The wordcloud figure
    :rtype: matplotlib.pyplot.figure
    """

    fig = plt.figure(figsize=figsize)
    plt.axis("off");
    plt.imshow(wordcloud) 
    if web_app: return fig


def show_wordcloud(docs, clean=False, figsize=(12, 8), bg='white', image=0, web_app=False):
    """Prepare and show a wordcloud

    :param docs: The collection of documents for preparing a wordcloud,
        default to None
    :type docs: pandas.DataFrame
    :param clean: The flag whether text preprocessing is needed,
        default to False
    :type clean: bool, optional
    :param figsize: Size (width, height) of word cloud, default to (12, 8)
    :type figsize: tuple, optional
    :param bg: The background color (name) of the wordcloud, default to 'white'
    :type bg: str, optional
    :param image: The filename of the presribed image as the mask of the wordcloud,
        or 1/2/3/4 for using an internal image (heart / disc / triangle / arrow),
        default to 0 (No image mask)
    :type image: int or str, optional
    :param web_app: The flag indicating the function is initiated from a web
        application, default to False
    :type web_app: bool
    :return: The wordcloud figure
    :rtype: matplotlib.pyplot.figure
    """

    masks = ['heart.jpg', 'disc.jpg', 'triangle.jpg', 'arrow.jpg']
 
    if image == 0:
        mask = None
    elif image in [1, 2, 3, 4]:  # Internal image file
        img_file = files('cwordtm.images').joinpath(masks[image-1])
        mask = np.array(Image.open(img_file))
    elif isinstance(image, str) and len(image) > 0:
        mask = np.array(Image.open(image))
    else:
        mask = None

    if isinstance(docs, pd.DataFrame):
        docs = ' '.join(list(docs.text.astype(str)))
    elif isinstance(docs, pd.Series):
        docs = ' '.join(list(docs.astype(str)))
    elif isinstance(docs, list) or isinstance(docs, np.ndarray):
        docs = ' '.join(str(doc) for doc in docs)

    if clean:
        docs = util.preprocess_text(docs)

    # wordcloud = WordCloud(background_color=bg, colormap='Set2', mask=mask) \
    wordcloud = WordCloud(background_color=bg, colormap='rainbow', mask=mask) \
                    .generate(docs)

    return plot_cloud(wordcloud, figsize=figsize, web_app=web_app)


def chi_wordcloud(docs, figsize=(15, 10), bg='white', image=0, web_app=False):
    """Prepare and show a Chinese wordcloud

    :param docs: The collection of Chinese documents for preparing a wordcloud,
        default to None
    :type docs: pandas.DataFrame
    :param figsize: Size (width, height) of word cloud, default to (15, 10)
    :type figsize: tuple, optional
    :param bg: The background color (name) of the wordcloud, default to 'white'
    :type bg: str, optional
    :param image: The filename of the presribed image as the mask of the wordcloud,
        or 1/2/3/4 for using an internal image (heart / disc / triangle / arrow),
        default to 0 (No image mask)
    :type image: int or str, optional
    :param web_app: The flag indicating the function is initiated from a web
        application, default to False
    :type web_app: bool
    :return: The wordcloud figure
    :rtype: matplotlib.pyplot.figure
    """

    util.set_lang('chi')
    diction = util.get_diction(docs)

    masks = ['heart.jpg', 'disc.jpg', 'triangle.jpg', 'arrow.jpg']

    if image == 0:
        mask = None
    elif image in [1, 2, 3, 4]:  # Internal image file
        img_file = files('cwordtm.images').joinpath(masks[image-1])
        mask = np.array(Image.open(img_file))
    elif isinstance(image, str) and len(image) > 0:
        mask = np.array(Image.open(image))
    else:
        mask = None

    font_file = files('cwordtm.data').joinpath('msyh.ttc')
    # wordcloud = WordCloud(background_color=bg, colormap='Set2', 
    wordcloud = WordCloud(background_color=bg, colormap='rainbow', 
                          mask=mask, font_path=str(font_file)) \
                    .generate_from_frequencies(frequencies=diction)

    return plot_cloud(wordcloud, figsize=figsize, web_app=web_app)
