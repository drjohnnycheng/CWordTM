# pivot.py
#    
# Show a pivot table for a precribed range of Scripture
#
# Copyright (c) 2024 CWordTM Project 
# Author: Johnny Cheng <drjohnnycheng@gmail.com>
#
# Created: 24 June 2022
# Updated: 16 June 2024 (0.6.4)
#
# URL: <https://github.com/drjohnnycheng/cwordtm.git>
# For license information, see LICENSE.TXT

import pandas as pd


def stat(df, chi=False):
    """Returns a pivot table from the DataFrame 'df' storing the input Scripture,
    with columns 'book', 'book_no', 'chapter', 'verse', 'text', 'testament',
    'category', 'cat', and 'cat_no'.

    :param df: The DataFrame storing the input Scripture, default to None
    :type df: pandas.DataFrame
    :param chi: If the value is True, assume the input text is in Chinese,
        otherwise, the input text is in English, default to False
    :type chi: bool, optional
    :return: The pivot table of the input Scripture grouped by category ('cat_no')
    :rtype: pandas.DataFrame
    """
 
    stat_df = pd.pivot_table(df, index = ['book_no', 'book', 'category', 'cat_no'],
                          values = ['chapter', 'verse', 'text'],
                          aggfunc = {'chapter': lambda ch: len(ch.unique()),
                                     'verse': 'count',
                                     'text': lambda ts: sum([len(t if chi else t.split()) for t in ts])})

    stat_df = stat_df[['chapter', 'verse', 'text']].sort_index()

    stat_df2 = stat_df.groupby('cat_no').apply(lambda sub: sub.pivot_table(
                        index = ['category', 'book_no', 'book'],
                        values = ['chapter', 'verse', 'text'],
                        aggfunc = {'chapter': 'sum',
                                   'verse': 'sum',
                                   'text': 'sum'},
                        margins = True,
                        margins_name = 'Sub-Total'))

    stat_df2.loc[('', '', 'Total', '')] = stat_df2.sum() // 2
    stat_df2.index = stat_df2.index.droplevel(0)
    stat_df2.fillna('', inplace=True)
    stat_df2 = stat_df2[['chapter', 'verse', 'text']]

    print("Book category information can be shown by invoking 'util.bible_cat_info()'")

    return stat_df2


def pivot(df, column='Category'):
    """Returns a pivot table from the DataFrame 'df' storing the input documents,
    grouped by the prescribed column.

    :param df: The DataFrame storing the input documents, default to None
    :type df: pandas.DataFrame
    :param column: The column to be the group-by column, default to 'Category'
    :type column: str, optional
    :return: The pivot table of the input documents grouped by the prescribed column
    :rtype: pandas.DataFrame
    """
 
    if column is None or not column in df.columns:
        print("No valid column has been specified!")
        return

    stat_df = df.pivot_table(values='Text',
                             index=column,
                             aggfunc='count',
                             margins=True,
                             margins_name='Total'
                            )
    return stat_df
