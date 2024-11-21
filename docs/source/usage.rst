CWordTM Package (cwordtm 0.7.1)
===============================

A Topic Modeling Toolkit from Low-code to Pro-code

Installation
------------

.. code:: shell

   $ pip install cwordtm

Usage
-----

``cwordtm`` can be used to perform some NLP pre-processing tasks, text
exploration, including Chinese one, text visualization (word cloud), and
topic modeling (BERTopic, LDA and NMF) as follows:

.. code:: python

   from cwordtm import meta, util, ta, tm, viz, pivot, quot

version Submodule
~~~~~~~~~~~~~~~~~

Provides some version information.

.. code:: python

   import cwordtm
   print(cwordtm.__version__)

meta Submodule
~~~~~~~~~~~~~~

Provides extracting source code of ``cwordtm`` module and adding timing and code-showing features
to all functions of the module.

.. code:: python

   print(meta.get_module_info())

   print(meta.get_submodule_info('viz', detailed=True))


quot Submodule
~~~~~~~~~~~~~~

Provides functions to extract the quotation source Scripture in OT based on the presribed NT Scripture.

.. code:: python

   cdf = util.load_word('cuv.csv')
   crom8 = util.extract2(cdf, 'Rom 8')
   
   quot.show_quot(crom8, lang='chi')

pivot Submodule
~~~~~~~~~~~~~~~

Provides a pivot table of the prescribed text.

.. code:: python

   cdf = util.load_word('cuv.csv')

   pivot.stat(cdf, chi=True)

ta Submodule
~~~~~~~~~~~~

Provides text analytics functions, including extracting the summarization of the prescribed text.

.. code:: python

   cdf = util.load_word('cuv.csv')
   crom8 = util.extract2(cdf, 'Rom 8')

   ta.summary_chi(crom8)

tm Submodule
~~~~~~~~~~~~~

Provides text modeling functions, including LDA, NMF and BERTopics modeling.

.. code:: python

   lda = tm.lda_process("web.csv", eval=True, timing=True)

   nmf = tm.nmf_process("web.csv", eval=True, code=1)

   btm = tm.btm_process("cuv.csv", chi=True, cat='ot', eval=True)

   btm = tm.btm_process("cuv.csv", chi=True, cat='nt', eval=True, code=2)

util Submodule
~~~~~~~~~~~~~~

Provides loading text and text preprocessing functions.

.. code:: python

   df = util.load_word()
   cdf = util.load_word('cuv.csv')

   df.head()
   cdf.head()

   rom8 = util.extract2(df, 'Rom 8')
   crom8 = util.extract2(cdf, 'Rom 8')

viz Submodule
~~~~~~~~~~~~~

Wordcloud plotting from the prescribed text.

.. code:: python

   cdf = util.load_word('cuv.csv')

   viz.chi_wordcloud(cdf)

Demo
----

Usage demo file with output:

#. On BBC News: `CWordTM_BBC.pdf <https://github.com/drjohnnycheng/CWordTM/blob/main/Demo/CWordTM_BBC.pdf>`_

#. On Chinese Bible (CUV): `CWordTM_CUV.pdf <https://github.com/drjohnnycheng/CWordTM/blob/main/Demo/CWordTM_CUV.pdf>`_

Paper
-----

For a more detailed overview, you can read the demo paper: https://link.springer.com/chapter/10.1007/978-3-031-70242-6_4

Documentation
-------------

``cwordtm`` documentation can be reached from: https://cwordtm.readthedocs.io

Contributing
------------

Interested in contributing? Check out the contributing guidelines.
Please note that this project is released with a Code of Conduct. By
contributing to this project, you agree to abide by its terms.

License
-------

``cwordtm`` was created by Dr. Johnny Cheng. It is licensed under the terms
of the MIT license.

Credits
-------

``cwordtm`` was created under the guidance of Jehovah, the Almighty God.
