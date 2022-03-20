Usage
=====

.. _installation:

Installation
------------

To use gmapi, first install it using git clone:

.. code-block:: console

   git clone https://github.com/iaea-nds/GMAP-Python 


Creating recipes
----------------

Here is an example of function documentation.
Try out the awesome function ``gmapi.new_gls_update()``.

.. py:function:: gmapi.new_gls_update(priortable, exptable, refvalues)

    Perform the GLS method to obtain evaluated estimates and uncertainties.

    :param priortable: dataframe with prior quantities
    :type priortable: DataFrame
    :return: A dictionary with ``upd_vals`` and ``upd_cov``.
    :type: dictionary


Here is the real deal:

.. autofunction:: gmapi.inference.new_gls_update

