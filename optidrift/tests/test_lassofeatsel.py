import pandas as pd
import numpy as np
from unittest.mock import patch
import os
# os.chdir('./..')
from optidrift import lassofeatsel
from random import randint


@patch('optidrift.lassofeatsel.input', create=True)
def test_Lasso_wrapper(mocked_input):
    """test function to ensure the wrapper function
    is properly working"""

    df = pd.read_csv(
        './../data/sample_unittest_data.csv',
        index_col='timestamp')

    valdf = df['2015-07-01':'2015-07-03']
    traindf = df['2015-07-04':'2015-07-05']
    obj = 'CH1CDWFLO'
    chosen_alpha = 0.1

    assert isinstance(obj, str), 'Date must be a string'
    assert isinstance(chosen_alpha, float), 'Alpha must be a float'

    mocked_input.side_effect = ['n']
    coef_data = lassofeatsel.FindFeatures(valdf, traindf, obj, chosen_alpha)
    assert isinstance(coef_data, pd.core.frame.DataFrame)

    features = list(coef_data.columns)
    assert len(features) != 0, 'LASSO features were not computed'


def test_find_nonNAcolumns():
    """obj is the thing we are trying to build a model for,
    test whether this function can find the freatures that
    contributes to obj"""

    df1 = pd.DataFrame({'A': [np.nan, 1, 2], 'B': [3, 1, 2], 'C': [2, 1, 5]})
    df2 = pd.DataFrame({'A': [5, 1, 2], 'B': [3, 1, 2], 'C': [2, 1, np.nan]})
    result1 = lassofeatsel.find_nonNAcolumns(df1, df2)
    assert (next(iter(result1)) ==
            'B'), ('Cannot find the columns that both df1 ' +
                   'and df2 do not have any NaN values in.')


def test_FindFeatures():
    """Test whether this function can find
    the features that contribute to obj"""

    a = []
    for i in range(100):
        a.append(randint(0, 10000))

    b = []
    for i in range(100):
        b.append(10 * i)

    c = []
    for i in range(100):
        c.append(100 * i)

    df = pd.DataFrame({'A': a, 'B': b, 'C': c})

    result2 = lassofeatsel.FindFeatures(df, df, 'C', 0.1)

    assert result2.columns[0] == 'B', ('Cannot find ' +
                                       'the features that contribute to obj')


@patch('optidrift.lassofeatsel.input', create=True)
def test_edit_features(mocked_input):
    """Test whether this function can take a set of features
    (obtained from LASSO) and gain user input on
    how to change the set of features or to keep it as is."""
    a = []
    for i in range(100):
        a.append(randint(0, 10000))

    b = []
    for i in range(100):
        b.append(10 * i)

    c = []
    for i in range(100):
        c.append(100 * i)

    df = pd.DataFrame({'A': a, 'B': b, 'C': c})

    feature_set = ['A', 'B']
    mocked_input.side_effect = ['y', 'rm', 'A', 'y', 'add', 'C', 'n']
    result3 = lassofeatsel.edit_features(feature_set, df)
    assert result3 == ['B', 'C'], 'Users cannot change the set of features'


def test_svm_error():
    """"""
    df = pd.read_csv(
        './../data/sample_unittest_data.csv',
        index_col='timestamp')

    valdf = df['2015-07-01':'2015-07-03']
    traindf = df['2015-07-04':'2015-07-05']
    obj = 'CH1CDWFLO'
    features = ['CDWP1SPD', 'CDWP1kW', 'CH1kW']

    assert isinstance(obj, str), 'Date must be a string'
    assert len(features) != 0, 'LASSO features were not computed'

    val, train = lassofeatsel.svm_error(traindf, valdf, obj, features)

    assert isinstance(val, float)
    assert isinstance(train, float)
    assert val >= 0, 'absolute mean error cannot be negative'
    assert train >= 0, 'absolute mean error cannot be negative'
