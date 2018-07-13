import pandas as pd
import numpy as np
import os
from unittest.mock import patch
# os.chdir('./..')
from optidrift import getdata

# optidrift.getdata.input


@patch('optidrift.getdata.input', create=True)
def test_process_data(mocked_input):
    """
    Testing the processing of the data
    after the cleaning step and user inputs
    """

    filepath = './../data/plant1/h_data/'
    obj = 'CH1CDWFLO'
    assert os.path.exists(filepath), 'filepath does not exist'

    dss = getdata.load_all_data(filepath)
    assert len(dss) != 0, 'data was not loaded properly'
    df = getdata.clean_data(dss, obj)
    # assert (df[obj].isnull().values.any() == False), ('data not' +
    #                                                  ' properly cleaned')
    mocked_input.side_effect = ['y', '2015-06-30 ', '2015-07-15', 'n']
    df = getdata.remove_time_slices(df, obj)
    assert str(df.index[0]) == '2015-07-15 00:00:00', 'time slice not removed'


def test_clean_data():
    """
    Testing the cleaning of the data
    from raw data.

    """
    df = pd.DataFrame({'A': [np.nan, 1, 2], 'timestamp': [
                      '2018-06-07', '2018-06-08', '2018-06-09']})
    df = getdata.clean_data(df, 'A')
    assert np.shape(df) == (2, 1), 'cannot remove NA value in the given subset'
    assert str(
        type(
            df.index)) == ("<class 'pandas.core.indexes" +
                           ".datetimes.DatetimeIndex'>")
    'cannot change index of dataframe to datetime'


def test_load_all_data():
    """
    Testing the loading of the data
    from file.

    """
    print(os.getcwd())
    filepath = './../data/plant1/h_data/'
    df = getdata.load_all_data(filepath)
    assert len(df) != 0, 'cannot load data'


@patch('optidrift.getdata.input', create=True)
def test_remove_time_slices(mocked_input):
    """
    Testing the remove time slices of the data
    from cleaning data.

    """
    df = pd.DataFrame({'A': [np.nan, 1, 2], 'timestamp': [
                      '2018-06-07', '2018-06-08', '2018-06-09']})
    mocked_input.side_effect = ['y', '2018-06-06', '2018-06-09', 'n']
    df = getdata.clean_data(df, 'A')
    df = getdata.remove_time_slices(df, 'A')
    assert np.shape(df) == (1, 1), 'cannot remove time slices'
