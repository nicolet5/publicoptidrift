import numpy as np
import os
import pandas as pd
from optidrift import lassofeatsel
from optidrift import model
from unittest.mock import patch
from random import randint


@patch('optidrift.model.input', create=True)
def test_build_model(mocked_input):
    """test whether build_model can take a train and validation set, build an SVR
model, and pickle it. """

    mocked_input.side_effect = ['test1.sav', 'y']

    a = []
    for i in range(50):
        a.append(randint(0, 10000))

    b = []
    for i in range(50):
        b.append(10 * i)

    df = pd.DataFrame({'A': a, 'B': b})
    df.index = df.index.map(str)
    train = df['0': '40']
    val_set = df['41': '45']
    features = ['B']
    obj = 'A'
    df_val, savepickleas = model.build_model(train, val_set, obj, features)
    assert savepickleas == 'test1.sav', 'Cannot return a sav file'


@patch('optidrift.model.input', create=True)
def test_retest_model(mocked_input):
    """Test whether retest_model can test the model on data that may or may not be
calibrated, use this function to see if the model retains the
accurate levels when the sensor begins to drift."""

    a = []
    for i in range(50):
        a.append(randint(0, 10000))

    b = []
    for i in range(50):
        b.append(10 * i)

    df = pd.DataFrame({'A': a, 'B': b})
    df.index = df.index.map(str)
    features = ['B']
    obj = 'A'
    mocked_input.side_effect = ['1', '2']
    result2 = model.retest_model('test1.sav', features, df, obj, '1', '2')
    assert np.shape(result2) == (2, 4), 'cannot return test result'
