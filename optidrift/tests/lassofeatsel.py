import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

############################
# Wrapper Function
############################


def Lasso_wrapper(df, obj, chosen_alpha):
    """ This is the wrapper function to find features"""

    coef_data = FindFeatures(df, obj, chosen_alpha)
    tryagain = 'y'
    while tryagain == 'y':
        tryagain = input('Would you like to attempt another alpha? (y/n): ')
        if tryagain == 'y':
            new_alpha = float(input('Please input the new alpha: '))
            coef_data = FindFeatures(df, obj, new_alpha)
        else:
            None
    features = list(coef_data.columns)

    return features

#########################
# Component Functions
#########################


def find_nonNAcolumns(df1, df2):
    """This function finds the columns that both df1 and
    df2 do not have any NaN values in."""
    nonnacols_df1 = []
    for i in range(len(df1.columns)):
        col = list(df1)[i]
        if df1[col].isnull().sum() == 0:
            nonnacols_df1.append(col)

    nonnacols_df2 = []

    for i in range(len(df2.columns)):
        col = list(df2)[i]
        if df2[col].isnull().sum() == 0:
            nonnacols_df2.append(col)

    nonnacols = set(nonnacols_df1).intersection(set(nonnacols_df2))

    return nonnacols


def FindFeatures(df, obj, chosen_alpha):
    """obj is the thing we are trying to build a model for,
    this function finds the features that contributes to obj"""

    featurenames = []
    for i in range(len(list(df))):
        col = list(df)[i]
        if df[col].isnull().sum() == 0:
            featurenames.append(col)

    featurenames = list(featurenames)

    train, test = train_test_split(df, test_size=0.2, random_state=1011)
    train_std = train.std()
    test_std = test.std()

    index_train = []

    for i in range(len(train_std)):
        if train_std[i] > 0.0001:
            index_train.append(i)

    index_test = []

    for i in range(len(test_std)):
        if test_std[i] > 0.0001:
            index_test.append(i)

    index = list(set(index_train).intersection(index_test))

    train = train[train.columns[index].values]
    test = test[test.columns[index].values]
    train_normalized = train / train.std()
    test_normalized = test / test.std()

    # will occasionally get NaN values in the train_norm and
    # test_norm dataframes, must clean those out
    # need to keep the columns that don't have NaN values for either
    # train_norm or test_norm

    nonnacols = find_nonNAcolumns(train_normalized, test_normalized)
    # sets the non-na coloms (AKA the ones that we will use as
    # descriptors) as the intersection between the non nas in the test
    # and the non nas in the train

    featurenames = list(nonnacols)

    i = featurenames.index(obj)
    del featurenames[i]
    # removing from the featurename list the descriptor that is the thing we
    # are trying to predict

    coefs = []
    trainerror = []
    testerror = []

    lambdas = np.logspace(-3, 0, 30)
    model = linear_model.Lasso()

    # loop over lambda values (strength of regularization)

    for l in lambdas:
        model.set_params(alpha=l, max_iter=1e6)
        model.fit(train_normalized[featurenames], train_normalized[obj])
        coefs.append(model.coef_)
        trainerror.append(
            mean_squared_error(
                train_normalized[obj],
                model.predict(
                    train_normalized[featurenames])))
        testerror.append(
            mean_squared_error(
                test_normalized[obj],
                model.predict(
                    test_normalized[featurenames])))

        if isclose(l, chosen_alpha, rel_tol=0.1):
            print("The error at the selected alpha is: " +
                  str(mean_squared_error(test_normalized[obj],
                                         model.predict(
                      test_normalized[featurenames]))))
            # this is to print the error close to the selected alpha just to
            # have a number

    lambdas = pd.DataFrame(lambdas, columns=['lambdas'])
    testerror = pd.DataFrame(testerror, columns=['testerror'])
    trainerror = pd.DataFrame(trainerror, columns=['trainerror'])
    # coefs = pd.DataFrame(coefs, columns = ['coefs'])

    plt.figure(figsize=(20, 6))
    # plt.locator_params(nbins = 5)
    plt.subplot(121)
    plt.plot(lambdas, coefs)
    plt.axvline(x=chosen_alpha, color='red', linestyle='dashed')
    plt.xscale('log')
    plt.xlabel('$\lambda$', fontsize=28)
    plt.ylabel('coefs', fontsize=28)
    plt.title('RR coefs vs $\lambda$', fontsize=28)
    # plt.legend()

    plt.subplot(122)
    plt.plot(lambdas, trainerror, label='train error')
    plt.plot(lambdas, testerror, label='test error')
    plt.axvline(x=chosen_alpha, color='red', linestyle='dashed')
    plt.xscale('log')
    plt.xlabel('$\lambda$', fontsize=28)
    plt.ylabel('error', fontsize=28)
    plt.legend(loc=1)
    plt.title('error vs $\lambda$', fontsize=28)
    plt.legend(fontsize=28)
    plt.ylim(0, 1)
    plt.show()

    model.set_params(alpha=chosen_alpha, max_iter=1e8)
    model.fit(train_normalized[featurenames], train_normalized[obj])
    coef_data = pd.DataFrame(
        np.reshape(
            model.coef_,
            (1,
             len(featurenames))),
        columns=featurenames)

    coef_data.replace(0, np.nan, inplace=True)
    coef_data.dropna(axis=1, how='any', inplace=True)
    print('The features at the selected alpha are: ' + str(coef_data.columns))

    return coef_data


def edit_features(feature_set, df):
    """Takes a set of features (obtained from LASSO) and gains user input on
    how to change the set of features or to keep it as is."""

    print('These are the features that LASSO selected: ' + str(feature_set))
    change_feats = 'y'

    while change_feats == 'y':
        change_feats = input('Would you like to change the features? (y/n): ')
        if change_feats == 'y':
            add_feats = input(
                'Would you like to add or remove features? (add/rm): ')
            if add_feats == 'add':
                edit = input(
                    'Please input the feature you would like to add: ')
                if edit in df.columns and edit not in feature_set:
                    feature_set.append(edit)
                    print(
                        'Here is the new feature set with that one added: ' +
                        str(feature_set))
                if edit not in df.columns:
                    print('The specified feature is not a column name '
                          + 'of the data.')
            elif add_feats == 'rm':
                edit = input(
                    'Please input the feature you would like to remove: ')
                if edit in feature_set:
                    feature_set.remove(edit)
                    print(
                        'Here is the new feature set with that one removed: ' +
                        str(feature_set))
                else:
                    print('That feature is already not in the list')

    print('Here is the final feature set: ' + str(feature_set))

    return feature_set
