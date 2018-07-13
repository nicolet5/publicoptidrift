import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import isclose
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn import preprocessing, svm

############################
# Wrapper Function
############################


def Lasso_wrapper(valdf, traindf, obj, chosen_alpha):
    """This is the wrapper function to obtain the important
    features, as selected by LASSO. The inputs are the validation
    dataframe (valdf), the training dataframe (traindf), the
    sensor of interest (obj), and an initial alpha (chosen_alpha).
    This alpha can be changed through the user inputs. This function
    returns the feature set chosen as a list. """

    coef_data = FindFeatures(valdf, traindf, obj, chosen_alpha)
    # this returns the coefficient data obtained from lasso using
    # the initial alpha. Below, the user is given the opportunity
    # to re-run the FindFeatures function with a new alpha.
    tryagain = 'y'
    while tryagain == 'y':
        tryagain = input('Would you like to attempt another alpha? (y/n): ')
        if tryagain == 'y':
            new_alpha = float(input('Please input the new alpha: '))
            coef_data = FindFeatures(valdf, traindf, obj, new_alpha)
        else:
            None
    features = list(coef_data.columns)
    # the columns of the coef_data are the features LASSO chose.
    # the weights of these coefficients can also be gotten from
    # the coef_data variable.

    return features

#########################
# Component Functions
#########################


def find_nonNAcolumns(df1, df2):
    """This function finds the columns that both dataframes
    do not have any NaN values in. Inputs are the two
    dataframes (df1, df2). This function returns the
    non-NA columns as a list."""
    nonnacols_df1 = []
    # initializes empty list to store the non-NA column names
    # in for df1
    for i in range(len(df1.columns)):
        # looping over the columns in df1
        col = list(df1)[i]
        if df1[col].isnull().sum() == 0:
            # this is counting the instances of "NA" in the column
            # (df1[col]) and appending them to the nonnacols_df1
            # list only if the sum is 0.
            nonnacols_df1.append(col)

    nonnacols_df2 = []
    # initializes empty list to store the non-NA column names
    # in for df2
    for i in range(len(df2.columns)):
        # looping over the columns in df2
        col = list(df2)[i]
        if df2[col].isnull().sum() == 0:
            # this is counting the instances of "NA" in the column
            # (df2[col]) and appending them to teh nonnacols_df2
            # list only if the sum is 0.
            nonnacols_df2.append(col)
    # then we find the intersection between the two lists,
    # nonnacols_df1 and nonnacols_df2, which gives the column
    # names where both df1 and df2 have no NA values.
    nonnacols = set(nonnacols_df1).intersection(set(nonnacols_df2))
    # returns nonnacols as a list
    return nonnacols


def FindFeatures(valdf, traindf, obj, chosen_alpha):
    """This function finds the features that contribute to the
    sensor we are trying to build a model for (obj). The
    valdf is the validation dataframe, traindf is the training
    dataframe, and the chosen_alpha is the alpha to be used in
    the LASSO."""
    df = traindf

    # this section grabs all the column names that have no
    # null values to feed into LASSO as all of the features
    featurenames = []
    for i in range(len(list(df))):
        col = list(df)[i]
        if df[col].isnull().sum() == 0:
            featurenames.append(col)

    featurenames = list(featurenames)

    # train, test = train_test_split(df, test_size=0.2, random_state=1011)
    train = df
    test = valdf
    # Instead of a test, train split, will be doing the val df as test
    # we do this because we are interested in how well the features
    # selected in one month can predict the next month.

    train_std = train.std()
    test_std = test.std()

    index_train = []
    # The below for loop is grabbing the columns that have a standard
    # deviation greater than 0.0001. This is because if the column value
    # isn't changing over the month, we don't need it to be fed into
    # LASSO because it doesn't add any new information and when we
    # normalize by the standard deviation, it will be a number close
    # to infinity.
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
    # this is how we get the values for the coef weight plot.
    for l in lambdas:
        model.set_params(alpha=l, max_iter=1e6)
        model.fit(train_normalized[featurenames], train_normalized[obj])
        coefs.append(model.coef_)
    lambdasdf = pd.DataFrame(lambdas, columns=['lambdas'])

    # set up the plot for the lasso coefficient weights, this
    # shows along with the other subplot (further down in the code)
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    plt.plot(lambdasdf, coefs)
    plt.axvline(x=chosen_alpha, color='red', linestyle='dashed')
    plt.xscale('log')
    plt.xlabel(str(chr(945)), fontsize=20)
    plt.ylabel('coefficient weights', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('LASSO coefficient weights vs ' + str(chr(945)), fontsize=20)

    # have to loop again for SVM errors - this could be optimized but
    # wasn't working all as 1 for loop when initialized
    for l in lambdas:
        model.set_params(alpha=l, max_iter=1e8)
        model.fit(train_normalized[featurenames], train_normalized[obj])
        coef_data = pd.DataFrame(
            np.reshape(
                model.coef_,
                (1,
                 len(featurenames))),
            columns=featurenames)
        # this was just looping through the lambda (alpha) values,
        # to get the coefficient data out of the model
        coef_data.replace(0, np.nan, inplace=True)
        coef_data.dropna(axis=1, how='any', inplace=True)
        # drop those coefs that have weights of 0
        features = list(coef_data.columns)
        # svm model function here
        if len(features) > 0:
            # can only run SVM if there is at least 1 feature
            # want to run svm a few times with each set of features to
            # get a good average error
            val_abserrors = []
            train_abserrors = []
            # getting a list of errors to plot at the end
            # we had to loop over 20 svm models for each value of
            # alpha/lambda in order to get consistent graphs for
            # alpha selection
            for i in range(20):
                svmvalerror, svmtrainerror = svm_error(
                    train, test, obj, features)
                i += 1
                val_abserrors.append(svmvalerror)
                train_abserrors.append(svmtrainerror)
            test_mean_abs_error = sum(val_abserrors) / len(val_abserrors)
            train_mean_abs_error = sum(train_abserrors) / len(train_abserrors)
        else:
            train_mean_abs_error = 0
            test_mean_abs_error = 0

        trainerror.append(train_mean_abs_error)
        testerror.append(test_mean_abs_error)

    lambdasdf = pd.DataFrame(lambdas, columns=['lambdas'])
    testerror = pd.DataFrame(testerror, columns=['testerror'])
    trainerror = pd.DataFrame(trainerror, columns=['trainerror'])

    # now we set up the second subplot and show both plots
    plt.subplot(122)
    plt.plot(lambdasdf, trainerror, label='train error')
    plt.plot(lambdasdf, testerror, label='test error')
    plt.axvline(x=chosen_alpha, color='red', linestyle='dashed')
    plt.xscale('log')
    plt.xlabel(str(chr(945)), fontsize=20)
    plt.ylabel('SVM Mean Absolute Error (gpm)', fontsize=20)
    plt.legend(loc=1)
    plt.title('SVM MAE vs ' + str(chr(945)), fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=28)
    plt.show()

# grab the features at the selected alpha. This runs the model fit
# again. Theres probably a more efficient way to do this so that the
# model fit only has to be done once at each alpha.
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
    how to change the set of features or to keep it as is. Inputs are the
    initial featureset and the dataframe the features are being pulled from,
    so that the suggested features must exist in that dataframe's column names.
    This returns the edited feature set."""

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


def svm_error(train, val_set, obj, features):
    """This function is modified from the build_model function in
    the model module. The inputs are the training data set (train),
    the validation set (val_set), and the sensor the model is for
    (obj), and the feature set fed into the svm. This is the function
    behind the right-most sublplot during the feature selection
    process. This returns the mean validation absolute error and the
    mean training absolute error."""
    val_set = val_set.dropna(subset=features)
    train = train.dropna(subset=features)

    train_y = train[obj]
    val_y = val_set[obj]

    train_x = train[features]
    val_x = val_set[features]

    lin_svr = svm.LinearSVR().fit(train_x, train_y)

    predict = lin_svr.predict(val_x)

    df_val = pd.DataFrame(val_y)
    df_val['Predicted'] = predict
    df_val['Absolute Error'] = abs(df_val['Predicted'] - df_val[obj])
    # this is the mean abs error of the validation set
    val_mean_abs_error = df_val['Absolute Error'].mean()

    train_pred = lin_svr.predict(train_x)
    df_train = pd.DataFrame(train_y)
    df_train['Predicted'] = train_pred
    df_train['Absolute Error'] = abs(df_train['Predicted'] - df_train[obj])

    train_mean_abs_error = df_train['Absolute Error'].mean()

    return val_mean_abs_error, train_mean_abs_error
