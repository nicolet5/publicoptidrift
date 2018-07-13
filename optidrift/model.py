import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from sklearn import preprocessing, svm
from lassofeatsel import Lasso_wrapper, edit_features

#####################
# Wrapper Function
#####################


def model_exploration(df, obj):
    """This function is the wrapper function of changing time slices for
    training, validation, and testing sets. It will perform lasso on the
    training data, allow features to be edited, build a model, and test
    the model. Then it will ask if the user would like to explore different
    time slices - this is useful in finding the optimum amount of data
    necessary to build an adequate model. This takes the entire dataframe
    (df) and the sensor to build a model for (obj)"""

    see_another_set = 'y'

    while see_another_set == 'y':
        # this while loop is so we don't have to load and reclean etc every
        # time we want to see a different timeslice of the data

        train_months_start = input('Input the start date of training data: ')
        train_months_end = input('Input the end date of training data: ')
        val_months_start = input('Input the start date of validation data: ')
        val_months_end = input('Input the end date of validation data: ')
        train = df[train_months_start: train_months_end]
        # Training dataframe
        val_set = df[val_months_start: val_months_end]
        # Testing (Validation set)

        feat_mo_og = Lasso_wrapper(val_set, train, obj, 0.1)
        # get features from lasso, with an initial alpha value of 0.1
        # this alpha can be changed by the user during the lasso_wrapper
        # function
        features = edit_features(feat_mo_og, train)
        # this allows the user to change features that don't make sense
        # df_val and df_test might have some NaN values in them for the
        # features selected by LASSO- clean those out
        # val_set = val_set.dropna(subset = features)

        df_val, savepickleas = build_model(train, val_set, obj, features)
        # (ability to catch out of calibration)

        # plot the train, validation:
        fig2 = plt.figure(figsize=(20, 10), facecolor='w', edgecolor='k')
        plt.subplot(211)
        myplot2 = plt.scatter(
            df_val.index,
            df_val[obj],
            color='red',
            label='val data-actual')
        plt.scatter(
            df_val.index,
            df_val.Predicted,
            color='blue',
            label='val data-model',
            alpha=0.5)
        plt.scatter(train.index, train[obj], color='green', label='train data')
        plt.ylabel(obj, fontsize=16)
        plt.xlabel('Index', fontsize=16)
        plt.title('Training, Validation, and Test Model of ' + obj,
                  fontsize=28)
        plt.legend(fontsize=16)
        plt.xlim()

        # plot the absolute error between the model and the test data
        # this is the metric that would be used to "raise an alarm" if sensor
        # begins to drift
        allow_error = input(
            'Please input the allowable error in ' +
            'this sensor (|predicted - actual|): ')
        # this allows the user to set the amount of drift that is acceptable
        # before an alarm should be raised
        plt.subplot(212)
        myplot3 = plt.plot(
            df_val.index,
            df_val['Absolute Error'],
            color='green')
        plt.axhline(y=int(allow_error), color='red', linestyle='dashed',
                    label='Allowable Error')
        plt.ylabel('Absolute Error (sensor dependent unit)', fontsize=16)
        plt.xlabel('Index', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()

        test_yn = input(
            'Would you like to test the model on the month ' +
            'subsequent to the validation data? If that data' +
            ' is not available in the folder, answer "n" (y/n): ')
        if test_yn == 'n':
            None
        else:
            test_initial_start = val_set.index[-1] + timedelta(hours=1)
            test_initial_end = val_set.index[-1] + timedelta(days=30)
        # want the first set of testing data to be after the
        # set validation date range
        # subsequent test sets will be after the training data
            df_test = retest_model(
                savepickleas,
                features,
                df,
                obj,
                test_initial_start,
                test_initial_end)
            # this is testing the model on the test dates - using the
            # test_initial_start and the test_initial_end

            # then we plot the test,train, and validation dataframes:
            plt.figure(figsize=(20, 10), facecolor='w', edgecolor='k')
            plt.subplot(211)
            myplot2 = plt.scatter(
                df_val.index,
                df_val[obj],
                color='red',
                label='val data-actual')
            plt.scatter(
                df_val.index,
                df_val.Predicted,
                color='blue',
                label='val data-model',
                alpha=0.5)
            plt.scatter(
                df_test.index,
                df_test[obj],
                color='purple',
                label='test data-actual',
                alpha=0.5)
            plt.scatter(
                df_test.index,
                df_test.Predicted,
                color='yellow',
                label='test data-model',
                alpha=0.5)
            plt.scatter(
                train.index,
                train[obj],
                color='green',
                label='train data',
                alpha=0.5)
            plt.ylabel(obj, fontsize=16)
            plt.xlabel('Index', fontsize=16)
            plt.title('Training, Validation, and Test Model of ' + obj,
                      fontsize=28)
            plt.legend(fontsize=16)
            plt.xlim()
            plt.subplot(212)
            myplot3 = plt.plot(
                df_test.index,
                df_test['Absolute Error'],
                color='green')
            plt.axhline(y=int(allow_error), color='red', linestyle='dashed',
                        label='Allowable Error')
            plt.ylabel('Absolute Error (sensor dependent unit)', fontsize=16)
            plt.xlabel('Index', fontsize=16)
            plt.legend(fontsize=16)
            plt.show()

            y_n = input(
                'Would you like to remove the out-of-calibration data from ' +
                'the training set, re-train, and predict the ' +
                'following month? (y/n):')
            # if the answer is 'y', this while loop starts, removing data.
            while y_n == 'y':
                df_train_raw = pd.concat([train, df_test])
                df_test = df_test[df_test['Absolute Error'] < int(allow_error)]
                # adding the df_test section where the sensor error is below
                # the allowable error
                add_train = df[df.index.isin(df_test.index)]
                train = pd.concat([train, add_train])
                # adding the "in calibration" data to the training dataframe

                plt.figure(figsize=(20, 4), facecolor='w', edgecolor='k')
                plt.scatter(
                    train.index,
                    train[obj],
                    color='green',
                    label='train data')
                plt.show()
                y_n2 = input(
                    'Is there a date range you would like to add ' +
                    'back in? (y/n): ')
                # this allows the user to add back in any date ranges
                # that were removed because they were above the
                # allowable sensor error.
                # this could probably be streamlined to have the date
                # ranges not removed before the user gives input,
                # since it's easier to see if you want to keep any
                # ranges while you can see them, before they
                # are removed.
                while y_n2 == 'y':
                    start = input('Input the start date: ')
                    end = input('Input the end date: ')
                    add_train2 = df[start:end]
                    train = pd.concat([train, add_train2])
                    train = train.sort_index()
                    plt.figure(figsize=(20, 4), facecolor='w', edgecolor='k')
                    plt.scatter(
                        train.index,
                        train[obj],
                        color='green',
                        label='train data')
                    plt.show()
                    y_n2 = input('Another date range? (y/n): ')
                if y_n2 == 'n':
                    pass
                elif y_n2 != 'y' or 'n':
                    break

                # now we are setting the new test set to thirty days
                # after the training set
                test_nmodel_start = df_train_raw.index[-1] + timedelta(hours=1)
                test_nmodel_end = df_train_raw.index[-1] + timedelta(days=30)

                # leave val set as the same one inputted at first

                feat_mo_og = Lasso_wrapper(val_set, train, obj, 0.1)
                # get the features from LASSO
                features = edit_features(feat_mo_og, train)
                # give the user the option to edit those features from LASSO
                df_val, savepickleas = build_model(
                    train, val_set, obj, features)
                # building the model based off of the training data and those
                # edited features
                df_test = retest_model(
                    savepickleas,
                    features,
                    df,
                    obj,
                    test_nmodel_start,
                    test_nmodel_end)
                # this is testing the model on the test data
                # set bound by test_nmodel_start
                # and test_nmodel_end

                # now we plot the train and test data sets
                plt.figure(figsize=(20, 10), facecolor='w', edgecolor='k')
                plt.subplot(211)
                myplot2 = plt.scatter(
                    df_val.index,
                    df_val[obj],
                    color='red',
                    label='val data-actual')
                plt.scatter(
                    df_val.index,
                    df_val.Predicted,
                    color='blue',
                    label='val data-model',
                    alpha=0.5)
                plt.scatter(
                    df_test.index,
                    df_test[obj],
                    color='purple',
                    label='test data-actual',
                    alpha=0.5)
                plt.scatter(
                    df_test.index,
                    df_test.Predicted,
                    color='yellow',
                    label='test data-model',
                    alpha=0.5)
                plt.scatter(
                    train.index,
                    train[obj],
                    color='green',
                    label='train data',
                    alpha=0.5)
                plt.ylabel(obj, fontsize=16)
                plt.xlabel('Index', fontsize=16)
                plt.title('Training and Testing Model of ' + obj,
                          fontsize=28)
                plt.legend(fontsize=16)
                plt.xlim()
                plt.subplot(212)
                myplot3 = plt.plot(
                    df_test.index,
                    df_test['Absolute Error'],
                    color='green')
                plt.axhline(
                    y=int(allow_error),
                    color='red',
                    linestyle='dashed',
                    label='Allowable Error')
                plt.ylabel(
                    'Absolute Error (sensor dependent unit)',
                    fontsize=16)
                plt.xlabel('Index', fontsize=16)
                plt.legend(fontsize=16)
                plt.show()

                # asking if we would like to repeat, adding on another month
                # of training data and retesting on the next month.
                # can only do this if there is enough data in the
                # given data folder.
                y_n = input('Would you like to repeat? (y/n):')

            if y_n == 'n':
                pass
        # this is if you want to change where the initial
        # training and validation
        # is - the second and third questions that pop up when the code is ran.
        see_another_set = input(
            'Would you like to see another set of '
            + 'training/validation/testing data? (y/n): ')

#####################
# Component Functions
#####################


def build_model(train, val_set, obj, features):
    """This function takes a train and validation set (train, val_set),
    which are both data frames, builds an SVR model for the
    sensor of interest (obj - a string) using the given
    features (features - a list of strings) and pickles it.
    This returns the validation dataframe with the errors
    and the filename the model was pickled as."""
    val_set = val_set.dropna(subset=features)
    train = train.dropna(subset=features)

    # set the train and val y values - which is the thing
    # we are trying to predict.
    train_y = train[obj]
    val_y = val_set[obj]
    # the train and val _x are the features used to predict
    # the _y
    train_x = train[features]
    val_x = val_set[features]
    # have to normalize the features by l1
    train_x_scaled = preprocessing.normalize(train_x, norm='l1')
    val_x_scaled = preprocessing.normalize(val_x, norm='l1')
    # gather the filname to save the pickled model as, so
    # it can be reloaded and referenced later.
    savepickleas = input(
        'Input the model name to save this as (example.sav): ')
    filenamesaveas = 'svr_model' + savepickleas

    # Change path to save sav files

    os.chdir(os.path.abspath(os.path.join(os.getcwd(), '..')))
    os.chdir(os.getcwd() + '/saved_models')
    # checks to see if the savepickle as file already exists or not
    # and asks if we should overwrite it if it does - or gives the
    # user the option to use a different .sav filename.
    if os.path.isfile(savepickleas):
        print('There is already a model for this!')
        rewrite = input('Would you like to overwrite the file? (y/n): ')
        if rewrite == 'y':
            # this is where the linear SVR model for the
            # sensor (train_y) is being built based off of the
            # features (train_x)
            lin_svr = svm.LinearSVR().fit(train_x, train_y)
            # then we can use that lin_svr to predict the
            # train and val sets based off of the scaled features
            trainpred = lin_svr.predict(train_x_scaled)
            valpred = lin_svr.predict(val_x_scaled)
            filename = filenamesaveas
            # then we pickle the model:
            pickle.dump(lin_svr, open(savepickleas, 'wb'))
        else:
            # this is the same as above - just would be a different
            # filename
            savepickleas_new = input(
                'Input a different name to save this as (example.sav): ')
            filenamesaveas_new = 'svr_model' + savepickleas_new
            lin_svr = svm.LinearSVR().fit(train_x, train_y)
            trainpred = lin_svr.predict(train_x_scaled)
            valpred = lin_svr.predict(val_x_scaled)
            filename = filenamesaveas_new
            pickle.dump(lin_svr, open(savepickleas_new, 'wb'))
        # this could be changed to overwrite the file
    else:
        # this is the same as above - just ran when there
        # is no previous file with the same name.
        lin_svr = svm.LinearSVR().fit(train_x, train_y)
        trainpred = lin_svr.predict(train_x_scaled)
        valpred = lin_svr.predict(val_x_scaled)
        filename = filenamesaveas
        pickle.dump(lin_svr, open(savepickleas, 'wb'))

# Should be reducing the number of things we need to type in.
# If only focusing on continuous real-time training, the
# model will never be reused anyway.

    # Calls the pickled model
    loaded_model = pickle.load(open(savepickleas, 'rb'))
    predict = loaded_model.predict(val_x)
    # predicting the validation set.
    result = loaded_model.score(val_x, val_y)
    # the model score is an R^2 value.
    print('the model score is: ' + str(result))

    df_val = pd.DataFrame(val_y)
    df_val['Predicted'] = predict
    df_val['Error'] = (abs(df_val['Predicted'] - df_val[obj])
                       ) / abs(df_val[obj])
    df_val['Absolute Error'] = abs(df_val['Predicted'] - df_val[obj])
    print('the mean absolute error is: ' +
          str(df_val['Absolute Error'].mean()))

    return df_val, savepickleas


def retest_model(
        savepickleas,
        features,
        df,
        obj,
        test_model_start,
        test_model_end):
    """This function tests the model for the sensor of interests (obj)
    on data that may or may not be calibrated,
    in the date range constrained by test_model_start
    and test_model_end (both strings) in the dataframe loaded (df)
    Use this function to see if the model retains the
    accurate levels when the sensor begins to drift.
    Features is a list of strings of the model features,
    savepickleas is the .sav filename where the model is saved.
    This function returns the df_test dataframe with calculated
    absolute errors."""

    df_test = df[test_model_start: test_model_end]
    # Need to clean out of dataframe sets that have nan values
    # in the features
    df_test = df_test.dropna(subset=features)

    test_y = df_test[obj]
    test_x = df_test[features]

    loaded_model = pickle.load(open(savepickleas, 'rb'))
    # load the pickled model
    predict = loaded_model.predict(test_x)
    # use that loaded model to predict based off of the features
    # in the test set.
    df_test = pd.DataFrame(test_y)
    df_test['Predicted'] = predict
    df_test['Error'] = (
        abs(df_test['Predicted'] - df_test[obj])) / abs(df_test[obj])
    df_test['Absolute Error'] = abs(df_test['Predicted'] - df_test[obj])
    # calculate the absolute error.

    return df_test
