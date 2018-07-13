import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

#######################
# Wrapper Function
#######################


def process_data(filepath, obj):
    """This is the wrapper function to load all data,
    clean it, and remove unnecessary timeslices. The filepath
    input is the location of the data, which is hard
    coded into the badsensorfinder.driftfinder() function.
    This function returns a cleaned dataframe."""

    dss_total = load_all_data(filepath)
    df = clean_data(dss_total, obj)
    # the remove_time_slices function can likely be removed from
    # this part of the project. However, it was kept in as example
    # code for how to manually remove time slices if necessary.
    df = remove_time_slices(df, obj)
    df.sort_index(inplace=True)

    return df

########################
# Component Functions
########################


def load_all_data(rootdir):
    """Load all data from a given filepath, returns a DataFrame that
    is all the concatenated files from the given folder."""

    # first we check if the root directory exists.
    if not os.path.exists(rootdir):
        print('The specified rootdir does not exist')
        # this would be a good place to add the option of creating
        # the root directory if it does not exist.
    assert isinstance(rootdir, str), 'Input must be a string'
    # this is ensuring the input to the load_all_data function is
    # the proper format (a string)
    file_list = [f for f in glob.glob(os.path.join(rootdir, '*.csv'))]
    # this function is grabbing all the files that end in .csv and putting
    # those file names in a file_list
    d = {}
    # initiate dictionary to store dataframes in
    for file in file_list:
        # this is iterating over all the files that end in .csv
        name = os.path.split(file)[1].split('.')[0]
        # takes the part of the filename without the ".csv" part
        data = pd.read_csv(file)
        # reading into a pandas dataframe
        new_set = {name: data}
        # updating the dictionary with the set (filename:dataframe)
        d.update(new_set)

    dss_total = pd.DataFrame()
    # initializing an empty dataframe to put all the dataframes
    # in the dictionary in.

    for key in d:
        # print(key)
        dss_total = pd.concat([dss_total, d[key]])
    dss_total = dss_total.reset_index(drop=True)
    # returns the dataframe with all the csv files concatenated.
    return dss_total


def clean_data(dataframe, subset):
    """Cleans a given dataframe by removing the rows containing an
    NA value in the given subset. Subset must be a string of a column
    name. Also sets the timestamp as the index. Inputs are any dataframe
    and the subset. This returns the dataframe with those rows dropped."""

    # drop the rows which have an NA value in the 'subset' column(s)
    df = dataframe.dropna(subset=[subset])
    df.set_index('timestamp', inplace=True)
    # reset the index as the timestamp column
    df.set_index(pd.to_datetime(df.index), inplace=True)
    # read the 'timestamp' column as a date-time and set that as the index.
    return df


def remove_time_slices(df, obj):
    """This function removes data from the initial dataframe that the
    user requests. Takes the dataframe and the sensor of interest (obj)
    for plotting, and returns a dataframe with those slices removed."""

    print('Please review the loaded data, take note if there are '
          + 'date ranges to be excluded.')
    # Plot the obj in the dataframe and have user indicate if there is a time
    # range to exclude:
    fig1 = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
    myplot = plt.scatter(df.index, df[obj], color='red', label=obj)
    plt.ylabel(obj, fontsize=16)
    plt.xlabel('Index', fontsize=16)
    plt.legend(fontsize=16)
    plt.xlim()
    plt.show()

    # set the initial value of more cleaning so the while loop starts
    more_cleaning = 'y'

    # this while loop removes the date ranges specified by the user.
    while more_cleaning == 'y':
        more_cleaning = input(
            'Is there any data that should be excluded from the '
            + 'model training data? (y/n): ')
        if more_cleaning == 'y':
            start_date = input(
                'Input start date of data to exclude from the training set: ')
            end_date = input(
                'Input end date of data to exclude from the training set: ')
            df = pd.concat([df[:start_date], df[end_date:]])
            # this is taking the dataframe of everything before the start date
            # specified (df[:start_date]) and concatenating it with everything
            # after the end date (df[end_date:])
            # replot the data with the date range excluded:
            fig1 = plt.figure(figsize=(20, 5), facecolor='w', edgecolor='k')
            myplot = plt.scatter(df.index, df[obj], color='red', label=obj)
            plt.ylabel(obj, fontsize=16)
            plt.xlabel('Index', fontsize=16)
            # this index is actually date
            plt.legend(fontsize=16)
            plt.xlim()
            plt.show()
            more_cleaning = input('Is there any more data to exclude? (y/n): ')
            # if this is "y", the while loop continues, otherwise the while
            # loop ends.
    return df
