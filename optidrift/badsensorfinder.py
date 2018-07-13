# first we import the three base modules, model.py,
# getdata.py and lassofeatsel.py
import model
import getdata
import lassofeatsel


def drift_finder():
    """This is the wrapper function for all of the bad sensor finder
    functions. There are no inputs, but the user will be prompted
    for inputs as the program runs. This returns nothing, but figures
    will appear as the program runs."""
    obj = input(
        'Please input the sensor to build a model for '
        + '(must match column name exactly): ')
    filepath = '../Data/plant1/h_data/'
    # this is the filepath that contains all the data to be used.
    # the program finds all .csv files in the specified folder
    # and combines them into one dataframe.
    df_processed = getdata.process_data(filepath, obj)
    model.model_exploration(df_processed, obj)
    return()

# drift_finder()
# uncomment the above line if you want to run through the git terminal
