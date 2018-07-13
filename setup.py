import os
os.chdir(os.getcwd() + '/optidrift')
import model
import getdata
obj = input(
    'Please input the sensor to build a model for' +
    '(must match column name exactly): ')
filepath = '../Data/plant1/h_data/'
df_processed = getdata.process_data(filepath, obj)
model.model_exploration(df_processed, obj)
