# Optidrift
Optidrift is a tool developed for Optimum Energy LLC. The purpose of this tool is to be able to detect sensor drift or failure based off surrounding measurements. The tool will take data during the chiller plant's operation and perform a feature set reduction on a given sensor. Using this reduced feature set, a training dataset, presumably from a time period where all sensors are in calibration, will be used to train a Support Vector Machine Regression model. If the predictions from the model drastically differ from the actual data that is collected, it might be indicative that the sensor may be out of calibration.

### SOFTWARE DEPENDENCIES
* Python Version 3.6
* Packages used: datetime, glob, math, matplotlib, os, pickle, numpy, pandas, scikit-learn, scipy

### PACKAGES INCLUDED IN THE TOOL 

### FOLDERS IN THE REPOSITORY
* **Optidrift**
Contains the .py files  
	* getdata - Loads all csv data of the Optimum Energy LLC format, clean it, and remove necessary timeslices.
	* model - Contains the code that uses a certain objective or sensor to train the support vector regression model, based off specific training and testing months. Also contains the code for the user-interface.
	* lasso_featset - Uses certain objective or sensor to perform LASSO to isolate the important features of that objective.
	* badsensorfinder - Wrapper function to utilize all .py files above.
	* test - For unit testing. 
* **jupyter_notebooks**	
Contains all the software development in jupyter notebooks.
* **data**
Contains all the data for an anonymous chiller plant.
* **docs** 
Contains files with information for the use cases, tech reviews and images used in the documentation. 
	* 
* **saved_models**
Contains all the pickled trained models that have been tested.

### SETUP
1. Use `git clone` to copy the repository onto your machine. 
2. Import the packages 

### HOW TO RUN 
Note: the data is being pulled from the folder /Data/plant1/h_data, this can be changed in the main .py file badsensorfinder. 

To run the program in a jupyter notebook: 
```
import badsensorfinder
```
```
badsensorfinder.drift_finder()
```

To run the program in the terminal, open up the badsensorfinder.py file into a text editor and remove the # from the following line
```
# drift_finder()
```
You can then type in the following line of code in terminal to run it:
```
python badsensorfinder.py
```

When you run drift_finder(), the following questions will appear as it runs:

* _**"Please input the sensor to build a model for (must match column name exactly):"**_
  - This can be any sensor that appears in the file, but it must match the column name exactly (case-sensitive). There can be no leading or trailing spaces. 
  - example input: CH1CDWFLO
  - After this is entered in, a plot of the sensor's value over all the data located in the /Data/h_data folder will be plotted. This gives you an opportunity to ensure whatever date ranges you choose as training and validation data seem to be in calibration.   


* _**"Please review the loaded data, take note if there are date ranges to be excluded. Is there any data to be excluded from the model training data? (y/n):"**_
  - This is largely an artifact of utilizing the entire date range in the file as training data. However, the program now asks for input from the user for training, validating, and testing sets. So, this question can be ignored (just say n), but was left in the program as future reference. 
  - example input: n     


* _**"Input the start date of training data", "Input the end date of training data:"**_
  - These are the dates that will be used as training data for the model. The time from between these dates should be considered "good" data, or data where the sensor is most likely in calibration. 
  - Format of this input is yyyy-mm-dd
  - example input (start): 2016-08-01
  - example input (end): 2016-08-31 
 

* _**"Input the start date of validation data", "Input the end date of validation data:"**_ 
  - These are the dates that will be used as validation data for the model. This data set is used to make sure the model that is built off of the training data actually builds an acceptable model. Thus, the time between these two dates should aslo be considered "good" data (in calibration).  
  - Format of this input is yyyy-mm-dd
  - example input (start): 2016-09-01
  - example input (end): 2016-09-30  
  - after this is input, LASSO will be performed to do feature set reduction. An alpha of 0.1 is programmed in as a starting point. Then the error at that alpha will be reported (this error is from the LASSO model, and is from a test-train split of 20/80 in the training data set. 
  - Two plots of the coefficients vs alpha and error vs alpha will appear. The plots' y-axis is labeled as lambda, but it is the same variable as alpha. 
  - Then, the features at that selected alpha are reported.


* _**"Would you like to attempt another alpha? (y/n): "**_
  - If answered 'y', the program will ask you to input a new alpha, will re-perform LASSO, and report the error and the coefficients again. Then the program prompts if you would liek to attempt another alpha - until the answer is n. 
  - Once this is answered 'n', then the program will report the features that LASSO selected. 
  
  
* _**"Would you like to change the features? (y/n): "**_ 
  - This gives the user the opportunity to change whichever features LASSO selected. This is useful if the features LASSO selected either don't make sense (i.e. Projected Annual Dollar Savings) or if the user knows one of the features is out of calibration, and shouldn't be used as a part of the model. 
  - If answered 'y', then the user will be prompted: "Would you like to add or remove features? (add/rm): ". The answer must be exactly "add" or "rm". Then it will ask which feature you would like to add or remove, at which point the feature must be entered exactly as it appears in the column header of the data file. 
  - Once answered 'n', then the features will be fed as-is into the model, and a final feature set list will be reported.  


* _**"Input the model name to save this as: "**_ 
  - Once the model is built, it will be "pickled" (saved) as this file in the "saved_models" directory. 
  - Format of this input is filename.sav
  - example input: CH1CDWFLOmodel.sav
  - Once this is input, a model score and mean absolute error (|predicted - actual|) will be reported. These are based off of the validation set data vs. what the model predicted.  
  
  
* _**"Input the start date of test data: ", "Input the end date of test data: "**_
  - This is the data that is selected to simulate the real-time use of the model. This can be either in calibration or out of calibration data, or some mixture of. 
  - example input (start): 2016-10-01
  - example input (end): 2018-03-30
  
  
* _**"Please input the allowable error in this sensor (|Predicted - Actual|): "**_
  - This is essentially asking for how much the sensor can drift before it is considered a problem. This is sensor dependent and will have varying units, but only requires an integer as input. 
  - example input: 250
  - After this is input, two plots should appear. The first is a plot of the sensor's value over time, including the train, validation, and test set. 
  
  
* _**"Would you like to test the model on the month subsequent to the validation data? If that data is not available in the folder, answer "n" (y/n): "**_
  - This is asking if there is data loaded into the folder to be used as testing data. There must be at least 30 days of data following the end date of the validation data in order to answer "y". 
  - If the answer is "n", then the program ends, but a trained model is saved. 


* _**"Would you like to remove the out-of-calibration data from the training set, re-train, and predict the following month? (y/n): "**_
  - If answered "y", this will take the previous set of training data, add on the previous testing data set, and use that as the new training set. If any data points were considered out of calibration based off of the user-input threshold, those data points will be excluded from the new training set. However, the program asks the user if they would like to add back in specified data points that were removed. 
  - The validation set is always the set input at the beginning of the program, so that set will always be excluded from the training data. 
  - The program is set to test on the 30 days following either the validation set or the training set (whichever is latest). 
  - Then the program loops back through gathering the appropriate features and building a model. It then will ask if you would like to repeat this process. 
  - If asnwered "n", the program ends. 
  

* _**"Would you like to see another set of training/validation/testing data? (y/n): "**_
  - If y is input, this starts the program over, using the same sensor to build a model for. 
  - If n is input, this ends the program. 

### RUNNING NOSETESTS TO CHECK THE UNITS OF THE CODE
```
$ nosetests
...........
----------------------------------------------------------------------
Ran 11 tests in 16.370s

OK

```

### ACKNOWLEDGEMENTS

We would like to thank Professor David Beck, Professor Jim Pfaendtner and the group of teaching assistants in the DIRECT program.

We would also like to thank the team at Optimum Energy for this opportunity. We would especially like to thank Fred Woo, Dana Lindquist, Michael Huguenard, and Tim Wehage for all their expertise, support, and guidance throughout this project.


### Notes for Improvement

1. Choosing the appropriate features 

	Currently, LASSO is performed, then those selected features are fed into the SVM model builder. A graph of the mean absolute error of the SVM model (based on the validation data set) is shown to allow the user to choose an appropriate alpha (corresponding to the appropriate features to use to build the SVM model). This was done because the error produced by LASSO does not always reflect the error that is given by SVM - thus choosing the appropriate features was a "shot-in-the-dark". 

	However, the method of incorporating the SVM error as feedback is slow, and produces graphs that may be difficult to interpret at a quick glance. For each alpha, an SVM model is created and the mean absolute error in the validation set is calculated. Since only one SVM model result is being graphed, the error looks very noisy. In an attempt to fix this, multiple SVM models were built and evaluated at each alpha, and the mean absolute error graphed is the average of those multiple SVM models. Unsurprisingly, this is a very slow process and could be optimized. 

2. How data is cleaned 

	Currently, the data cleaning is minimal. First, data points are cleaned out at which the sensor of interest has an NaN value (in the getdata module, clean_data function). Then the model is built based on the training set, which only allows features to be used that have no NaN values in that time range (in the lassofeatsel module, FindFeatures function). Then, if any data point has an NaN value for any of the features used in the model, that data point is dropped (since the model relies on having values for all the features to make a prediction). This could likely be made more robust in someway, allowing one feature to have an NaN value and still being able to make a prediction (interpolation to replace the NaN?)

3. Incorporating a "backup" for when a feature used in a model goes out of calibration itself 
	
	For example, if CH1CDWFLO is used as a feature for CH2CDWFLO and the CH1CDWFLO sensor goes out of calibration, the model for CH2CDWFLO will not be accurate since not all of the features are accurate. 

