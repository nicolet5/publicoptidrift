Use Case:
Eventually, the company would like to take our code and apply it to a new plant. 
1. At a new plant, company will collect "in calibration" data for a set amount of time (dictated by our exploration of how much is enough)
2. Once enough "ground truth" data is collected, for every sensor in the plant the program will apply Lasso/decision trees to determine which are the important parameters 
3. Once important parameters for each sensor are established (introduce a user confirmation that those parameters make sense?) the program will use those parameters to train whichever model is chosen. 
4. Acceptable variance in each sensor will be inputted manually by the user.
5. Model will be used to predict what the sensor value should be, and compare it to what the sensor is actually reading in real time (once every hour?)
6. If the program deems a sensor "out of calibration", then there will be some user input to confirm it actually is, and then that slice of data will be flagged to be excluded from the next training of the model. 
7.  Model will be retrained every (three?) months. Will feature selection be reperformed every three months too? Depends on computational time. 

Notes:
* Want code that can be implemented by their software engineers to predict sensor drift in chiller plants
* Developing code around one specific chiller plant, but want to be able to translate that to other chiller plants
* Based off of data that is considered accurate (no sensor drift): select the best type of model to use (Regression, SVM, Decision Trees, Random Forests, Neural Net), train that model on the data considered to be within calibration to predict what the sensor should be reading, based off of that output give a warning if sensor's actual readings are out of a certain range (numerical, set range, not percentage as indicated by OE)
* Be able to edit the acceptable variation of each sensor (model prediction - actual reading), as this will vary plant to plant and sensor to sensor
* Indicate how much initial, in-calibration data is necessary to build an acceptable model.
