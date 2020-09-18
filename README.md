Project Title: Handwritten Character Recognition.

Steps to run the code:

1) Run the command "python train.py". This will generate the model.py file using the training data. 
2) Run the command "python test.py" to generate the predicted labels on the test data. 
The predicted labels will be stored in the file name specified by the parameter "TEST_DATA_OUTPUT_FILENAME" in the config.py file. Currently, we are using the file "finalLabelsTest.npy" to store the predicted labels on the test data.


We have created a config file called config.py which is checked-in along with this project. The config file "config.py" contains different parameters which we can set before running our code. The config parameter name and their description is given below:

1)
Config Parameter Name and Current Value:                              
TRAIN_DATA_FILENAME = 'train_data.pkl'             		

Config Description:
TRAIN_DATA_FILENAME : Give the train data file name here.This file contains data which our code will use to train our model. Over here we are using a data file called 'train_data.pkl' which will be used by our train.py file to train our model.

2)
Config Parameter Name and Current Value:                              
TRAIN_LABELS_FILENAME = 'finalLabelsTrain.npy'

Config Description:
TRAIN_LABELS_FILENAME : Give train data's labels file name here. This file will contain the actual labels corresponding to our training data. Here we are using finalLabelsTrain.npy file which contains the actual labels corresponding to our training data.

3)
Config Parameter Name and Current Value:                              
TEST_DATA_FILENAME = 'test_data.pkl' 

Config Description:
TEST_DATA_FILENAME : Give the input test file name here. This parameter will contain the name of the file which we want to use to test our model. Currently, we are using test_data.pkl file which contains the test data.

4)
Config Parameter Name and Current Value:                              
TEST_DATA_OUTPUT_FILENAME = 'finalLabelsTest.npy' 

Config Description:
TEST_DATA_OUTPUT_FILENAME : Give the name of the file where you want to store the output which our model generated on the test data. Here we are using finalLabelsTest.npy as the output file. This file will contain the labels which our model predicted for the test data. 

5)
Config Parameter Name and Current Value:                              
MODEL_FILENAME = 'model.py'

Config Description:
MODEL_FILENAME : This config parameter will contain the file name where you want to store the trained model. We created a model using the training data and we are storing the trained model in model.py file currently.


The following parameter's value can be changed in the config.py file during the training of our model to get different results:

1) RESIZE_ROW : resize image to 'RESIZE_ROW' number of rows. This cannot be more than 54.Currently, we are using 28 as the value of this parameter.

2) RESIZE_COLUMN : resize image to 'RESIZE_COLUMN' number of columns. This cannot be more than 54. Currently, we are using 28 as the value of this parameter.

3) PCA_VARIANCE : PCA variance value for pre-processing i.e. for feature reduction. The current value being used by us for this parameter is 0.60 .

4) KNN_NEIGHBORS : K of KNN classifier. We are using a value of k=1 for this parameter.

5) KNN_DISTANCE_METRIC_VALUE : This is the distance metric which the KNN will use. Currently, we have given this value as 'minkowski'. The values can be changed to 'minkowski', 'euclidean' or 'hamming'.

6) BONUS_MODE : A value "True" (This is case-sensitive) for this parameter will make our model classify all the characters in the test data that is ’a’, ’b’,’c’,’d’,’h’, ’i’, ’j’ and ’k’ . We should set this value as "True" to compete for the bonus points for our project. A value of "False" (This is case-sensitive) will make our model only classify the characters 'a' and 'b' in the test data. The default value given by us is currently False.

7) DO_EXPERIMENTS : Make the value of this parameter as "True" if you want to see experiments done by us for getting optimum values of different hyper parameters for getting our results. Currently the value is "False".

After we have set the config values in config.py file for the different parameters mentioned above, we can run the code using the steps mentioned in the beginning of this document.

Note: We  have created one images folder named "experiment-results-images" which is also checked-in here. This contains all the images for results for all our experiments carried out for this project.
