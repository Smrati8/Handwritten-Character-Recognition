""" =======================  Config Variables ========================== """

TRAIN_DATA_FILENAME = 'train_data.pkl' 			# Give train data file name here.
TRAIN_LABELS_FILENAME = 'finalLabelsTrain.npy' 	# Give train data's labels file name here.
TEST_DATA_FILENAME = 'EasyData.pkl' 			# Give input test file name
TEST_DATA_OUTPUT_FILENAME = 'finalLabelsTest.npy' 		# Output file name.
MODEL_FILENAME = 'model.py'

RESIZE_ROW = 28 		# resize image to 'RESIZE_ROW' number of rows. Can not be more than 54.
RESIZE_COLUMN = 28		# resize image to 'RESIZE_COLUMN' number of columns. Can not be more than 54.
PCA_VARIANCE = 0.60		# PCA variance for pre processing i.e for feature reduction.
KNN_NEIGHBORS = 1		# K of KNN classifier.

KNN_DISTANCE_METRIC_VALUE = 'minkowski' #Values can be 'minkowski', 'euclidean', 'manhattan', 'hamming'.

BONUS_MODE = True 		# Make this as true to test for all the characters.

DO_EXPERIMENTS = False 	# Turn on this flag, If you want to see experiments done for getting optimum values of hyper parameters.