""" =======================  Import dependencies ========================== """
from config import *
from utils import preprocessImageSet
from loaddata import load_pkl
import numpy as np

""" ======================  Read Test data from files ========================== """
knn, standard_scaler_instace, pca = load_pkl(MODEL_FILENAME)
test_data = load_pkl(TEST_DATA_FILENAME)
# test_labels = np.load('labels.npy')
test_data = np.asarray(test_data)

""" ======================  Preprocess the test data ========================== """
preprocessed_test_images = preprocessImageSet(test_data)

""" ======================  Reduce the Dimensionality of the data ========================== """
rescaled_test_images = standard_scaler_instace.transform(preprocessed_test_images)
pca_test_images = pca.transform(rescaled_test_images)

""" ======================  Predict the teset data ========================== """
t_prediction = knn.predict(pca_test_images)
print(t_prediction.shape)

""" ======================  Save predicted test labels in output file ========================== """
np.save(TEST_DATA_OUTPUT_FILENAME, t_prediction)
# knnAccuracy = knn.score(pca_test_images, test_labels)
