""" =======================  Import dependencies ========================== """
from config import *
from utils import preprocessImageSet, getEasyDataSet
from loaddata import load_pkl, save_pkl

import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as standard_scaler
from sklearn.decomposition import PCA
import pandas as pd

""" ======================  Read training data from files ========================== """
train_data = load_pkl(TRAIN_DATA_FILENAME)
train_labels = np.load(TRAIN_LABELS_FILENAME)

""" ======================  Preprocess the training data ========================== """
preprocessed_train_images= preprocessImageSet(train_data)

""" ======================  Filter the data if BONUS_MODE is false ========================== """
if BONUS_MODE == False:
    filtered_train_images, filtered_train_labels = getEasyDataSet(preprocessed_train_images, train_labels)
else:
	filtered_train_images = preprocessed_train_images
	filtered_train_labels = train_labels

""" ======================  Reduce the Dimensionality of the data ========================== """

# Rescale the feature vectors by subtracting the mean and dividing the standard deviation of the dataset.
standard_scaler_instace = standard_scaler()
standard_scaler_instace.fit(filtered_train_images)

rescaled_train_images = standard_scaler_instace.transform(filtered_train_images)

# Apply PCA
pca = PCA(PCA_VARIANCE)
pca.fit(rescaled_train_images)
pca_train_images = pca.transform(rescaled_train_images)

""" ======================  Fit the Model ========================== """
knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
knn.fit(pca_train_images, filtered_train_labels)

""" ======================  Save the Model ========================== """
save_pkl(MODEL_FILENAME, [knn, standard_scaler_instace, pca])
# knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='KNN_DISTANCE_METRIC_VALUE')
# knn_no_pca.fit(train_imges_rescaled, train_labels)

print("\nTraining is done. You can now test the model.")

""" ======================  Experiments ========================== """

if DO_EXPERIMENTS == True:
    
    def getExperimentResultsForDataSet(images, labels):
        train_images, test_images, train_labels, test_labels = train_test_split(images, 
                                                                    labels, 
                                                                    test_size=0.25, random_state=0)
        
        # Fit on training set only.
        standard_scaler_instace = standard_scaler()
        standard_scaler_instace.fit(train_images)
        # Apply transform to both the training set and the test set.
        train_imges_rescaled = standard_scaler_instace.transform(train_images)
        test_imges_rescaled = standard_scaler_instace.transform(test_images)
    
        # Make an instance of the Model
        pca_95 = PCA(.95)
        pca_95.fit(train_imges_rescaled)
        
        pca_90 = PCA(.90)
        pca_90.fit(train_imges_rescaled)
        
        pca_80 = PCA(.80)
        pca_80.fit(train_imges_rescaled)
        
        pca_70 = PCA(.70)
        pca_70.fit(train_imges_rescaled)
        
        pca_60 = PCA(.60)
        pca_60.fit(train_imges_rescaled)
        
        pca_50 = PCA(.50)
        pca_50.fit(train_imges_rescaled)
    
        train_imges_pca_95 = pca_95.transform(train_imges_rescaled)
        test_imges_pca_95 = pca_95.transform(test_imges_rescaled)
        
        train_imges_pca_90 = pca_90.transform(train_imges_rescaled)
        test_imges_pca_90 = pca_90.transform(test_imges_rescaled)
        
        
        train_imges_pca_80 = pca_80.transform(train_imges_rescaled)
        test_imges_pca_80 = pca_80.transform(test_imges_rescaled)
        
        
        train_imges_pca_70 = pca_70.transform(train_imges_rescaled)
        test_imges_pca_70 = pca_70.transform(test_imges_rescaled)
        
        
        train_imges_pca_60 = pca_60.transform(train_imges_rescaled)
        test_imges_pca_60 = pca_60.transform(test_imges_rescaled)
        
        train_imges_pca_50 = pca_50.transform(train_imges_rescaled)
        test_imges_pca_50 = pca_50.transform(test_imges_rescaled)
        
        pca_innverted_images_95 = pca_95.inverse_transform(train_imges_pca_95)
        pca_innverted_images_90 = pca_90.inverse_transform(train_imges_pca_90)
        pca_innverted_images_80 = pca_80.inverse_transform(train_imges_pca_80)
        pca_innverted_images_70 = pca_70.inverse_transform(train_imges_pca_70)
        pca_innverted_images_60 = pca_60.inverse_transform(train_imges_pca_60)
        pca_innverted_images_50 = pca_50.inverse_transform(train_imges_pca_50)
        
        # Plots to demonstrate PCA - START
        plt.figure(figsize=(16,4));
        
        # Original Image
        plt.subplot(1, 7, 1);
        plt.imshow(train_images[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(RESIZE_ROW*RESIZE_COLUMN) + ' components';
        plt.xlabel(label, fontsize = 14)
        plt.title('Original Image', fontsize = 14);
        
    
        plt.subplot(1, 7, 2);
        plt.imshow(pca_innverted_images_95[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(pca_95.n_components_) + ' components';
        plt.xlabel(label, fontsize = 14)
        plt.title('95% of Variance', fontsize = 14);
        
    
        plt.subplot(1, 7, 3);
        plt.imshow(pca_innverted_images_90[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(pca_90.n_components_) + ' components';
        plt.xlabel('236 components', fontsize = 14)
        plt.title('90% of Variance', fontsize = 14);
        
    
        plt.subplot(1, 7, 4);
        plt.imshow(pca_innverted_images_80[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(pca_80.n_components_) + ' components';
        plt.xlabel(label, fontsize = 14)
        plt.title('80% of Variance', fontsize = 14);
        
        plt.subplot(1, 7, 5);
        plt.imshow(pca_innverted_images_70[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(pca_70.n_components_) + ' components';
        plt.xlabel(label, fontsize = 14)
        plt.title('70% of Variance', fontsize = 14);
        
        plt.subplot(1, 7, 6);
        plt.imshow(pca_innverted_images_60[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(pca_60.n_components_) + ' components';
        plt.xlabel(label, fontsize = 14)
        plt.title('60% of Variance', fontsize = 14);
        
        plt.subplot(1, 7, 7);
        plt.imshow(pca_innverted_images_50[1].reshape(RESIZE_ROW, RESIZE_COLUMN),
                      cmap = plt.cm.gray, interpolation='nearest');
        label = str(pca_50.n_components_) + ' components';
        plt.xlabel(label, fontsize = 14)
        plt.title('50% of Variance', fontsize = 14);
        # Plots to demosntrate PCA - END
        plt.show()
        
        knn_no_pca = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca_95 = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca_90 = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca_80 = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca_70 = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca_60 = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca_50 = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        
        knn_no_pca.fit(train_imges_rescaled, train_labels)
        knn_pca_95.fit(train_imges_pca_95, train_labels)
        knn_pca_90.fit(train_imges_pca_90, train_labels)
        knn_pca_80.fit(train_imges_pca_80, train_labels)
        knn_pca_70.fit(train_imges_pca_70, train_labels)
        knn_pca_60.fit(train_imges_pca_60, train_labels)
        knn_pca_50.fit(train_imges_pca_50, train_labels)
        
        knn_no_pca = knn_no_pca.score(test_imges_rescaled, test_labels)
        knn_pca_95 = knn_pca_95.score(test_imges_pca_95, test_labels)
        knn_pca_90 = knn_pca_90.score(test_imges_pca_90, test_labels)
        knn_pca_80 = knn_pca_80.score(test_imges_pca_80, test_labels)
        knn_pca_70 = knn_pca_70.score(test_imges_pca_70, test_labels)
        knn_pca_60 = knn_pca_60.score(test_imges_pca_60, test_labels)
        knn_pca_50 = knn_pca_50.score(test_imges_pca_50, test_labels)
        
        df_knn = pd.DataFrame({"Variance %" : [100, 95, 90, 80, 70, 60, 50],
                           "Feature Vector Dimension" : [RESIZE_ROW*RESIZE_COLUMN, pca_95.n_components_, pca_90.n_components_, pca_80.n_components_, pca_70.n_components_, pca_60.n_components_, pca_50.n_components_],
                           "Performance" : [knn_no_pca, knn_pca_95, knn_pca_90, knn_pca_80, knn_pca_70, knn_pca_60, knn_pca_50]})
    
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            display(df_knn)
    
    def getExperimentResults():
        train_data = load_pkl(TRAIN_DATA_FILENAME)
        train_labels = np.load(TRAIN_LABELS_FILENAME)
        preprocessed_train_images = preprocessImageSet(train_data)
        
        print("\nExperiment results for Easy Dataset i.e. (a and b):")
        easy_dataset_images, easy_dataset_labels = getEasyDataSet(preprocessed_train_images, train_labels)
        getExperimentResultsForDataSet(easy_dataset_images, easy_dataset_labels)
        
        print("\nExperiment results for Hard Dataset i.e. (a, b, c, d, h, i, j, and k):")
        easy_dataset_images, easy_dataset_labels = getEasyDataSet(preprocessed_train_images, train_labels)
        getExperimentResultsForDataSet(preprocessed_train_images, train_labels)
         
    def annot_max(x,y, ax=None):
        x = np.asarray(x)
        y = np.asarray(y)
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text= "k={:.3f}, Accuracy={:.3f}".format(xmax, ymax)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->")
        kw = dict(xycoords='data',textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
        
    def findOptimumK(images, lables):
        krange = np.arange(1,100,2)
        
        accuracy_pca_95 = []
        accuracy_pca_90 = []
        accuracy_pca_80 = []
        accuracy_pca_70 = []
        accuracy_pca_60 = []
        accuracy_pca_50 = []
        for k in krange:
            accuracy_pca_95.append(getExperimentAccuracy(images, lables, 0.95, k))
            accuracy_pca_90.append(getExperimentAccuracy(images, lables, 0.90, k))
            accuracy_pca_80.append(getExperimentAccuracy(images, lables, 0.80, k))
            accuracy_pca_70.append(getExperimentAccuracy(images, lables, 0.70, k))
            accuracy_pca_60.append(getExperimentAccuracy(images, lables, 0.60, k))
            accuracy_pca_50.append(getExperimentAccuracy(images, lables, 0.50, k))
        
        plt.figure(4, figsize=(16,16))
        plt.subplot(4, 2, 1)
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of KNN')
        plt.plot(krange, accuracy_pca_95)
        annot_max(krange,accuracy_pca_95, plt)
        plt.title('Accuracy vs K of KNN (PCA - 0.95)', fontsize = 14);
        
        plt.subplot(4, 2, 2);
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of KNN')
        plt.plot(krange, accuracy_pca_90)
        annot_max(krange,accuracy_pca_90, plt)
        plt.title('Accuracy vs K of KNN (PCA - 0.90)', fontsize = 14)
        plt.show()
        
        plt.figure(5, figsize=(16,16))
        plt.subplot(5, 2, 1);
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of KNN')
        plt.plot(krange, accuracy_pca_80)
        annot_max(krange,accuracy_pca_80, plt)
        plt.title('Accuracy vs K of KNN (PCA - 0.80)', fontsize = 14);
        
        plt.subplot(5, 2, 2);
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of KNN')
        plt.plot(krange, accuracy_pca_70)
        annot_max(krange,accuracy_pca_70, plt)
        plt.title('Accuracy vs K of KNN (PCA - 0.70)', fontsize = 14);
        plt.show()
        
        plt.figure(6, figsize=(16,16))
        plt.subplot(6, 2, 1);
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of KNN')
        plt.plot(krange, accuracy_pca_60)
        annot_max(krange,accuracy_pca_60, plt)
        plt.title('Accuracy vs K of KNN (PCA - 0.60)', fontsize = 14);
        
        plt.subplot(6, 2, 2);
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of KNN')
        plt.plot(krange, accuracy_pca_50)
        annot_max(krange,accuracy_pca_50, plt)
        plt.title('Accuracy vs K of KNN (PCA - 0.50)', fontsize = 14);
        plt.show()
            
    def getExperimentAccuracy(images, labels, pca_variance, k):
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=0)
        # Fit on training set only.
        standard_scaler_instace = standard_scaler()
        standard_scaler_instace.fit(train_images)
        # Apply transform to both the training set and the test set.
        train_imges_rescaled = standard_scaler_instace.transform(train_images)
        test_imges_rescaled = standard_scaler_instace.transform(test_images)
        
        
        pca = PCA(pca_variance)
        pca.fit(train_imges_rescaled)
        
        train_imges_pca = pca.transform(train_imges_rescaled)
        test_imges_pca = pca.transform(test_imges_rescaled)
        
        knn_pca = KNeighborsClassifier(n_neighbors=k)
        knn_pca.fit(train_imges_pca, train_labels)
        
        return knn_pca.score(test_imges_pca, test_labels)
      
    def showPreprocessedImages():
        plt.figure(figsize=(16,4))    
        plt.subplot(1, 3, 1);
        plt.imshow(preprocessed_train_images[1].reshape(RESIZE_ROW, RESIZE_COLUMN), cmap = plt.cm.gray, interpolation='nearest')
        plt.subplot(1, 3, 2);
        plt.imshow(preprocessed_train_images[10].reshape(RESIZE_ROW, RESIZE_COLUMN), cmap = plt.cm.gray, interpolation='nearest')
        plt.subplot(1, 3, 3);
        plt.imshow(preprocessed_train_images[20].reshape(RESIZE_ROW, RESIZE_COLUMN), cmap = plt.cm.gray, interpolation='nearest')
        plt.show()
        
    def findOptimumKForEasyDataSet():
        train_data = load_pkl(TRAIN_DATA_FILENAME)
        train_labels = np.load(TRAIN_LABELS_FILENAME)
        preprocessed_train_images = preprocessImageSet(train_data)
        easy_dataset_images, easy_dataset_labels = getEasyDataSet(preprocessed_train_images, train_labels)
        print("\nEasy Dataset:\n")
        findOptimumK(easy_dataset_images, easy_dataset_labels)
        
    def findOptimumKForHardDataSet():
        print("\nHard Dataset:")
        train_data = load_pkl(TRAIN_DATA_FILENAME)
        train_labels = np.load(TRAIN_LABELS_FILENAME)
        preprocessed_train_images = preprocessImageSet(train_data)
        findOptimumK(preprocessed_train_images, train_labels)
        
    def applySimpleKnn(images, labels, k):
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=0)
        knn_simple = KNeighborsClassifier(k)
        knn_simple.fit(train_images, train_labels)
        return knn_simple.score(test_images, test_labels)
        
    def getKOptimumForSimpleKnn():
        train_data = load_pkl(TRAIN_DATA_FILENAME)
        train_labels = np.load(TRAIN_LABELS_FILENAME)
        preprocessed_train_images = preprocessImageSet(train_data)
        easy_dataset_images, easy_dataset_labels = getEasyDataSet(preprocessed_train_images, train_labels)
        
        krange = np.arange(1,100,2)
        
        accuracyForEasyDataSet = []
        accuracyForHardDataSet = []
        for k in krange:
            accuracyForEasyDataSet.append(applySimpleKnn(easy_dataset_images, easy_dataset_labels,k))
            accuracyForHardDataSet.append(applySimpleKnn(preprocessed_train_images, train_labels, k))
        
        plt.figure(5)
        plt.title("Accuracy vs K of Simple KNN")
        
        # Training using kFold - 7d
        p1 = plt.plot(krange, accuracyForEasyDataSet)
        # On Test Data - 7d
        p2 = plt.plot(krange, accuracyForHardDataSet, 'g')
        
        plt.ylabel('Accuracy') # label x and y axes
        plt.xlabel('K of Simple KNN')
        plt.legend((p1[0],p2[0]),['Easy Dataset', 'Hard Dataset'], loc=1)
        plt.show()
                    
    def plotDistanceMetricVariation():
        train_data = load_pkl(TRAIN_DATA_FILENAME)
        train_labels = np.load(TRAIN_LABELS_FILENAME)
        preprocessed_train_images = preprocessImageSet(train_data)
        easy_dataset_images, easy_dataset_labels = getEasyDataSet(preprocessed_train_images, train_labels)
        
        easy_minkowski_accuracy, easy_euclidean_accuracy, easy_hamming_accuracy, easy_manhattan_accuracy = getAccuracyAccordingToDifferentDistanceMetrics(easy_dataset_images, easy_dataset_labels)
        hard_minkowski_accuracy, hard_euclidean_accuracy, hard_hamming_accuracy, hard_manhattan_accuracy = getAccuracyAccordingToDifferentDistanceMetrics(preprocessed_train_images, train_labels)
        
        print('\n')
        df_knn_distance = pd.DataFrame({"Distace Metric" : ['minkowski', 'euclidean', 'hamming', 'manhattan'],
                           "For Easy Dataset" : [easy_minkowski_accuracy, easy_euclidean_accuracy, easy_hamming_accuracy, easy_manhattan_accuracy],
                           "For Hard Dataset" : [hard_minkowski_accuracy, hard_euclidean_accuracy, hard_hamming_accuracy, hard_manhattan_accuracy]})
    
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            display(df_knn_distance)
        
                
    def getAccuracyAccordingToDifferentDistanceMetrics(images, labels):
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=0)
        # Fit on training set only.
        standard_scaler_instace = standard_scaler()
        standard_scaler_instace.fit(train_images)
        # Apply transform to both the training set and the test set.
        train_imges_rescaled = standard_scaler_instace.transform(train_images)
        test_imges_rescaled = standard_scaler_instace.transform(test_images)
        
        
        pca = PCA(PCA_VARIANCE)
        pca.fit(train_imges_rescaled)
        
        train_imges_pca = pca.transform(train_imges_rescaled)
        test_imges_pca = pca.transform(test_imges_rescaled)
        
        knn_pca_minkowski = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='minkowski')
        knn_pca_minkowski.fit(train_imges_pca, train_labels)
        
        knn_pca_euclidean = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='euclidean')
        knn_pca_euclidean.fit(train_imges_pca, train_labels)
        
        knn_pca_hamming = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='hamming')
        knn_pca_hamming.fit(train_imges_pca, train_labels)
        
        knn_pca_manhattan = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric='manhattan')
        knn_pca_manhattan.fit(train_imges_pca, train_labels)
        
        
        
        return knn_pca_minkowski.score(test_imges_pca, test_labels), knn_pca_euclidean.score(test_imges_pca, test_labels), knn_pca_hamming.score(test_imges_pca, test_labels), knn_pca_manhattan.score(test_imges_pca, test_labels)
        
    def printConfusionMatrix(images, labels):
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=0)
        # Fit on training set only.
        standard_scaler_instace = standard_scaler()
        standard_scaler_instace.fit(train_images)
        # Apply transform to both the training set and the test set.
        train_imges_rescaled = standard_scaler_instace.transform(train_images)
        test_imges_rescaled = standard_scaler_instace.transform(test_images)
        
        pca = PCA(PCA_VARIANCE)
        pca.fit(train_imges_rescaled)
        
        train_imges_pca = pca.transform(train_imges_rescaled)
        test_imges_pca = pca.transform(test_imges_rescaled)
        
        knn_pca = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric=KNN_DISTANCE_METRIC_VALUE)
        knn_pca.fit(train_imges_pca, train_labels)
        
        t_pred = knn_pca.predict(test_imges_pca)
        print(confusion_matrix(test_labels, t_pred))
        
        
    def generateConfusionMatrix():
        train_data = load_pkl(TRAIN_DATA_FILENAME)
        train_labels = np.load(TRAIN_LABELS_FILENAME)
        preprocessed_train_images = preprocessImageSet(train_data)
        easy_dataset_images, easy_dataset_labels = getEasyDataSet(preprocessed_train_images, train_labels)
        
        print("\nConfustion matrix for Easy Dataset\n")
        printConfusionMatrix(easy_dataset_images, easy_dataset_labels)
        print("\nConfustion matrix for Hard Dataset\n")
        printConfusionMatrix(preprocessed_train_images, train_labels)
        
    # showPreprocessedImages()
    # getExperimentResults()
    # findOptimumKForEasyDataSet()
    # findOptimumKForHardDataSet()
    # getKOptimumForSimpleKnn()
    # plotDistanceMetricVariation()
    # generateConfusionMatrix()
        
    