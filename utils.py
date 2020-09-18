""" =======================  Import dependencies ========================== """
from config import *
import numpy as np
from skimage.transform import resize

def preprocessImageSet(images):
        images_processed = []
        for i in range (images.shape[0]):
                tdnp = np.asarray(images[i], dtype=np.float32)
                preproceed = tdnp
                # preproceed = np.hstack((tdnp, np.zeros((tdnp.shape[0], (54 - tdnp.shape[1])), dtype = np.float32)))
                # preproceed = np.vstack((preproceed, np.zeros((54 - preproceed.shape[0], preproceed.shape[1]), dtype = np.float32)))
                preproceed = resize(np.asarray(preproceed), (RESIZE_ROW, RESIZE_COLUMN), anti_aliasing=False)
                preproceed = preproceed.flatten()
                images_processed.append(preproceed)
        images_processed = np.asarray(images_processed)
        return images_processed

def getEasyDataSet(images, labels):
        filtered_train_images = []
        filtered_train_labels = []
        for i in range (images.shape[0]):
                if labels[i] == 1 or labels[i] == 2:
                        filtered_train_images.append(images[i])
                        filtered_train_labels.append(labels[i])
        filtered_train_images = np.asarray(filtered_train_images)
        filtered_train_labels = np.asarray(filtered_train_labels)
        return filtered_train_images, filtered_train_labels