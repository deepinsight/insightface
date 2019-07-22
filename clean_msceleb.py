import os
import random
import numpy as np
import matplotlib.pyplot as plt
from gap_statistic import optimalK
from keras import models
from PIL import Image
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans

'''
Calculates the intra-class centroid, then delete faces that are X SD away from the centroid

Point to a root directory where the structure is like this:
Person A:
 - image_1.jpg
 - image_2.jpg
 - image_3.jpg
Person B:
 - image_4.jpg
 - image_5.jpg
 - image_6.jpg
Person C:
 - image_7.jpg
 - image_8.jpg
 - image_9.jpg
 - image_10.jpg
...
Person N:
 - image_100.jpg
 - image_101.jpg

THIS WILL REMOVE SOME DATA, SO MAKE SURE YOU HAVE A COPY OF YOUR ORIGINAL DATASET
'''


ROOT_IMAGE_DIR = './data/msceleb_retina_crop'
model = models.load_model('model_data/facenet_keras.h5')
identities = os.listdir(ROOT_IMAGE_DIR)
if '.DS_Store' in identities:
    identities.remove('.DS_Store')

identities2label = {v:k for k, v in enumerate(identities)}

for person_name in identities:
    print('Processing:', person_name)
    subdir = os.path.join(ROOT_IMAGE_DIR, person_name)
    filenames = os.listdir(subdir)
    img_lst = []
    for filename in filenames:
        image_path = os.path.join(subdir, filename)
        img = Image.open(image_path)
        img = img.resize((160,160))
        img = np.asarray(img)
        if img.shape[-1] != 3:
            continue
        img = img[:,:,0:3] / 255.
        img_lst.append(img)
    arr = np.stack(img_lst)
    pred = model.predict(arr)

    # count number of clusters with gap statistics
    opt = optimalK.OptimalK()
    n_clusters = opt(pred, cluster_array=np.arange(1, 9))
    kmeans = KMeans(n_clusters=n_clusters).fit(pred)
    biggest_cluster_ind = np.bincount(kmeans.labels_).argmax()
    ind_not_identity = np.where(kmeans.labels_ != biggest_cluster_ind)

    for i in ind_not_identity[0]:
        fname_to_delete = os.path.join(subdir, filenames[i])
        os.remove(fname_to_delete)






