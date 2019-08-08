# Copyright (c) 2019 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import h5py
import numpy as np

np.random.seed(seed=0)


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    valset_ind = np.random.choice(range(60000), size=5000, replace=False)
    trainset_ind = np.array([i for i in range(60000) if i not in valset_ind])
    
    train_set_images = x_train[trainset_ind]
    train_set_images = train_set_images.reshape((55000,28,28,1))
    train_set_labels = y_train[trainset_ind]
    
    val_set_images = x_train[valset_ind]
    val_set_images = val_set_images.reshape((5000,28,28,1))
    val_set_labels = y_train[valset_ind]
    
    x_test = x_test.reshape((10000,28,28,1))
    
    f = h5py.File("mnist_train", "w")
    f.create_dataset('images', data=train_set_images)
    f.create_dataset('labels', data=train_set_labels)
    f.close()
    
    f = h5py.File("mnist_val", "w")
    f.create_dataset('images', data=val_set_images)
    f.create_dataset('labels', data=val_set_labels)
    f.close()
    
    f = h5py.File("mnist_test", "w")
    f.create_dataset('images', data=x_test)
    f.create_dataset('labels', data=y_test)
    f.close()


if __name__ == '__main__':
    main()
