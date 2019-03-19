import scipy
# import imageio
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random

class DataLoader():
    def __init__(self, directory, location, size, scale, data_type, channelA, channelB):
        self.directory = directory
        self.location = location
        self.size = size
        self.scale = scale
        self.data_type = data_type
        self.channelA = channelA
        self.channelB = channelB

        self.path = '%s/%s/%s/%s/%s' % (directory, location, size, scale, data_type)

        self.pathsA = glob('%s/%s/*' % (self.path, self.channelA))
        self.pathsB = glob('%s/%s/*' % (self.path, self.channelB))


        self.dataA = [np.reshape(self.imread(img), (size, size, 1)) / 127.5 - 1 for img in self.pathsA]
        print('Loaded A channels')

        self.dataB = [np.reshape(self.imread(img), (size, size, 1)) / 127.5 - 1 for img in self.pathsB]
        print('Loaded B channels')

        self.numData = min( [len(self.dataA), len(self.dataB)] )

    def load_data(self, batch_size=1, is_testing=False):
        imgs_A, imgs_B = [], []

        for _ in range(batch_size):
            rand_idx = random.randint(0, self.numData - 1)
            imgs_A.append(self.dataA[rand_idx])
            imgs_B.append(self.dataB[rand_idx])

        return np.array(imgs_A), np.array(imgs_B)

        # data_type = "train" if not is_testing else "test"
        
        # pathA = glob('./datasets/%s/%s/%s/*' % (self.dataset_name, data_type, self.channelA))
        # pathB = glob('./datasets/%s/%s/%s/*' % (self.dataset_name, data_type, self.channelB))

        # # select random index from image paths
        # batch_images = []
        # for bi in range(batch_size):
        #     # idx = random.randint(0, len(pathA)-1)
        #     img_pair = (pathA[bi * 3], pathB[bi * 3])
        #     batch_images.append(img_pair)

        # imgs_A = []
        # imgs_B = []
        # for img_path in batch_images:
        #     img_A = self.imread(img_path[0])
        #     img_B = self.imread(img_path[1])

        #     # img_A = scipy.misc.imresize(img_A, self.img_res)
        #     # img_B = scipy.misc.imresize(img_B, self.img_res)

        #     # If training => do random flip
        #     if not is_testing and np.random.random() < 0.5:
        #         img_A = np.fliplr(img_A)
        #         img_B = np.fliplr(img_B)

        #     imgs_A.append(img_A)
        #     imgs_B.append(img_B)

        # imgs_A = np.array(imgs_A)/127.5 - 1.
        # imgs_B = np.array(imgs_B)/127.5 - 1.

        # return np.reshape(imgs_A, (batch_size, 256, 256, 1)), np.reshape(imgs_B, (batch_size, 256, 256, 1))


    def load_batch(self, batch_size=1, is_testing=False):
        imgs_A, imgs_B = [], []

        for _ in range(batch_size):
            rand_idx = random.randint(0, self.numData - 1)
            imgs_A.append(self.dataA[rand_idx])
            imgs_B.append(self.dataB[rand_idx])

        return np.array(imgs_A), np.array(imgs_B)

        # data_type = "train" if not is_testing else "val"

        # pathA = glob('%s/%s/%s/*' % (self.path, data_type, self.channelA))
        # pathB = glob('%s/%s/%s/*' % (self.path, data_type, self.channelB))

        # self.n_batches = int(len(pathA) / batch_size)

        # for i in range(self.n_batches-1):
        #     batchA = pathA[i*batch_size:(i+1)*batch_size]
        #     batchB = pathB[i*batch_size:(i+1)*batch_size]

        #     imgs_A, imgs_B = [], []
        #     for i in range(len(batchA)):
        #         img_A = self.imread(batchA[i])
        #         img_B = self.imread(batchB[i])

        #         # if not is_testing and np.random.random() > 0.5:
        #         #         img_A = np.fliplr(img_A)
        #         #         img_B = np.fliplr(img_B)

        #         imgs_A.append(np.reshape(img_A, (256, 256, 1)))
        #         imgs_B.append(np.reshape(img_B, (256, 256, 1)))

        #     imgs_A = np.array(imgs_A)/127.5 - 1.
        #     imgs_B = np.array(imgs_B)/127.5 - 1.

        #     yield imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='L').astype(np.float)
