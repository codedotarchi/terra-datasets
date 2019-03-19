import os
import random
import numpy as np
from osgeo import gdal
from PIL import Image

geo = gdal.Open('imgn38w120_13.img')
arr = geo.ReadAsArray()

# Scaling function
def scaleArray(X, x_min, x_max):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom / denom

# Scale the Array

# arr = scaleArray(arr, 0, 2 ** bitdepth)

# Core Dimensions of the DEM
imgX = arr.shape[0]
imgY = arr.shape[1]

bitdepth = 8
location = '_test_dataset'

def generatePatches(folder, location, patchSize, scale, dataset, numPatches, loResSize=8, loResQuantize=True):
    
    # Patch Size
    patchX = patchSize * scale
    patchY = patchSize * scale

    for n in range(1, numPatches + 1):

        # Uniform Sample of DEM
        samplePixelX = random.randint(0, imgX - patchX)
        samplePixelY = random.randint(0, imgY - patchY)

        print(n)
        # print(samplePixelX)
        # print(samplePixelY)

        patch = arr[samplePixelX : samplePixelX + patchX, samplePixelY : samplePixelY + patchY]
        # patch = scaleArray(patch, 0, hiResBitDepth)
        patch = scaleArray(patch, 0, 2 ** bitdepth)
        patch = patch.astype(np.uint8)

        # print(patch.shape)
        # print(patch)

        hiResImg = Image.fromarray(patch, mode='L')
        hiResImg = hiResImg.resize((patchSize, patchSize), resample=Image.HAMMING)
        loResImg = hiResImg.resize((loResSize, loResSize), resample=Image.HAMMING)

        # If quantize, set values to 0 or 1
        if loResQuantize:
            for i in range(0, loResSize):
                for j in range(0, loResSize):
                    val = loResImg.getpixel((i, j))

                    if val < 2 ** bitdepth / 2:
                        val = 0
                    else:
                        val = 2 ** bitdepth - 1

                    loResImg.putpixel((i, j), val)

        # resize to hiRes size (currently at 256x256)
        loResImg = loResImg.resize((patchSize, patchSize), resample=Image.BOX)
        # print(loResImg)

        # Create a new image at 512 x 256
        # imgBand = Image.new('L', (hiResSize * 2, hiResSize), color=0)

        # for y in range(0, hiResSize):
        #     for x in range(0, hiResSize):
        #         imgBand.putpixel((x, y), hiResImg.getpixel((x, y)))
        #         imgBand.putpixel((x + hiResSize, y), loResImg.getpixel((x, y)))

        # merge single band images into multiband images
        # img = Image.merge('RGB', [imgBand, imgBand, imgBand])

        grid = 'grid_' + str(loResSize) + '_bin/'
        topo = 'topo/'
        path = folder + '/' + location + '/' + str(patchSize) + '/' + str(scale) + '/' + dataset + '/'

        # name = folder + str(n) + '.png'

        hiResImg.save(path + topo + str(n) + '.png')
        loResImg.save(path + grid + str(n) + '.png')
        # TODO: Add Thumbnail versions

for patchSize in [256]:
# for patchSize in [128, 256, 512]:
    for scale in [1, 2, 4, 8]:
        for dataset in ['train', 'test']:
            numPatches = 1000;
            if dataset == 'test':
                numPatches = 100
            generatePatches('../public', location, patchSize, scale, dataset, numPatches)



