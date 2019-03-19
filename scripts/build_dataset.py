import os
import random
import numpy as np
from osgeo import gdal
from PIL import Image, ImageFilter


directory = '../public'

location = 'Yos'

patch_sizes = [512, 256]

num_samples = {
    'train': 400,
    'test': 20,
    'val': 20
}

def read_raw_channel_as_array(directory, location, channel_name):
    raw_channel_names = {
        'aspect':           location + '_Aspect_Final',
        'contour100':       location + '_Contour100_Final',
        'hillshade':        location + '_Hillshade_Final',
        'hydro':            location + '_Hydro_Final',
        'slope':            location + '_Slope_Final',
        'topo':             location + '_Topo_Final'
    }

    # TODO - Add try/catch here for error is no file exists or fail on load
    geo = gdal.Open(directory +'/' + location + '/raw/' + raw_channel_names[channel_name] + '.img')
    return geo.ReadAsArray()

# default raw channel data
raw_channel_data = {
    'aspect':           read_raw_channel_as_array(directory, location, 'aspect'),
    'contour100':       read_raw_channel_as_array(directory, location, 'contour100'),
    'hillshade':        read_raw_channel_as_array(directory, location, 'hillshade'),
    'hydro':            read_raw_channel_as_array(directory, location, 'hydro'),
    'slope':            read_raw_channel_as_array(directory, location, 'slope'),
    'topo':             read_raw_channel_as_array(directory, location, 'topo')
}

# dups for generated data
# raw_channel_data['aspectbin8'] = raw_channel_data['aspect']
# raw_channel_data['hillshadebin4'] = raw_channel_data['hillshade']
# raw_channel_data['slopebin5'] = raw_channel_data['slope']

raw_channel_data['grid8bin2'] = raw_channel_data['topo']
raw_channel_data['grid8bin4'] = raw_channel_data['topo']
raw_channel_data['grid16bin2'] = raw_channel_data['topo']
raw_channel_data['grid16bin4'] = raw_channel_data['topo']


# ?? ------------------- ADD SCRIPT VARS ABOVE THIS LINE --------------------------------------------

# sets the random seed for repeating the sampling numbers
random.seed(2019)

# Scaling function
def scale_array(X, x_min, x_max):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom / denom

# Bins the Z value into a certain number of bins
def bin_array(X, num_bins):
    return scale_array(X, 0, num_bins)

# Set Val in np array to new val
def set_data_val(X, val, new_val):
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            if X[i][j] == val:
                X[i][j] = new_val

# Set Vals based on divisions
def bin_array_divs(X, divs):
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            val = 0
            for d in range(len(divs)):
                if X[i][j] >= divs[d]:
                    val += 1
            X[i][j] = val

# channel specific processing procedures...should return A PIL image (8-bit) between 0 - 255 vals
def process_channel(data_arr, channel, patch_size):
    if channel == 'grid8bin2':
        patch = data_arr.copy()
        patch = scale_array(patch, 0, 256)
        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((8, 8), resample=Image.HAMMING)

        # binary thresehold
        for i in range(0, 8):
            for j in range(0, 8):
                val = image.getpixel((i, j))
                if val < 128: val = 0
                else: val = 255
                image.putpixel((i, j), val)

        image = image.resize((patch_size, patch_size), resample=Image.BOX)
        return image
    
    elif channel == 'grid8bin4':
        patch = data_arr.copy()
        patch = scale_array(patch, 0, 256)
        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((8, 8), resample=Image.HAMMING)

        # binary thresehold
        for i in range(0, 8):
            for j in range(0, 8):
                val = image.getpixel((i, j))
                if val < 64: val = 0
                elif val >= 64 and val < 128: val = 85
                elif val >= 128 and val < 192: val = 170
                else: val = 255
                image.putpixel((i, j), val)

        image = image.resize((patch_size, patch_size), resample=Image.BOX)
        return image

    elif channel == 'grid16bin2':
        patch = data_arr.copy()
        patch = scale_array(patch, 0, 256)
        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((16, 16), resample=Image.HAMMING)

        # binary thresehold
        for i in range(0, 16):
            for j in range(0, 16):
                val = image.getpixel((i, j))
                if val < 128: val = 0
                else: val = 255
                image.putpixel((i, j), val)


        image = image.resize((patch_size, patch_size), resample=Image.BOX)
        return image
    
    elif channel == 'grid16bin4':
        patch = data_arr.copy()
        patch = scale_array(patch, 0, 256)
        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((16, 16), resample=Image.HAMMING)

        # binary thresehold
        for i in range(0, 16):
            for j in range(0, 16):
                val = image.getpixel((i, j))
                if val < 64: val = 0
                elif val >= 64 and val < 128: val = 85
                elif val >= 128 and val < 192: val = 170
                else: val = 255
                image.putpixel((i, j), val)

        image = image.resize((patch_size, patch_size), resample=Image.BOX)
        return image

    elif channel == 'hydro':
        patch = data_arr.copy()
        for i in range(0, patch.shape[0]):
            for j in range(0, patch.shape[1]):
                if patch[i][j] == 1: patch[i][j] = 255

        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((patch_size, patch_size), resample=Image.HAMMING)
        # image = image.filter(ImageFilter.GaussianBlur(radius=int(patch_size/100)))
        return image
    # Default Process for general 8-bit conversion

    elif channel == 'contour100':
        patch = data_arr.copy()
        for i in range(0, patch.shape[0]):
            for j in range(0, patch.shape[1]):
                if patch[i][j] > 65000 or patch[i][j] == 0: patch[i][j] = 0
                else: patch[i][j] = 255

        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((patch_size, patch_size), resample=Image.HAMMING)
        # image = image.filter(ImageFilter.GaussianBlur(radius=int(patch_size/100)))
        return image

    elif channel == 'slope':
        patch = data_arr.copy()
        d_min = patch.min()
        if d_min < -3.4e30:
            set_data_val(patch, d_min, 0.0)

        patch = scale_array(patch, 0, 256)
        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        image = image.resize((patch_size, patch_size), resample=Image.HAMMING)
        image = image.filter(ImageFilter.GaussianBlur(radius=int(patch_size/100)))
        return image
    
    else:
        patch = data_arr.copy()
        d_min = patch.min()
        if d_min < -3.4e30:
            set_data_val(patch, d_min, 0.0)

        patch = np.copy(patch)
        patch = scale_array(patch, 0, 256)
        patch = patch.astype(np.uint8)
        image = Image.fromarray(patch, mode='L')
        return image.resize((patch_size, patch_size), resample=Image.HAMMING)


#?? --------------------------PATCH GENERATION ----------------------------------------------------


# Saves the patch_data dict
def save_patches(patch_data, directory, location, patch_size, scale, data_type):
    # Create the Directory Structure for the patches
    for channel in patch_data.keys():
        dir = '%s/%s/%s/%s/%s/%s' % (directory, location, patch_size, scale, data_type, channel)
        os.makedirs(dir, exist_ok=True)

        # Save all of the channel data to disk
        for i, data in enumerate(patch_data[channel]):
            data.save('%s/%s.png' % (dir, str(i+1)))

# generate the patches for a given location, patch_size, and scale
def generate_patches(raw_data, sample_coords, directory, location, patch_size, scale):
    for data_type in sample_coords.keys():  # Train, Test, Val
        # Dict containg PIL image data of each sampled patch
        patch_data = dict()
        
        # First generate the data channels from the preprocessed Raw Data
        for channel in raw_data.keys():
            print('Building %s/%s/%s/%s' % (patch_size, scale, data_type, channel))

            data_samples = []
            raw_data_arr = raw_data[channel]
            for sample_coord in sample_coords[data_type]:
                # dims to sample at
                # EX: patch size of 256 at scale 4 has sample dimensions of 1024x1024 and is downscaled to 256x256
                sample_dim = patch_size * scale
                half_sample_dim = int(sample_dim / 2)

                y_min = sample_coord[0] - half_sample_dim
                y_max = y_min + sample_dim
                x_min = sample_coord[1] - half_sample_dim
                x_max = x_min + sample_dim

                # Slice the data from the raw data
                data_sample_arr = raw_data_arr[y_min:y_max, x_min:x_max]
        
                # Apply any special processing to the data such as bining or scaling...
                data_sample_img = process_channel(data_sample_arr, channel, patch_size)

                # Add to the data samples Array
                data_samples.append(data_sample_img)
            patch_data[channel] = data_samples

        save_patches(patch_data, directory, location, patch_size, scale, data_type)

# Generates the dataset for a given location
def generate_dataset(directory, location, patch_sizes, num_samples):
    # initial patch size to sample, ex: 1024... other scaled patch sizes are downsampled from this
    # ex: 512 scale 2 = 1024 sample downsampled by a factor of 2
    initial_patch_size = patch_sizes[0]

    # Core Dimensions of the DEM
    # TODO - test the xy shape on non square input
    y_raw = raw_channel_data['topo'].shape[0] 
    x_raw = raw_channel_data['topo'].shape[1]

    # Sampling Bounds set to largest patch size, set a saftey buffer of a few pixels in all dims in case raw data is slightly misaligned
    y_min = int(initial_patch_size / 2) + 1
    y_max = y_raw - int(initial_patch_size / 2) - 3
    x_min = int(initial_patch_size / 2) + 1
    x_max = x_raw - int(initial_patch_size / 2) - 3

    sample_coords = {
        'train':        [(random.randint(y_min, y_max), random.randint(x_min, x_max)) for _ in range(num_samples['train'])],
        'test':         [(random.randint(y_min, y_max), random.randint(x_min, x_max)) for _ in range(num_samples['test'])],
        'val':          [(random.randint(y_min, y_max), random.randint(x_min, x_max)) for _ in range(num_samples['val'])]
    }

    # print(sample_coords['train'])

    for i, patch_size in enumerate(patch_sizes):
        scales = [s for s in range(1, 2 ** i + 1)]
        for scale in scales:
            generate_patches(raw_channel_data, sample_coords, directory, location, patch_size, scale)

#?? ------------------------------------- MAIN----------------------------------------------------

generate_dataset(directory, location, patch_sizes, num_samples)
