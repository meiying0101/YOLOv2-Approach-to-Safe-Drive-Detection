"""
Convert Pascal VOC 2007+2012 detection dataset to HDF5.

Does not preserve full XML annotations.
Combines all VOC subsets (train, val test) with VOC2012 train for full
training set as done in Faster R-CNN paper.

Code based on:
https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
"""

import argparse
import os

import h5py
import numpy as np

train_set = 'train'
test_set = 'test'

classes = [
    "leftHand_driver", "rightHand_driver", "leftHand_passenger", "rightHand_passenger"
]

parser = argparse.ArgumentParser(
    description='Convert VIVA detection dataset to HDF5.')
parser.add_argument(
    '-p',
    '--path_to_viva',
    help='path to VIVA directory',
    default='../detectiondata')


def get_boxes_for_id(viva_path, dataset, image_id):
    """Get object bounding boxes annotations for given image.

    Parameters
    ----------
    viva_path : str
        Path to VIVA directory.
    year : str
        Year of dataset containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    boxes : array of int
        bounding box annotations of class label, xmin, ymin, xmax, ymax as a
        5xN array.
    """
    fname = os.path.join(viva_path, '{}/posGt/{}.txt'.format(dataset, image_id))
    rf = open(fname, 'r')
    boxes = []
    for line in rf.readlines():
        for i in range(len(classes)):
            if line.split()[0] == classes[i]: # "leftHand_driver", "rightHand_driver", "leftHand_passenger", "rightHand_passenger"
                x = int(line.split()[1])
                y = int(line.split()[2])
                w = int(line.split()[3])
                h = int(line.split()[4])
                bbox = (i, x,
                        y, x+w,
                        y+h)
                boxes.extend(bbox)
                #print(x,y,w,h)
    return np.array(boxes)  # .T  # return transpose so last dimension is variable length


def get_image_for_id(viva_path, dataset, image_id):
    """Get image data as uint8 array for given image.

    Parameters
    ----------
    viva_path : str
        Path to VIVA directory.
    year : str
        Year of dataset containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    image_data : array of uint8
        Compressed JPEG byte string represented as array of uint8.
    """
    fname = os.path.join(viva_path, '{}/pos/{}.png'.format(dataset, image_id))

    with open(fname, 'rb') as in_file:
        data = in_file.read()
    # Use of encoding based on: https://github.com/h5py/h5py/issues/745
    return np.fromstring(data, dtype='uint8')


def get_ids(viva_path, dataset):
    """Get image identifiers for corresponding list of dataset identifies.

    Parameters
    ----------
    viva_path : str
        Path to VIVA directory.
    datasets : list of str tuples
        List of dataset identifiers in the form of (year, dataset) pairs.

    Returns
    -------
    ids : list of str
        List of all image identifiers for given datasets.
    """
    dataset_path = os.path.join(viva_path, dataset, 'pos')
    ids = [dir.replace('.png', '') for dir in os.listdir(dataset_path)]
    return ids


def add_to_dataset(viva_path, dataset, ids, images, boxes, start=0):
    """Process all given ids and adds them to given datasets."""
    for i, viva_id in enumerate(ids):
        image_data = get_image_for_id(viva_path, dataset, viva_id)
        image_boxes = get_boxes_for_id(viva_path, dataset, viva_id)
        images[start + i] = image_data
        boxes[start + i] = image_boxes
    return i


def _main(args):
    viva_path = os.path.expanduser(args.path_to_viva)
    train_ids = get_ids(viva_path, train_set)
    test_ids = get_ids(viva_path, test_set)

    # Create HDF5 dataset structure
    print('Creating HDF5 dataset structure.')
    fname = os.path.join(viva_path, 'viva.hdf5')
    viva_h5file = h5py.File(fname, 'w')
    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))  # variable length uint8
    vlen_int_dt = h5py.special_dtype(
        vlen=np.dtype(int))  # variable length default int
    train_group = viva_h5file.create_group('train')
    test_group = viva_h5file.create_group('test')

    # store class list for reference class ids as csv fixed-length numpy string
    viva_h5file.attrs['classes'] = np.string_(str.join(',', classes))

    # store images as variable length uint8 arrays
    train_images = train_group.create_dataset(
        'images', shape=(len(train_ids), ), dtype=uint8_dt)
    test_images = test_group.create_dataset(
        'images', shape=(len(test_ids), ), dtype=uint8_dt)

    # store boxes as class_id, xmin, ymin, xmax, ymax
    train_boxes = train_group.create_dataset(
        'boxes', shape=(len(train_ids), ), dtype=vlen_int_dt)
    test_boxes = test_group.create_dataset(
        'boxes', shape=(len(test_ids), ), dtype=vlen_int_dt)

    # process all ids and add to datasets
    print('Processing VIVA training set.')
    add_to_dataset(viva_path, 'train', train_ids, train_images, train_boxes)
    print('Processing VIVA testing set.')
    add_to_dataset(viva_path, 'test', test_ids, test_images, test_boxes)

    print('Closing HDF5 file.')
    viva_h5file.close()
    print('Done.')


if __name__ == '__main__':
    _main(parser.parse_args())
