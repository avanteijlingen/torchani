# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:08:53 2020

@author: avtei
"""

import torch
#import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import sys, h5py
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class anidataloader:

    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - ' + store_file)
        self.store = h5py.File(store_file, 'r')

    def h5py_dataset_iterator(self, g, prefix=''):
        """Group recursive iterator

        Iterate through all groups in all branches and return datasets in dicts)
        """
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                data = {'path': path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])

                        if isinstance(dataset, np.ndarray):
                            if dataset.size != 0:
                                if isinstance(dataset[0], np.bytes_):
                                    dataset = [a.decode('ascii')
                                               for a in dataset]
                        data.update({k: dataset})
                yield data
            else:  # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)"""
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        """Returns a list of all groups in the file"""
        return [g for g in self.store.values()]

    def iter_group(self, g):
        """Allows interation through the data in a given group"""
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        """Returns the requested dataset"""
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k][()])

                if isinstance(dataset, np.ndarray):
                    if dataset.size != 0:
                        if isinstance(dataset[0], np.bytes_):
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    def group_size(self):
        """Returns the number of groups"""
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    def cleanup(self):
        """Close the HDF5 file"""
        self.store.close()

def load(path, additional_properties=()):
    PROPERTIES = ('energies',)
    properties = PROPERTIES + additional_properties

    def h5_files(path):
        """yield file name of all h5 files in a path"""
        if os.path.isdir(path):
            for f in os.listdir(path):
                f = os.path.join(path, f)
                yield from h5_files(f)
        elif os.path.isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
            yield path

    def molecules():
        for f in h5_files(path):
            anidata = anidataloader(f)
            anidata_size = anidata.group_size()
            use_pbar = PKBAR_INSTALLED and verbose
            if use_pbar:
                pbar = pkbar.Pbar('=> loading {}, total molecules: {}'.format(f, anidata_size), anidata_size)
            for i, m in enumerate(anidata):
                yield m
                if use_pbar:
                    pbar.update(i)

    def conformations():
        for m in molecules():
            species = m['species']
            coordinates = m['coordinates']
            for i in range(coordinates.shape[0]):
                ret = {'species': species, 'coordinates': coordinates[i]}
                for k in properties:
                    if k in m:
                        ret[k] = m[k][i]
                yield ret

    return TransformableIterable(IterableAdapter(lambda: conformations()))
