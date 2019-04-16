from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import os
# import PIL
import numpy as np
import scipy.sparse
import uuid
import pickle
from .imdb import imdb
from model.utils.config import cfg


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class fabu(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'fabu_{}'.format(image_set))
        self._image_set = image_set
        self._data_path = self._get_default_path()
        self._classes = ('__background__',  # always index 0
                         'car', 'van', 'truck',
                         'pedestrian', 'person_sitting', 'cyclist', 'tram',
                         'misc', 'dontCare', 'otherdynamic'
                         )
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_path()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'fabu_data')

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, img_name):
        """
        Construct an image path from the image's "path" identifier.
        """
        img_path = os.path.join(self._data_path, 'image_2', img_name)
        assert os.path.exists(img_path), \
            'Path does not exist: {}'.format(img_path)
        return img_path

    def _load_image_set_path(self):
        """
        Load the paths listed in this dataset's image set file.
        """
        # Example path to image set file:
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_paths = [x.strip() for x in f.readlines()]
        return image_paths

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_fabu_annotation(img_path)
                    for img_path in self._image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'val':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_fabu_annotation(self, img_path):
        """
        Load image and bounding boxes info from XML file in the fabu format txt label file.
        """
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        filename = os.path.join(self._data_path, 'label_2', '{}.txt'.format(base_name))
        with open(filename, 'r') as fin:
            objs = [line.strip() for line in fin.readlines() if line.strip() != '']
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            (class_name, truncated, occluded, alpha,
             left, top, right, bottom, height, width, length, x, y, z, ry) = obj.split()

            # Make pixel indexes 0-based
            x1 = float(left)
            y1 = float(top)
            x2 = float(right)
            y2 = float(bottom)
            diffc = occluded
            difficult = 0 if diffc is None else int(float(diffc))
            ishards[ix] = difficult

            cls = self._class_to_ind[class_name.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def evaluate_detections(self, all_boxes, output_dir=None):
        pass


if __name__ == '__main__':
    d = fabu('train')
    res = d.roidb
    from IPython import embed; embed()
