from __future__ import print_function, division

import os

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, set_name))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        print('classes', self.classes)
        print('coco labels', self.coco_labels)
        print('Inverser', self.coco_labels_inverse)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    # 根据coco文件夹中的注释，读取其中的图片文件
    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'patches', image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations

        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 7))

        # some images appear to miss annotations (like image with id 257034) 或者就是没有注释(在本任务中的采样方法)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 7))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'] - 1)
            annotation[0, 5] = self.scannerId_to_label(a['scanner_id'])
            annotation[0, 6] = self.cancerId_to_label(a['tumor_label'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def scannerId_to_label(self, scanner_id):
        if scanner_id == "3DHistech Pannoramic Scan II":
            label = 1
        elif scanner_id == "Aperio ScanScope CS2":
            label = 2
        elif scanner_id == "Hamamatsu NanoZoomer XR":
            label = 3
        else:
            label = 0
        return label

    def cancerId_to_label(self, cancer_id):
        if cancer_id == "breast":
            label = 1
        elif cancer_id == "lung":
            label = 2
        elif cancer_id == "lymphoma":
            label = 3
        elif cancer_id == "mast celll tumor":
            label = 4
        elif cancer_id == "paostate":
            label = 5
        elif cancer_id == "melanoma":
            label = 6
        else:
            label = 0
        return label

    def num_classes(self):
        return 2

