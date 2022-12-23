import cv2, os
import json
import numpy as np
import hydra
from tqdm import tqdm

def decode_label(id):
    id += 1
    if (1 <= id <= 150): scanner, tumor_label = "unknown", "unknown"
    elif (150 <= id <= 194): scanner, tumor_label = "3DHistech Pannoramic Scan II", "breast"
    elif (195 <= id <= 249): scanner, tumor_label = "3DHistech Pannoramic Scan II", "lung"
    elif (250 <= id <= 299): scanner, tumor_label = "Aperio ScanScope CS2", "lymphoma"
    elif (300 <= id <= 354): scanner, tumor_label = "Hamamatsu NanoZoomer XR", "paostate"
    elif (354 <= id <= 404): scanner, tumor_label = "Hamamatsu NanoZoomer XR", "melanoma"
    else: scanner, tumor_label = "unknown", "unknown"
    return scanner, tumor_label


@hydra.main(config_path='./configs', config_name="preprocess")
def main(cfg):
    with open(cfg.read_dir.json, 'r') as f:
        coco = json.load(f)
    print(coco)


if __name__ == '__main__':
    main()

