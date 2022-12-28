import random
import cv2, os
import json
import numpy as np
import hydra
from tqdm import tqdm


def decode_label(id):
    id += 1
    if 1 <= id <= 150:
        scanner, tumor_label = "unknown", "unknown"
    elif 150 <= id <= 194:
        scanner, tumor_label = "3DHistech Pannoramic Scan II", "breast"
    elif 195 <= id <= 249:
        scanner, tumor_label = "3DHistech Pannoramic Scan II", "lung"
    elif 250 <= id <= 299:
        scanner, tumor_label = "Aperio ScanScope CS2", "lymphoma"
    elif 300 <= id <= 354:
        scanner, tumor_label = "Hamamatsu NanoZoomer XR", "paostate"
    elif 354 <= id <= 404:
        scanner, tumor_label = "Hamamatsu NanoZoomer XR", "melanoma"
    else:
        scanner, tumor_label = "unknown", "unknown"
    return scanner, tumor_label


@hydra.main(config_path='./configs', config_name="preprocess")
def main(cfg):
    with open(cfg.read_dir.json, 'r') as f:
        coco = json.load(f)

    coco_images, coco_anns = coco['images'], coco['annotations']
    num_images, num_anns = len(coco_images), len(coco_anns)
    images, annotations = list(), list()
    image_index, ann_index = 0, 0

    for i, data in enumerate(tqdm(coco_images, colour='green', leave=True, position=0)):
        ## getting the file name from the json property
        image_id = data['file_name']
        ## reading the image using opencv
        img = cv2.imread(os.path.join(cfg.read_dir.image, image_id))
        ## getting the shape of the image
        image_h, image_w = img.shape[0], img.shape[1]

        ## now getting all the bboxes for the ith image
        bbox, category = list(), list()
        for j in range(0, num_anns):
            if coco_anns[j]["image_id"] == data["id"]:
                bbox.append(coco_anns[j]["bbox"])
                category.append(coco_anns[j]["category_id"])

        # now for each bbox, we need to segment the picture
        for j in range(len(bbox)):
            box = bbox[j]
            ## increment variables for each new image and annotations created from the
            ## original image
            image_index += 1
            ann_index += 1

            cx, cy = int(box[0]), int(box[1])
            cx, cy = max(cx, 0), max(cy, 0)

            x1 = max(cx - 231, 0)
            y1 = max(cy - 231, 0)
            x1 = min(x1, image_w - cfg.size.patch)
            y1 = min(y1, image_h - cfg.size.patch)

            slice = img[y1: y1 + cfg.size.patch, x1: x1 + cfg.size.patch]
            images.append(
                {
                    "license": 1,
                    "file_name": '{}_{}_{}.png'.format(i + 1, cx, cy),
                    "id": image_index, "width": cfg.size.patch,
                    "height": cfg.size.patch
                }
            )
            annotations.append(
                {
                    "iscrowd": 0,
                    "area": 2500,
                    "bbox": [cx - x1, cy - y1, 50, 50],
                    "category_id": category[j],
                    "image_id": image_index,
                    "id": ann_index,
                    "scanner_id": decode_label(i)[0],
                    "tumor_label": decode_label(i)[1]
                }
            )

            for _ in bbox:
                xx, yy = int(box[0]), int(box[1])
                if (x1 <= xx <= x1 + 512 and y1 <= yy <= y1 + 512 and cx != xx):
                    ann_index += 1

                    annotations.append(
                        {
                            "iscrowd": 0,
                            "area": 2500,
                            "bbox": [cx - xx, cy - yy, 50, 50],
                            "category_id": category[j],
                            "image_id": image_index,
                            "id": ann_index,
                            "scanner_id": decode_label(i)[0],
                            "tumor_label": decode_label(i)[1]
                        }
                    )
            cv2.imwrite(os.path.join(cfg.write_dir.image, '{}_{}_{}.png'.format(i + 1, cx, cy)), slice)

            

    coco.update({"info": {"description": "MItosis Domain Generalization Challenge (MIDOG) 2022 - Training set",
                          "version": "1.0", "year": 2022,
                          "contributor": "Marc Aubreville, Christof Bertram, Mitko Veta, Robert Klopfleisch, Nikolas Stathonikos, Samir Jabari, Taryn Donovan, Katharina Breininger",
                          "date_created": "2022/05/13"}, "licenses": [
        {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 1,
         "name": "Attribution-NonCommercial-NoDerivs License"}]})
    coco.update({"images": images})
    coco.update({"categories": [{"id": 0, "name": "mitotic figure"}, {"id": 1, "name": "not mitotic figure"}]})
    coco.update({"annotations": annotations})

    data = json.dumps(coco, indent=1)
    with open(cfg.write_dir.json, 'w', encoding='utf-8', newline='\n') as f:
        f.write(data)

    print('Finished the operation!')


if __name__ == '__main__':
    main()
