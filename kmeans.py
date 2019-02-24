
import os
import shutil

import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans
import numpy as np

def get_image_ids(root):
    path = f"{root}/VOC2012"
    with open(f"{path}/ImageSets/Main/trainval.txt") as f:
        fnames = []
        for line in f.read().strip().split('\n'):
            cols = line.split()
            if len(cols) > 1:
                score = cols[1]
                if score != '1':
                    continue
            fnames.append(cols[0])
        return fnames
def get_detection(node):
    bndbox = node.find('bndbox')
    top = float(bndbox.find('ymin').text)
    left = float(bndbox.find('xmin').text)
    right = float(bndbox.find('xmax').text)
    bottom = float(bndbox.find('ymax').text)
    width = right-left
    height = bottom-top
    return [width, height]

def get_image_detection(root, image_id):
    path = f"{root}/VOC2012"
    image_path = f"{path}/JPEGImages/{image_id}.png"
    if not os.path.isfile(image_path):
        raise Exception(f"Expected {image_path} to exist.")
    annotation_path = f"{path}/Annotations/{image_id}.xml"
    if not os.path.isfile(annotation_path):
        raise Exception(f"Expected annotation file {annotation_path} to exist.")
    tree = ET.parse(annotation_path)
    xml_root = tree.getroot()
    size = xml_root.find('size')
    segmented = xml_root.find('segmented').text == '1'
    segmented_path = None
    if segmented:
        segmented_path = f"{path}/SegmentationObject/{image_id}.png"
        if not os.path.isfile(segmented_path):
            raise Exception(f"Expected segmentation file {segmented_path} to exist.")
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)

    detection = [get_detection(node) for node in xml_root.findall('object')]
    return detection
    


image_names = get_image_ids('../data/progress')
detection_all = []
for image_name in image_names:
    detection = get_image_detection('../data/progress', image_name)
    detection_all+=(detection)

# print(detection_all)
detection_all = np.array(detection_all)
# print(detection_all)

kmeans = KMeans(n_clusters=9, random_state=0).fit(detection_all)
print(kmeans.cluster_centers_)

