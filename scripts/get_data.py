
import os
import sys
# import tensorflow_hub as hub
import pandas as pd
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import maskRCNN
sys.path.append(ROOT_DIR)
import geopandas as gpd
import DataProcessing as DP
import json
import skimage


def create_json(path, jsonpath):

    files = [os.path.join(path, f) for f in os.listdir(path)]
    file_dict = dict()

    for file in files:
        f = open(file)
        img_id = os.path.basename(file).split('.')[0] 
        file_dict[img_id] = json.load(f)

    with open(jsonpath, 'w') as js:
        json.dump(file_dict, js)

if __name__ == '__main__':

    """
    img = os.path.join(ROOT_DIR, "dataset/raw_image/NAIP_williamsburg_2016.tif")
    polys = os.path.join(ROOT_DIR, "dataset/raw_image/building_img_wgs84.geojson")
    out = os.path.join(ROOT_DIR, "dataset/processing_data/clipped")
    """
    
    data_path = os.path.join(ROOT_DIR, "datasets", "open_cities_ai_challenge_train_tier_1_source")
    label_path = os.path.join(ROOT_DIR, "datasets", "open_cities_ai_challenge_train_tier_1_labels")
    
    data_dict = dict()
    
    for data in os.listdir(data_path):
        
        if "image" not in data_dict:
            data_dict["image"] = list()
            data_dict["label"] = list()
        else:
            data_dict["image"].append(os.path.join(data_path, data, "image.tif"))
            data_dict["label"].append(os.path.join(label_path, data, "labels.geojson"))

    
    # test an image
    img = data_dict["image"][0]
    label = data_dict["label"][0]
    out = os.path.join(ROOT_DIR, "processing_data/clipped")

    # detector = hub.Module("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")
    # print(detector)

    # df = pd.read_csv(os.path.join(ROOT_DIR, "out_meta.csv"), encoding='utf-8', sep=',')
    a = DP.DataProcessing(img, label)
    
    # The data in clipped folder has been manually checked to reflect buildings in residential area
    a.get_patches(256, out, nrandom=True)
    a.create_meta_df(out, "building")
    a.split_train_val(out, split_rate=0.1)
