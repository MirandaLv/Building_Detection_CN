import rasterio
import os
import numpy as np
import fiona
from shapely.geometry import Polygon, shape
import geopandas as gpd
import pandas as pd
import shapely
import json
import random
import shutil


class DataProcessing:

    def __init__(self, IMAGE_PATH, POLY_PATH):

        self.ImagePath = IMAGE_PATH
        self.PolyPath = POLY_PATH
#         self.ROOT_DIR = ROOT_DIR
        # self.polys = gpd.read_file(POLY_PATH)

    def create_meta_df(self, FolderPath, label):
        """
        :param FolderPath: the directory to path of image patches
        :param label: object label, may change to a list later for multi-object identification
        :return: a csv that might be used later for training inputs (tentative)
        """

        img_paths = [os.path.join(FolderPath, f) for f in os.listdir(FolderPath)]
        polys = gpd.read_file(self.PolyPath)
        outmeta = {'img_path': [], 'poly_ids': []}

        for img_path in img_paths:

            try:
                img = rasterio.open(img_path)
            except:
                print(img_path)
                raise

            bound_xmin, bound_ymin, bound_xmax, bound_ymax = img.bounds

            polys['centroid_x'] = polys['geometry'].centroid.x
            polys['centroid_y'] = polys['geometry'].centroid.y

            polys['inorout'] = polys.apply(lambda z: 'in' if bound_xmin < z['centroid_x'] < bound_xmax
                                                             and bound_ymin < z['centroid_y'] < bound_ymax else 'out',
                                           axis=1)

            poly_geos = polys[polys['inorout'] == 'in']['geometry'].tolist()
            poly_ids = polys[polys['inorout'] == 'in']['GEO_ID'].tolist()

            self.generate_annotation(img_path, poly_geos, label)

            outmeta['img_path'].append(img_path)
            outmeta['poly_ids'].append(poly_ids)

        # df = pd.DataFrame.from_dict(outmeta)
        # df.to_csv(os.path.join(self.ROOT_DIR, 'out_meta.csv'), encoding='utf-8', sep=',')

    def generate_annotation(self, single_image_path, polygeos, label):

        """
        :param single_image_path: image path
        :param polygeos: a list of shapely geometries
        :param label: object label
        :return: save the generated annotation files to annotations folder
        """

        try:
            img = rasterio.open(single_image_path)
            width = img.width
            height = img.height
        except:
            print(img)
            raise

        basename = os.path.basename(single_image_path)
        annotationfile = '/'.join(os.path.dirname(single_image_path).split('/')[:-1]) + '/annotations/' + \
                         basename.split('.')[0] + '.json'

        image_dict = {"image_path": single_image_path, "image_name": basename, "annotations": [], "width": width,
                      "height": height}

        for idx in range(len(polygeos)):
            # assert isinstance(polygeos[idx], shapely.geometry.polygon.Polygon)

            label_dict = {"label": label}
            regions = dict()

            minx, miny, maxx, maxy = polygeos[idx].bounds

            """
            if polygeos[idx].type == 'MultiPolygon':
                count = 0
                for i in polygeos[idx]:
                    x, y = i.exterior.coords.xy
                    count += 1

                if count != 1:
                    continue

            rlist, clist = self.xy2idx(img, list(x), list(y))

            regions['all_points_x'] = clist
            regions['all_points_y'] = rlist

            label_dict['region'] = regions
            image_dict['annotations'].append(label_dict)



            lur, luc = img.index(minx, maxy)
            brr, brc = img.index(maxx, miny)

            # win = ((r - dim / 2, r + dim / 2), (c - dim / 2, c + dim / 2))

            regions['minx'] = lur
            regions['maxy'] = luc
            regions['maxx'] = brr
            regions['miny'] = brc

            label_dict['region'] = regions
            image_dict['annotations'].append(label_dict)
            """

            # Just read the bounding box of a sample, above commented block is used to read all the vertices
            lur, luc = img.index(minx, maxy)
            brr, brc = img.index(maxx, miny)

            # win = ((r - dim / 2, r + dim / 2), (c - dim / 2, c + dim / 2))

            # TODO: add all vertices to the annotation file

            regions['minx'] = lur
            regions['maxy'] = luc
            regions['maxx'] = brr
            regions['miny'] = brc

            label_dict['region'] = regions
            image_dict['annotations'].append(label_dict)

        with open(annotationfile, 'w') as js:
            json.dump(image_dict, js)

    def xy2idx(self, img_array, xlist, ylist):

        assert len(xlist) == len(ylist)

        rlist = list()
        clist = list()

        for x, y in zip(xlist, ylist):
            r, c = img_array.index(x, y)
            rlist.append(r)
            clist.append(c)

        return rlist, clist

    def generate_json(self):

        image_dict = {"image": '', "annotations": []}
        label_dict = {"label": '', "coordinates": {}}
        # coord_dict = {"x": float, "y": float, "width": int, "height": int}

        coord_dict = {"geometry": self.polys['geometry']}

        label_dict["label"] = 'bird'
        label_dict["coordinates"] = coord_dict

        # image_dict["image"] = self.image_name
        image_dict["annotations"].append(label_dict)

        # annotations.append(image_dict)


    
    def get_patches(self, dim, Out_PATH, nrandom=False):

        """
        :param dim: the dimension of image patches
        :param Out_PATH: output path
        :return:
        """

        try:
            poly_shapes = fiona.open(self.PolyPath)
        except:
            print(self.PolyPath)
            raise

        try:
            img_raw = rasterio.open(self.ImagePath)
        except:
            print(self.ImagePath)
            raise

        out_meta = img_raw.meta

        if nrandom:
            # generate a list of random numbers, these will be used to extract training/val polygons
            ixlist = random.sample(range(0, len(poly_shapes)), 300)
        else:
            ixlist = range(0, len(poly_shapes))

        for ix in ixlist:

            img_id = poly_shapes[ix]['properties']['GEO_ID']
            img_out = os.path.join(Out_PATH, str(img_id) + '.tif')
            poly = shape(poly_shapes[ix]['geometry'])

            lon = poly.centroid.x
            lat = poly.centroid.y

            r, c = img_raw.index(lon, lat)
            win = ((r - dim / 2, r + dim / 2), (c - dim / 2, c + dim / 2))
            # window = rasterio.windows.Window(lon - dim//2, lat - dim//2, dim, dim)

            try:
                data = img_raw.read(window=win)
            except:
                print(win)
                raise

            out_meta.update({
                "width": dim,
                "height": dim,
                "transform": rasterio.windows.transform(win, img_raw.transform),
                "nodata": None
            })

            with rasterio.open(img_out, 'w', **out_meta) as dst:
                dst.write(data)

    def split_train_val(self, img_folder, split_rate=None):
      
      # split_rate is the percentage of training dataset among all data

        rpath = '/'.join(os.path.dirname(img_folder).split('/'))
        print(rpath)
        train_path = os.path.join(rpath, "train")
        val_path = os.path.join(rpath, "val")

        # create training and val folder

        try:
            os.mkdir(train_path)
        except OSError:
            print("Creation of the directory %s failed" % train_path)

        try:
            os.mkdir(val_path)
        except OSError:
            print("Creation of the directory %s failed" % val_path)

        img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)]
        # define the idx of images that are used as training dataset
        nt = len(img_paths) * split_rate
        tidxs = random.sample(range(0, len(img_paths)), int(nt))
        vidxs = [i for i in range(0, len(img_paths)) if i not in tidxs]

        trains_json = list()
        vals_json = list()

        for idx in tidxs:
            basename = os.path.basename(img_paths[idx])
            annotationfile = '/'.join(os.path.dirname(img_paths[idx]).split('/')[:-1]) + '/annotations/' + \
                             basename.split('.')[0] + '.json'

            shutil.copy(img_paths[idx], train_path)
            trains_json.append(annotationfile)

        for idx in vidxs:
            basename = os.path.basename(img_paths[idx])
            annotationfile = '/'.join(os.path.dirname(img_paths[idx]).split('/')[:-1]) + '/annotations/' + \
                             basename.split('.')[0] + '.json'

            shutil.copy(img_paths[idx], val_path)
            vals_json.append(annotationfile)

        self.agg_annotation(trains_json, train_path)
        self.agg_annotation(vals_json, val_path)

    def agg_annotation(self, annote_list, despath):

        file_dict = dict()
        desfile = os.path.join(despath, "annotation.json")
        for file in annote_list:
            f = open(file)
            img_id = os.path.basename(file).split('.')[0]
            file_dict[img_id] = json.load(f)

        with open(desfile, 'w') as js:
            json.dump(file_dict, js)









