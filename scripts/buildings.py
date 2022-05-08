import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage
import rasterio

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import maskRCNN
sys.path.append(ROOT_DIR)
from mrcnn1.config import Config
from mrcnn1 import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


###################################################
# Configuration
###################################################

class BuildingConfig(Config):
    """
    Configuration for training on the toy dataset.
    Drives from the base Config class and overrides some values
    """
    # Give the configuration a recognizable name
    NAME = "building"

    # We use a GPU with 12GB memory,which can fit two images.
    # Adjust down if you use a smaller GPU
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with <90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    USE_MINI_MASK = False

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)


#####################################################
# Dataset
#####################################################

class BuildingDataset(utils.Dataset):

    def load_building(self, dataset_dir, subset, poly_mask=None):

        self.poly_mask = poly_mask
        # Add classes, there is only one class to add
        self.add_class("building", 1, "building")  # source, class id, class name

        # Train or validation dataset
        assert subset in ['train', 'val']
        dataset_dir = os.path.join(dataset_dir, subset)

        # load the annotation from the image boundary
        """
        image_dict = {"image_path": single_image_path, "image_name": basename,"annotations": []}
        label_dict = {"label": 'building'}

        annotations = {"img_id": 
                            {"image_path": single_image_path, 
                            "image_name": basename,
                            "annotations": [
                                {"label": "building",
                                 "regions":{
                                        minx: minx,
                                        maxx: maxx,
                                        miny: miny,
                                        maxy: maxy
                                 }}
                            ]
                            }
                        }
        """

        """
                [{
                    'image_path':'',
                    'width': '',
                    'height':'',
                    'annotations':[
                                    {
                                    'label': 'building',
                                    'region':
                                            {
                                            "all_points_x": [],
                                            "all_points_y": []
                                            }
                                    }
                                ]
                }]
                """

        annotations = json.load(open(os.path.join(dataset_dir, "annotation.json")))
        annotations = list(annotations.values())  # dont need the dict keys

        # annotations = [a['annotations'] for a in annotations if len(a['annotations'])>0]  # dict in a list

        # Add images
        for a in annotations:
            """
            Get the x, y coordinates of points of the polygons that make up the outline of each object instance.
            These are stores in the shape_attribute (see json format above. The sample only includes one instance in
            the image, so the key is 0; )
            """
            # load_mask() needs the image size to convert polygons to masks.
            polygons = [p["region"] for p in a["annotations"]]

            # Read the image shapes and conver to a mask

            self.add_image(
                "building",
                image_id=a['image_name'],
                path=a['image_path'],
                width=a['width'], height=a['height'],
                polygons=polygons)

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        # if image_info["source"] != "buildings":
        #    return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        # create an empty mask N dimensional mask, each dimension mask means an instance
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.int8)

        for i, p in enumerate(info["polygons"]):
            # p is a dictionary of instance, loop each instance in info["polygons"]
            # Get indexes of pixels inside the polygon and set them to 1

            """
            # vertices

            minx = p['region']['minx']
            miny = p['region']['miny']
            maxx = p['region']['maxx']
            maxy = p['region']['maxy']
            """

            minx = p['minx']
            miny = p['miny']
            maxx = p['maxx']
            maxy = p['maxy']

            if minx < 0:
                minx = 0
            if miny < 0:
                miny = 0

            mask[minx:maxx, maxy:miny, i] = 1

        """
        # this building block is to pass vertices polygon reading, above uses the bounding box

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        """

        # Return mask, and array of class IDs of each instance
        # one class ID only, we return an array of 1s

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        # return the path of the image
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model"""
    # Training dataset
    dataset_train = BuildingDataset()
    dataset_train.load_building(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BuildingDataset()
    dataset_val.load_building(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=2 * config.LEARNING_RATE,
                epochs=30,
                layers="heads")

    history = model.keras_model.history.history


def color_splash(image, mask):
    """Apply color splash effect
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255  # (why multiply 255?)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We are treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)  # If axis is negative it counts from the last to the first axis.
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # image = image[:,:,[0,1,2]]
        # Detect objects
        r = model.detect([image], verbose=1)[0]  # detect: returns a list of dicts, one dicts per image.
        # Color spash
        splash = color_splash(image, r["masks"])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)

            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to ave image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BuildingConfig()
    else:
        class InferenceConfig(BuildingConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


