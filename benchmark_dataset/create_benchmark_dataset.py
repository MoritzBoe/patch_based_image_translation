import numpy as np
import random
import os
from skimage.draw import circle, ellipse
import warnings
from pathlib import Path
from PIL import Image
import json


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Dataset:
    """This class creates a new benchmark dataset. After creation, methods can be added to the pipeline to fill
    images with objects.
    """
    def __init__(self, label_shape, target_shape, label_train_shape, num_images,
                 label_dtype="uint16", target_dtype="uint16", label_train_dtype="uint16"):
        """Init benchmark dataset

        Args:
            label_shape (tuple): Shape of label images
            target_shape (tuple): Shape of target images
            label_train_shape (tuple): Shape of label images for train. This can e.g. be a semantic segmentation of the instance
                                       segmentation created for the label images.
            num_images (int): Number of images
            label_dtype (str, optional): Data type of label images. Defaults to "uint16".
            target_dtype (str, optional): Data type of target images. Defaults to "uint16".
            label_train_dtype (str, optional): Data type of label images used for training a GAN. Defaults to "uint16".
        """
        self.label_shape = label_shape
        self.target_shape = target_shape
        self.label_train_shape = label_train_shape
        self.num_images = num_images
        self.label_dtype = label_dtype
        self.target_dtype = target_dtype
        self.label_train_dtype = label_train_dtype
        self.images = []

        for i in range(self.num_images):
            self.images.append({"label": np.zeros(self.label_shape, dtype=self.label_dtype),
                                "target": np.zeros(self.target_shape, dtype=self.target_dtype),
                                "label_train": np.zeros(self.label_train_shape, dtype=self.label_train_dtype)})

        self.pipeline = []

    def add_to_pipeline(self, shape, repetitions):
        """Add a geometric shape to the pipeline

        Args:
            shape (Shape object): The shape to add to the pipeline
            repetitions (int): Number of times this shape is placed in each image
        """
        self.pipeline.append({"shape": shape, "repetitions": repetitions})

    def run_pipeline(self):
        """Run the pipeline and place objects in the images
        """
        for idx, images in enumerate(self.images):
            for step in self.pipeline:
                for i in range(step["repetitions"]):
                    self.images[idx] = step["shape"].draw(self.images[idx])
            print(f"Images {idx} created")

    def save(self, path):
        """Save created dataset

        Args:
            path (Path): Pathlib Path to the folder of the new dataset
        """
        label_path = path.joinpath("labels")
        label_train_path = path.joinpath("labels_train")
        target_path = path.joinpath("targets")

        label_path.mkdir(parents=True, exist_ok=True)
        label_train_path.mkdir(parents=True, exist_ok=True)
        target_path.mkdir(parents=True, exist_ok=True)

        info = self.create_dataset_info()
        with open(path.joinpath("info.json"), "w") as f:
            json.dump(info, f, indent=2)
        for idx, image_set in enumerate(self.images):
            label_train = Image.fromarray(image_set["label_train"])
            label = Image.fromarray(image_set["label"])
            target = Image.fromarray(image_set["target"])

            label.save(label_path.joinpath(f"{idx:04d}.png"))
            label_train.save(label_train_path.joinpath(f"{idx:04d}.png"))
            target.save(target_path.joinpath(f"{idx:04d}.png"))

    def create_dataset_info(self):
        """Create the info of the pipeline used to create this dataset

        Returns:
            dict: Dict with all information to create the dataset
        """
        info = {"label_shape": self.label_shape, "target_shape": self.target_shape, "label_train_shape": self.label_train_shape, 
                "num_images": self.num_images, "label_dtype": self.label_dtype, "target_dtype": self.target_dtype,
                "label_train_dtype": self.label_train_dtype, "pipeline": []}

        for step in self.pipeline:
            info["pipeline"].append({"repetitions": step["repetitions"],
                                     "shape": type(self.pipeline[0]["shape"]).__name__,
                                     "shape_params": self.pipeline[0]["shape"].__dict__})
        return info


class EllipseWithDot:
    """Place ellipses with a dot inside in the images.
    Works only for 2D images
    """
    def __init__(self, eccentricity, equivalent_circle_diameter, max_overlap, dot_radius, random_rotation=False):
        """Initialize the pipeline step to place ellipses with a dot in the images

        Args:
            eccentricity (tuple): Mean, std-deviation, max of the eccentricity of the ellipses
            equivalent_circle_diameter (_type_): mean, std-deviation, max of the equivalent circle diameter of the ellipses
            max_overlap (float): Maximum overlap with existing elements in percent, with respect to the total size of the ellipse
            dot_radius (int): Radius of the dot inside the ellipses
            random_rotation (bool, optional): Rotate the ellipses or not. Defaults to False.
        """
        self.eccentricity = eccentricity
        self.equivalent_circle_diameter = equivalent_circle_diameter
        self.max_overlap = max_overlap
        self.minor_axis_length = 0
        self.major_axis_length = 0
        self.random_rotation = random_rotation
        self.dot_radius = dot_radius

    def draw(self, images):
        """Draw a single ellipse

        Args:
            images (dict of ndarray): Dict with ["label", "label_train", "target"] images

        Returns:
            dict of ndarray: Return the images with the new shape
        """
        shape = images["label"].shape
        eccentricity = self.get_eccentricity()
        equivalent_circle_diameter = self.get_equivalent_circle_diameter(eccentricity)
        minor_axis_length, major_axis_length = self.get_axis_length(eccentricity, equivalent_circle_diameter)
        rotation = self.get_rotation()

        for j in range(10000):  # try to place the ellipse 1000 times
            center = self.get_random_center(shape)
            x, y = ellipse(center[0], center[1], minor_axis_length, major_axis_length,
                           shape=shape, rotation=rotation)
            if self.check_no_overlap(images["label"], x, y):
                images["label"][...,x, y] = images["label"].max()+1
                images["label_train"][...,x, y] = np.iinfo(images["label_train"].dtype).max
                images["target"][...,x,y] = np.iinfo(images["target"].dtype).max
                # calculate smaller ellipse for possible dot center points
                dot_possibilities_x, dot_possibilities_y = ellipse(center[0], center[1], minor_axis_length-self.dot_radius,
                                                                   major_axis_length-self.dot_radius, shape=shape, rotation=rotation)
                random_pick = random.randint(0, len(dot_possibilities_x)-1)
                dot_x, dot_y = circle(dot_possibilities_x[random_pick], dot_possibilities_y[random_pick], self.dot_radius, shape=shape)
                images["target"][...,dot_x,dot_y] = np.iinfo(images["target"].dtype).min
                break
            if j == 9999:
                warnings.warn("No spot found to place the ellipse with the given radius and max_overlap!")
        return images

    def get_eccentricity(self):
        """Return eccentricity drawn from normal distribution in range [0, self.eccentricity[2]]

        Returns:
            float: Eccentricity [0, self.eccentricity[2]]
        """
        if self.eccentricity[1] != 0:
            eccentricity = np.random.normal(self.eccentricity[0], self.eccentricity[1])
        else:
            eccentricity = self.eccentricity[0]

        # Sanity checks (values from the normal distribution are not always [0,1)
        if eccentricity < 0:  # Circle
            eccentricity = 0
        elif eccentricity > self.eccentricity[2]:
            eccentricity = self.eccentricity[2]
        return eccentricity

    def get_equivalent_circle_diameter(self, eccentricity):
        """Return equivalent circle diameter drawn from normal distribution in range [0, self.equivalent_circle_diameter[2]]

        Returns:
            float: Equivalent circle diameter [0, self.equivalent_circle_diameter[2]]
        """
        # calculate minimal cirlce diameter to fit the dot (minor_axis_length>=dot+1)
        min_minor = self.dot_radius+1
        min_diameter = np.ceil(2*min_minor/(1-eccentricity**2)**(1/4))

        # Get equivalent_circle_diameter (Area of the ellipse) from distribution or given value
        if self.equivalent_circle_diameter[1] != 0:
            equivalent_circle_diameter = np.random.normal(self.equivalent_circle_diameter[0],
                                                          self.equivalent_circle_diameter[1])
        else:
            equivalent_circle_diameter = self.equivalent_circle_diameter[0]
        # Sanity checks (values from the normal distribution are not always [1,max)
        if equivalent_circle_diameter < min_diameter:
            equivalent_circle_diameter = min_diameter
        elif equivalent_circle_diameter >= self.equivalent_circle_diameter[2]:
            equivalent_circle_diameter = self.equivalent_circle_diameter[2]
        return equivalent_circle_diameter

    def get_axis_length(self, eccentricity, equivalent_circle_diameter):
        """Calculate major and minor axis length from eccentricity and equivalent circle diameter

        Args:
            eccentricity (float): Eccentricity of the ellipse
            equivalent_circle_diameter (float): Equivalent circle diameter of the ellipse

        Returns:
            float, float: Major axis length and minor axis length
        """
        equivalent_circle_radius = equivalent_circle_diameter/2
        major_axis_length = equivalent_circle_radius/(1-eccentricity**2)**(1/4)
        # self.minor_axis_length = math.sqrt(equivalent_circle_radius**2*math.sqrt(1-eccentricity**2))
        minor_axis_length = equivalent_circle_radius**2/major_axis_length
        return minor_axis_length, major_axis_length

    def get_random_center(self, img_size):
        """Get random center of the ellipse

        Args:
            img_size (tuple): Shape of image

        Returns:
            tuple: Random center
        """
        center = (random.randint(0, img_size[0]-1), random.randint(0, img_size[1]-1))
        return center

    def get_rotation(self):
        """Get random rotation

        Returns:
            float: Random rotation of ellipse
        """
        if self.random_rotation:
            rotation = np.random.uniform(-np.pi, np.pi)
        return rotation

    def check_no_overlap(self, label, x, y):
        """Check if chosen position is OK and overlap is smaller than self.max_overlap

        Args:
            label (ndarray): Label image with objects already present
            x (ndarray): Array containing x coordinates of ellipse
            y (ndarray): Array containing y coordinates of ellipse

        Returns:
            bool: True, when object can be placed at chosen position
        """
        im_ellipse = np.zeros(label.shape)
        im_ellipse[x, y] = np.iinfo('uint16').max
        ellipse_pixels = np.sum(im_ellipse > 0)
        im_overlap = im_ellipse*label
        overlap_pixels = np.sum(im_overlap > 0)
        overlap_percent = overlap_pixels/ellipse_pixels*100
        if overlap_percent <= self.max_overlap:
            return True
        else:
            return False


class EllipseWithColor:
    """Place ellipses of a distinct color, size and overlap in the images.
    Works only for 2D images
    """
    def __init__(self, eccentricity, equivalent_circle_diameter, colors, max_overlap, random_rotation=False):
        """Initialize the pipeline step to place ellipses of distinct colors in the images

        Args:
            eccentricity (tuple): Mean, std-deviation, max of the eccentricity of the ellipses
            equivalent_circle_diameter (_type_): mean, std-deviation, max of the equivalent circle diameter of the ellipses
            colors (list of tuples): List of tuples containing the colors in R,G,B
            max_overlap (float): Maximum overlap with existing elements in percent, with respect to the total size of the ellipse
            random_rotation (bool, optional): Rotate the ellipses or not. Defaults to False.
        """
        self.eccentricity = eccentricity
        self.equivalent_circle_diameter = equivalent_circle_diameter
        self.max_overlap = max_overlap
        self.minor_axis_length = 0
        self.major_axis_length = 0
        self.random_rotation = random_rotation
        self.colors = colors

    def draw(self, images):
        """Draw a single ellipse

        Args:
            images (dict of ndarray): Dict with ["label", "label_train", "target"] images

        Returns:
            dict of ndarray: Return the images with the new shape
        """
        shape = images["label"].shape
        eccentricity = self.get_eccentricity()
        equivalent_circle_diameter = self.get_equivalent_circle_diameter(eccentricity)
        minor_axis_length, major_axis_length = self.get_axis_length(eccentricity, equivalent_circle_diameter)
        rotation = self.get_rotation()

        for j in range(10000):  # try to place the ellipse 1000 times
            center = self.get_random_center(shape)
            x, y = ellipse(center[0], center[1], minor_axis_length, major_axis_length,
                           shape=shape, rotation=rotation)
            if self.check_no_overlap(images["label"], x, y):
                images["label"][x, y,...] = images["label"].max()+1
                images["label_train"][x, y,...] = np.iinfo(images["label_train"].dtype).max
                images["target"][x,y,...] = np.array(self.colors[np.random.randint(0, len(self.colors))])
                break
            if j == 9999:
                warnings.warn("No spot found to place the ellipse with the given radius and max_overlap!")
        return images

    def get_eccentricity(self):
        """Return eccentricity drawn from normal distribution in range [0, self.eccentricity[2]]

        Returns:
            float: Eccentricity [0, self.eccentricity[2]]
        """
        # Get eccentricity from distribution or given value
        if self.eccentricity[1] != 0:
            eccentricity = np.random.normal(self.eccentricity[0], self.eccentricity[1])
        else:
            eccentricity = self.eccentricity[0]

        # Sanity checks (values from the normal distribution are not always [0,1)
        if eccentricity < 0:  # Circle
            eccentricity = 0
        elif eccentricity > self.eccentricity[2]:
            eccentricity = self.eccentricity[2]
        return eccentricity

    def get_equivalent_circle_diameter(self, eccentricity):
        """Return equivalent circle diameter drawn from normal distribution in range [0, self.equivalent_circle_diameter[2]]

        Returns:
            float: Equivalent circle diameter [0, self.equivalent_circle_diameter[2]]
        """
        # calculate minimal cirlce diameter to fit the dot (minor_axis_length>=dot+1)
        min_minor = 1
        min_diameter = np.ceil(2*min_minor/(1-eccentricity**2)**(1/4))

        # Get equivalent_circle_diameter (Area of the ellipse) from distribution or given value
        if self.equivalent_circle_diameter[1] != 0:
            equivalent_circle_diameter = np.random.normal(self.equivalent_circle_diameter[0],
                                                          self.equivalent_circle_diameter[1])
        else:
            equivalent_circle_diameter = self.equivalent_circle_diameter[0]
        # Sanity checks (values from the normal distribution are not always [1,max)
        if equivalent_circle_diameter < min_diameter:
            equivalent_circle_diameter = min_diameter
        elif equivalent_circle_diameter > self.equivalent_circle_diameter[2]:
            equivalent_circle_diameter = self.equivalent_circle_diameter[2]
        return equivalent_circle_diameter

    def get_axis_length(self, eccentricity, equivalent_circle_diameter):
        """Calculate major and minor axis length from eccentricity and equivalent circle diameter

        Args:
            eccentricity (float): Eccentricity of the ellipse
            equivalent_circle_diameter (float): Equivalent circle diameter of the ellipse

        Returns:
            float, float: Major axis length and minor axis length
        """
        equivalent_circle_radius = equivalent_circle_diameter/2
        major_axis_length = equivalent_circle_radius/(1-eccentricity**2)**(1/4)
        # self.minor_axis_length = math.sqrt(equivalent_circle_radius**2*math.sqrt(1-eccentricity**2))
        minor_axis_length = equivalent_circle_radius**2/major_axis_length
        return minor_axis_length, major_axis_length

    def get_random_center(self, img_size):
        """Get random center of the ellipse

        Args:
            img_size (tuple): Shape of image

        Returns:
            tuple: Random center
        """
        center = (random.randint(0, img_size[0]-1), random.randint(0, img_size[1]-1))
        return center

    def get_rotation(self):
        """Get random rotation

        Returns:
            float: Random rotation of ellipse
        """
        if self.random_rotation:
            rotation = np.random.uniform(-np.pi, np.pi)
        return rotation

    def check_no_overlap(self, label, x, y):
        """Check if chosen position is OK and overlap is smaller than self.max_overlap

        Args:
            label (ndarray): Label image with objects already present
            x (ndarray): Array containing x coordinates of ellipse
            y (ndarray): Array containing y coordinates of ellipse

        Returns:
            bool: True, when object can be placed at chosen position
        """
        im_ellipse = np.zeros(label.shape)
        im_ellipse[x, y] = np.iinfo('uint16').max
        ellipse_pixels = np.sum(im_ellipse > 0)
        im_overlap = im_ellipse*label
        overlap_pixels = np.sum(im_overlap > 0)
        overlap_percent = overlap_pixels/ellipse_pixels*100
        if overlap_percent <= self.max_overlap:
            return True
        else:
            return False


class Noise():
    """Apply channel-wise noise
    """
    def __init__(self, noise, noise_params=None, targets=["label_train", "target"]):
        """Initialize the pipeline step to add noise to images

        Args:
            noise (str): Noise to add
            noise_params (list, optional): Parameters needed for noise type, e.g. mean and std for Gaussian noise. Defaults to None.
                                           std is multiplied by np.iinfo(images[target].dtype).max.
            targets (list, optional): Images noise is applied to. Defaults to ["label_train", "target"].
        """
        self.noise_params = noise_params
        self.targets = targets
        if noise == "gauss":
            self.noise = self.get_gaussian_noise
        else:
            assert ValueError(f"Noise {noise} not implemented")

    def draw(self, images):
        """Draw noise on images

        Args:
            images (dict of ndarray): Dict with ["label", "label_train", "target"] images

        Returns:
            dict of ndarray: Return the images with added noise
        """
        for target in self.targets:
            images[target] = np.clip(images[target]+self.noise(images[target].shape, np.iinfo(images[target].dtype).max),
                                     np.iinfo(images[target].dtype).min,
                                     np.iinfo(images[target].dtype).max).astype(images[target].dtype)

        return images

    def get_gaussian_noise(self, img_shape, image_max):
        """Get Gaussian noise with shape of provided image.
        Return noise is rounded towards int => this works only for integer dtypes.

        Args:
            img_shape (tuple of int): Shape of image noise is used for
            image_max (int): Max value of image dtype

        Returns:
            ndarray: Rounded gaussian noise with shape of img_shape
        """
        return np.round(np.random.normal(self.noise_params[0], image_max*self.noise_params[1], size=img_shape))


if __name__ == '__main__':
    # dataset 1
    dataset = Dataset(label_shape=(2048, 2048), target_shape=(2048, 2048, 3), label_train_shape=(2048, 2048),
                      target_dtype="uint8", num_images=512)
    ellipse1 = EllipseWithColor(eccentricity=(0.0, 0.0, 0.0), equivalent_circle_diameter=(40, 0, 40),
                                colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], max_overlap=0, random_rotation=True)
    noise = Noise("gauss", noise_params=[0, 0.22888], targets=["label_train"])  # 0.22888 will result in ~15000 for uint16 and 58 for uint8
    dataset.add_to_pipeline(ellipse1, 1000)
    dataset.add_to_pipeline(noise, 1)

    dataset.run_pipeline()
    dataset.save(Path(__file__).parent.joinpath("../datasets/tiling_strategy_benchmark_1/"))

    # dataset 2 (to create unpaired data)
    dataset = Dataset(label_shape=(2048, 2048), target_shape=(2048, 2048, 3), label_train_shape=(2048, 2048),
                      target_dtype="uint8", num_images=512)
    ellipse1 = EllipseWithColor(eccentricity=(0.0, 0.0, 0.0), equivalent_circle_diameter=(40, 0, 40),
                                colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], max_overlap=0, random_rotation=True)
    noise = Noise("gauss", noise_params=[0, 0.22888], targets=["label_train"])  # 0.22888 will result in ~15000 for uint16 and 58 for uint8
    dataset.add_to_pipeline(ellipse1, 1000)
    dataset.add_to_pipeline(noise, 1)

    dataset.run_pipeline()
    dataset.save(Path(__file__).parent.joinpath("../datasets/tiling_strategy_benchmark_2/"))