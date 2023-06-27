from PIL import Image
import numpy as np
from pathlib import Path
from skimage.morphology import area_opening, disk, binary_dilation
from skimage.filters import sobel
from skimage.measure import label, regionprops_table
import pandas as pd
from skimage.filters.rank import mean
from multiprocessing import Pool
import os
from PIL import ImageDraw
from skimage.segmentation import flood
from argparse import ArgumentParser


def save_img(img, path):
    Image.fromarray(img).save(path)


def get_real_fake_pairs(real_folder, fake_folder):
    real_folder = Path(real_folder)
    fake_folder = Path(fake_folder)

    fake_paths = list(fake_folder.glob('*_pred_stitch.png'))
    real_paths = list(real_folder.glob('*.png'))

    pairs = []
    for fake_path in fake_paths:
        id = fake_path.stem.split('_')[0]
        for real_path in real_paths:
            if id == real_path.stem:
                pairs.append({"real": real_path, "fake": fake_path})
    assert len(pairs), f"No pairs found for {real_folder} and {fake_folder}"
    return pairs


def get_hull(img_real):
    return sobel(img_real) > 0


def get_circle_colors(img_real, img_fake):
    """Find out, whether a color (R,G,B) is present inside a circle of img_real in img_fake.
    If the color is present, fill the whole circle in the corresponding color channel of a new image circle_colors

    Args:
        img_real (np.array): Instance image of circles
        img_fake (np.array): Generated RGB image of circles

    Returns:
        np.array: Image with filled circles in each color channel, where colors are present in img_fake
    """
    circle_colors = np.zeros_like(img_fake)
    for i in range(img_fake.shape[-1]):
        img_fake_channel = img_fake[..., i]

        # remove background
        img_fake_channel = img_fake_channel*(img_real > 0)

        # smooth
        img_fake_channel = mean(img_fake_channel, selem=disk(2))

        # circles with values <= 60 will not be accounted
        img_fake_channel = img_fake_channel > 60

        # remove dark spots smaller than 30 px in circles
        img_fake_channel = ~area_opening(~img_fake_channel, area_threshold=30)  # remove dark spots smaller than 30px

        # separate merged cells (remove hull from img_fake_channel)
        img_fake_channel = np.logical_xor(img_fake_channel, np.logical_and(img_fake_channel, binary_dilation(sobel(img_real) > 0)))

        # remove bright spots smaller than 30 px (pixel in circles of other colors)
        img_fake_channel = area_opening(img_fake_channel, area_threshold=30)  # remove bright spots smaller than 30px

        # fill whole circle, when part of it is present
        seed_points = pd.DataFrame(regionprops_table(label(img_fake_channel, connectivity=img_fake_channel.ndim),
                                                     properties=["label", "centroid", "coords"]))
        circles_flood = np.zeros_like(img_fake_channel)
        for idx, seed_point in seed_points.iterrows():
            # ensure dot is no hallucination and not in background of img_real
            if img_real[int(seed_point["centroid-0"]), int(seed_point["centroid-1"])] > 0:
                circles_flood = circles_flood + flood(img_real, (int(seed_point["centroid-0"]), int(seed_point["centroid-1"])))

        circle_colors[..., i] = circles_flood

    return circle_colors


def get_colors_per_circle(img_real, circle_colors):
    """Extract number of colors for each circle in img_real.

    Args:
        img_real (np.array): Instance image of circles
        circle_colors (np.array): Circles present in each color channel

    Returns:
        DataFrame: Statistics for the image
    """

    # merge color channels
    circle_colors = circle_colors[..., 0] + circle_colors[..., 1] + circle_colors[..., 2]

    circles = pd.DataFrame(regionprops_table(label(img_real, connectivity=img_real.ndim), properties=["label", "area", "centroid", "coords"]))
    circles["colors_in_circle"] = 0
    for idx, circle in circles.iterrows():
        circles.loc[idx, "colors_in_circle"] = circle_colors[int(circle["centroid-0"]), int(circle["centroid-1"])]

    return circles


def get_statistics(results):
    """Combine individual results of each image to final statistics

    Args:
        results (DataFrame): Individual results fo reach circle in all images

    Returns:
        DataFrame: Final results with colors per circle, number of occurrences and overall percentage. -1 is for total number of circles.
    """
    statistics = pd.DataFrame(columns=["n_colors_per_circle", "n_circles"])
    statistics = statistics.append(pd.DataFrame([[-1,0], [0,0], [1,0],[2,0],[3,0]], columns=["n_colors_per_circle", "n_circles"]),
                                   ignore_index=True)  # initial populate
    for result in results:
        statistics.loc[statistics["n_colors_per_circle"] == -1, "n_circles"] = statistics[statistics["n_colors_per_circle"] == -1]["n_circles"] + len(result)
        for n_dots in np.unique(result["colors_in_circle"]):
            if n_dots in statistics["n_colors_per_circle"].values:
                statistics.loc[statistics["n_colors_per_circle"]==n_dots, "n_circles"] = statistics[statistics["n_colors_per_circle"]==n_dots]["n_circles"] + sum(result['colors_in_circle']==n_dots)
            else:
                statistics = statistics.append(pd.DataFrame([[n_dots,sum(result['colors_in_circle']==n_dots)]], columns=["n_colors_per_circle", "n_circles"]), ignore_index=True)

    statistics.reset_index(inplace=True, drop=True)
    percent = statistics["n_circles"].values/statistics[statistics["n_colors_per_circle"]==-1]["n_circles"].values*100
    percent = pd.DataFrame(percent, columns=["n_circles_percent"])
    statistics = pd.concat([statistics, percent], axis=1)
    statistics = statistics.astype({"n_colors_per_circle": int, "n_circles": int, "n_circles_percent": float})
    return statistics


def create_results_overlay(img_fake, img_fake_path, results):
    img_fake_pil = Image.fromarray(img_fake)
    img_fake_draw = ImageDraw.Draw(img_fake_pil)
    for idx, result in results.iterrows():
        if result["colors_in_circle"] != 1:
            img_fake_draw.rectangle([(result["centroid-1"]+23, result["centroid-0"]+23), (result["centroid-1"]-23, result["centroid-0"]-23)], outline="red", width=4)
    img_fake_pil.save(img_fake_path.parent / (img_fake_path.stem + "_overlay.png"))


def get_image_results(img_pair, crop):
    print(f"Processing image {img_pair['fake'].name}...")
    img_real = np.array(Image.open(img_pair["real"]))
    img_fake = np.array(Image.open(img_pair["fake"]))

    circle_colors = get_circle_colors(img_real, img_fake)
    results = get_colors_per_circle(img_real[crop[0]:crop[1], crop[2]:crop[3], ...], circle_colors[crop[0]:crop[1], crop[2]:crop[3], ...])
    create_results_overlay(img_fake, img_pair["fake"], results)
    return results


def eval(crop, img_real_folder, img_fake_folder):
    img_pairs = get_real_fake_pairs(img_real_folder, img_fake_folder)

    pool = Pool(processes=np.min([len(img_pairs), 40, os.cpu_count()]))
    args = [[img_pair, crop] for img_pair in img_pairs]
    results = pool.starmap(get_image_results, args)

    statistics = get_statistics(results)
    print(statistics.to_string(index=False, formatters={"n_circles_percent": "{:.2f}".format}))
    return statistics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--crop", "-cp", type=int, nargs='+', default=[0, 2048, 0, 2048],
                        help="Apply cropping to omit counting errors at border patches. [x_start, x_end, y_start, y_end]. When no cropping is desired set to real image resolution.")
    parser.add_argument("--img_real", "-ir", type=str, help="Path to groundtruth images folder from domain A")
    parser.add_argument("--img_fake", "-if", type=str, help="Path to synthetic images folder from domain B created by the GAN")
    args = parser.parse_args()

    eval(args.crop, args.img_real, args.img_fake)
