import torch
import wandb
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from argparse import ArgumentParser

from lit_cyclegan import LitCycleGAN
from data_modules import get_data_module

""" Perform weighted inference according to Bel et. al. 2019
Stain-Transforming Cycle-Consistent Generative Adversarial Networks for Improved Segmentation of Renal Histopathology
"""


def infer(ckpt_artifact, patch_size, crop_size, shift, num_images):
    """Weighted inference with overlapping final patches

    Args:
        ckpt_artifact (str): WandB to checkpoint artifact
        patch_size (int): Size of patches
        crop_size (int): Size of crops after prediction
        shit (int): Shift between predicted images
        num_images (int): Number of images to infer on
    """
    # resume previous project
    wandb.login()

    # restore checkpoint by downloading .ckpt file
    api = wandb.Api()
    artifact = api.artifact(ckpt_artifact, type='model')

    results_folder = Path(__file__).parent.joinpath("results", artifact.project, artifact.name, "weighted", "patch_size_"+str(patch_size), "crop_"+str(crop_size), "shift_"+str(shift))
    results_folder.mkdir(exist_ok=True, parents=True)
    artifact_dir = artifact.download(root=results_folder.parent)

    # load model from .ckpt file
    model = LitCycleGAN.load_from_checkpoint(Path(artifact_dir).joinpath("model.ckpt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    data_module = get_data_module(model.hparams.data_module, batch_size=1, num_workers=0, crop_size=crop_size,
                                  minmax_scale_A=model.hparams.minmax_scale_A, minmax_scale_B=model.hparams.minmax_scale_B)
    dataloader = data_module.test_dataloader()

    # get scaling
    scaling_B = [(model.hparams.minmax_scale_B[1]-model.hparams.minmax_scale_B[0])/2, 
                 model.hparams.minmax_scale_B[0]-(-1)*((model.hparams.minmax_scale_B[1]-model.hparams.minmax_scale_B[0])/2)]
    model.gen_AB.eval()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Processing image {idx+1} of {num_images}")
            batch["A"] = batch["A"].to(device)
            coords = create_patch_grid_weighted(batch["A"].shape, patch_size, shift)
            img_shape = batch["A"].shape
            pred = np.zeros(img_shape)
            weights = np.zeros(img_shape[-2:])
            save_img(batch["A"], model.hparams.minmax_scale_A[0], model.hparams.minmax_scale_A[1], results_folder.joinpath(batch["A_stem"][0]+"_gt_img_"+str(idx)+".png"))

            for coord in coords:
                # predict and scale
                p = model.gen_AB(batch["A"][..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]])*scaling_B[0]+scaling_B[1]
                w_map = get_weight_map(coord, img_shape, crop_size)

                weights[coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]] = weights[..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]] + w_map
                pred[..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]] = pred[..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]] + w_map*p.detach().cpu().numpy().astype(np.float64)

                gt_crop_path = results_folder.joinpath(batch["A_stem"][0]+"_gt_img_"+str(idx)+"_crop_"+str(coord["x_start"])+"_"+str(coord["y_start"])+".png")
                pred_crop_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_img_"+str(idx)+"_crop_"+str(coord["x_start"])+"_"+str(coord["y_start"])+".png")
                save_img(batch["A"][..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]],
                         model.hparams.minmax_scale_A[0], model.hparams.minmax_scale_A[1], gt_crop_path)        
                save_img(p, model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_crop_path)

            # divide prediction by weights
            pred = pred/weights

            pred_stitch_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_stitch.png")
            save_img(pred, model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_stitch_path)

            if idx+1 == num_images:
                wandb.finish()
                break


def save_img(img, scale_min, scale_max, save_path):
    # convert img to numpy
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.squeeze()
    # move color channel to the back if necessary. Fails when img.shape = (3,X,3) and the last channel already is the color channel!!
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.moveaxis(img, 0, -1)
    # scale image and convert to uint8
    img = np.round((img-scale_min)*255/(scale_max-scale_min)).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def create_patch_grid_weighted(img_shape, patch_size, shift):
    """ Create coordinates for the prediction of a large image with overlapping patches and returns a list of dict with coords["x_start", "x_end", "y_start", "y_end"]

    Args:
        img_shape ([type]): shape of the image predictions are performed on
        patch_size ([type]): size of the patches
        shift ([type]): shift between the crops of the patches
    """
    coords = []
    img_shape = img_shape[-2:]  # only keep last two dimensions
    for x in np.arange(0, img_shape[0]-patch_size+1, shift):
        x_start = x
        x_end = x+patch_size
        for y in np.arange(0, img_shape[1]-patch_size+1, shift):
            y_start = y
            y_end = y+patch_size
            coords.append({"x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end})
    return coords


def get_weight_map(coord, img_shape, crop_size):
    """ Cropping is implicitly done by setting the pixels outside the crop to zero.

    Args:
        coord (dict): Dict with position of the patch x_start, x_end, y_start, y_end
        img_shape (array): Size of the final image in x and y
        crop_size (int): Size the input image is cropped finally

    Returns:
        w_map
    """
    crop_remove_len = int(crop_size/2)
    w_map = np.zeros([coord["x_end"]-coord["x_start"], coord["y_end"]-coord["y_start"]])
    # set middle crop to one
    w_map[crop_remove_len:-crop_remove_len, crop_remove_len:-crop_remove_len] = 1
    # handle sides
    if coord["x_start"] == 0:
        w_map[0:crop_remove_len, crop_remove_len:-crop_remove_len] = 1
    if coord["y_start"] == 0:
        w_map[crop_remove_len:-crop_remove_len, 0:crop_remove_len] = 1
    if coord["x_end"] == img_shape[-2]:
        w_map[-crop_remove_len:, crop_remove_len:-crop_remove_len] = 1
    if coord["y_end"] == img_shape[-1]:
        w_map[crop_remove_len:-crop_remove_len, -crop_remove_len:] = 1
    # handle corners
    if coord["x_start"] == 0 and coord["y_start"] == 0:
        w_map[0:crop_remove_len, 0:crop_remove_len] = 1
    if coord["x_start"] == 0 and coord["y_end"] == img_shape[-1]:
        w_map[0:crop_remove_len, -crop_remove_len:] = 1
    if coord["y_start"] == 0 and coord["x_end"] == img_shape[-2]:
        w_map[-crop_remove_len:, 0:crop_remove_len] = 1
    if coord["x_end"] == img_shape[-2] and coord["y_end"] == img_shape[-1]:
        w_map[-crop_remove_len:, -crop_remove_len:] = 1

    w_map = distance_transform_edt(w_map)
    w_map = w_map/w_map.max()
    return w_map


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_artifact', '-ca', default="moritzb/color_RGB_inpainting/model-2ikrfkzt:v0",
                        type=str, help="WandB checkpoint path.")
    parser.add_argument('--patch_size', '-ps', default=512, type=int, help="Size of patches")
    parser.add_argument('--crop_size', '-cs', default=512/2, type=int, help="Size of crops after prediction")
    parser.add_argument('--shift', '-sh', default=512/2/2, type=int, help="Shift for next prediction")
    parser.add_argument('--num_images', '-ni', default=1, type=int, help='Number of images to predict')
    args = parser.parse_args()

    infer(ckpt_artifact=args.ckpt_artifact, patch_size=args.patch_size, crop_size=args.crop_size,
          shift=args.shift, num_images=args.num_images)
