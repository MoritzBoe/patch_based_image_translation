import torch
import wandb
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import tifffile as tiff

from lit_cyclegan import LitCycleGAN
from data_modules import get_data_module


def infer(model, device, results_folder, crop_size, overlap, keep_predicted, num_images, save_crops=False):
    """Infer using Stitching Aware Inference with overlapping patches from both domains.
    Can be used for simple tiling, when the overlap is set to zero.

    Args:
        model (LitCycleGAN): Trained LitCycleGAN model
        device (torch.device): Device (GPU or CPU). GPU is highly recommended
        results_folder (Path): Path to the results folder.
        crop_size (int): Size of the cropping used
        overlap (int): Size of the overlap in percent used
        keep_predicted (int): Amount of the overlapping area in percent kept, since it has already been predicted in previous patches.
        num_images (int): Number of images to infer on
        save_crops (bool): Save intermediate crops
    """
    print(f"Starting prediction for: crop_size: {crop_size}, overlap: {overlap}, keep_predicted: {keep_predicted}, num_images: {num_images}")
    data_module = get_data_module(model.hparams.data_module, batch_size=1, num_workers=0, crop_size=None,
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
            coords = create_patch_grid(batch["A"].shape, crop_size, overlap)
            pred = batch["A"].detach().clone()
            save_img(pred, model.hparams.minmax_scale_A[0], model.hparams.minmax_scale_A[1], results_folder.joinpath(batch["A_stem"][0]+"_gt_img_"+str(idx)+".png"))

            for coord in coords:
                p = model.gen_AB(pred[..., coord["z_start"]:coord["z_end"], coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]])
                p = p*scaling_B[0]+scaling_B[1]  # scale patch

                # get the amount of pixels that stay
                z_stay = int(np.round((coord["z_new_start"]-coord["z_start"])*keep_predicted/100))
                x_stay = int(np.round((coord["x_new_start"]-coord["x_start"])*keep_predicted/100))
                y_stay = int(np.round((coord["y_new_start"]-coord["y_start"])*keep_predicted/100))
                # remove pixels that stay from prediction and add prediction to final pred
                p = p[..., z_stay:, x_stay:, y_stay:]
                pred[..., (coord["z_start"]+z_stay):coord["z_end"], (coord["x_start"]+x_stay):coord["x_end"], (coord["y_start"]+y_stay):coord["y_end"]] = p

                gt_crop_path = results_folder.joinpath(batch["A_stem"][0]+"_gt_img_"+str(idx)+"_crop_"+str(coord["z_start"])+"_"+str(coord["x_start"])+"_"+str(coord["y_start"])+".png")
                pred_crop_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_img_"+str(idx)+"_crop_"+str(coord["z_start"])+"_"+str(coord["x_start"])+"_"+str(coord["y_start"])+".png")
                if save_crops:
                    save_img(batch["A"][..., coord["z_start"]:coord["z_end"], coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]],
                            model.hparams.minmax_scale_A[0], model.hparams.minmax_scale_A[1], gt_crop_path)                
                    save_img(pred[...,  coord["z_start"]:coord["z_end"], coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]],
                            model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_crop_path)

            pred_stitch_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_stitch.png")
            save_img(pred, model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_stitch_path)

            if idx+1 == num_images:
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
    tiff.imsave(save_path, img, imagej=True)


def create_patch_grid(img_shape, crop_size, overlap):
    """ Create coordinates for the prediction of a large image with overlapping patches and returns a list
    of dict with coords["x_start", "x_end", "x_new_start", "y_start", "y_end", "y_new_start", "z_start", "z_end", "z_new_start"]

    Args:
        img_shape (list): shape of the image predictions are performed on
        crop_size (int): size of the patches
        overlap (int): overlap between the crops of the patches in percent
    """
    overlap = [round(crop_size[-3]*overlap/100), round(crop_size[-2]*overlap/100), round(crop_size[-1]*overlap/100)]
    coords = []
    img_shape = img_shape[-3:]  # only keep last three dimensions
    for x in np.arange(0, img_shape[1]-overlap[1], crop_size[1]-overlap[1]):
        if x == 0:  # first row
            x_new_start = 0
            x_start = x
            x_end = x+crop_size[1]
        elif x+crop_size[1] > img_shape[1]:  # last row when more overlap is needed
            x_new_start = x+overlap[1]          
            x_start = img_shape[1]-crop_size[1]
            x_end = img_shape[1]
        else:  # all other rows
            x_new_start = x+overlap[1]
            x_start = x
            x_end = x+crop_size[1]
        for y in np.arange(0, img_shape[2]-overlap[2], crop_size[2]-overlap[2]):
            if y == 0:
                y_new_start = 0
                y_start = y
                y_end = y+crop_size[2]
            elif y+crop_size[2] > img_shape[2]:  # last row when more overlap is needed
                y_new_start = y+overlap[2]
                y_start = img_shape[2]-crop_size[2]
                y_end = img_shape[2]
            else:
                y_new_start = y+overlap[2]
                y_start = y
                y_end = y+crop_size[2]
            for z in np.arange(0, img_shape[0]-overlap[0], crop_size[0]-overlap[0]):
                if z == 0:
                    z_new_start = 0
                    z_start = z
                    z_end = z+crop_size[0]
                elif z+crop_size[0] > img_shape[0]:  # last row when more overlap is needed
                    z_new_start = z+overlap[0]
                    z_start = img_shape[0]-crop_size[0]
                    z_end = img_shape[0]
                else:
                    z_new_start = z+overlap[0]
                    z_start = z
                    z_end = z+crop_size[0]
                coords.append({"x_start": x_start, "x_end": x_end, "x_new_start": x_new_start, "y_start": y_start, "y_end": y_end, "y_new_start": y_new_start,
                               "z_start": z_start, "z_end": z_end, "z_new_start": z_new_start})
    return coords


def start_inference(args):
    # resume previous project
    wandb.login()

    # restore checkpoint by downloading .ckpt file
    api = wandb.Api()
    artifact = api.artifact(args.ckpt_artifact, type='model')


    artifact_dir = artifact.download(root=Path(__file__).parent.joinpath("results", artifact.project, artifact.name))
    # load model from .ckpt file
    model = LitCycleGAN.load_from_checkpoint(Path(artifact_dir).joinpath("model.ckpt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # preprocess crop_size
    if len(args.crop_size) == 1:
        args.crop_size = [args.crop_size[0], args.crop_size[0], args.crop_size[0]]
    crop_size_str = ""
    crop_size_str = str().join([crop_size_str+"_"+str(s) for s in args.crop_size])

    results_folder = Path(__file__).parent.joinpath("results", artifact.project, artifact.name, "crop"+str(crop_size_str),
                                                    "overlap_"+str(args.overlap), "keep_predicted_"+str(args.keep_predicted))
    results_folder.mkdir(exist_ok=True, parents=True)

    infer(model, device, results_folder, crop_size=args.crop_size, overlap=args.overlap,
            keep_predicted=args.keep_predicted, num_images=args.num_images, save_crops=args.save_crops)
    wandb.finish()

    # delete wandb cached artifacts
    wandb_cache_folder = Path(wandb.env.get_cache_dir())
    shutil.rmtree(wandb_cache_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_artifact', '-ca', default="moritzb/color_RGB_inpainting/model-2ikrfkzt:v0",
                        type=str, help="WandB checkpoint path.")    
    parser.add_argument('--crop_size', '-cs', type=int, default=256, nargs='*', help="Size of the crops. When multiple values are passed, dims have to be passed as Z X Y.")
    parser.add_argument('--overlap', '-o', default=25, type=int, help='Overlap between patches in percent')
    parser.add_argument('--keep_predicted', '-kp', default=50, type=int,
                        help="Amount of the overlapping area in percent kept, since it has already been predicted.")
    parser.add_argument('--num_images', '-ni', default=1, type=int, help='Number of images to predict')
    parser.add_argument('--save_crops', '-sc', default=False, action='store_true', help='Save crops of single predictions')
    args = parser.parse_args()
    start_inference(args)
