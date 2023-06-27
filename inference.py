import torch
import wandb
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from argparse import ArgumentParser


from lit_cyclegan import LitCycleGAN
from data_modules import get_data_module


def infer(model, device, results_folder, crop_size, overlap, keep_predicted, num_images):
    """Infer using Stitching Aware Inference with overlapping patches from both domains.
    Can be used for simple tiling, when the overlap is set to zero.

    Args:
        model (LitCycleGAN): Trained LitCycleGAN model
        device (torch.device): Device (GPU or CPU). GPU is highly recommended
        results_folder (Path): Path to the results folder.
        crop_size (int): Size of the cropping used
        overlap (int): Size of the overlap used
        keep_predicted (int): Amount of the overlapping area in percent kept, since it has already been predicted in previous patches.
        num_images (int): Number of images to infer on
    """
    print(f"Starting prediction for: crop_size: {crop_size}, overlap: {overlap}, keep_predicted: {keep_predicted}, num_images: {num_images}")
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
            coords = create_patch_grid(batch["A"].shape, crop_size, overlap)
            pred = batch["A"].detach().clone()
            save_img(pred, model.hparams.minmax_scale_A[0], model.hparams.minmax_scale_A[1],
                     results_folder.joinpath(batch["A_stem"][0]+"_gt_img_"+str(idx)+".png"))

            for coord in coords:
                p = model.gen_AB(pred[..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]])
                p = p*scaling_B[0]+scaling_B[1]  # scale patch

                # get the amount of pixels that stay
                x_stay = int(np.round((coord["x_new_start"]-coord["x_start"])*keep_predicted/100))
                y_stay = int(np.round((coord["y_new_start"]-coord["y_start"])*keep_predicted/100))
                # remove pixels that stay from prediction and add prediction to final pred          
                p = p[..., x_stay:, y_stay:]
                pred[..., (coord["x_start"]+x_stay):coord["x_end"], (coord["y_start"]+y_stay):coord["y_end"]] = p

                gt_crop_path = results_folder.joinpath(batch["A_stem"][0]+"_gt_img_"+str(idx)+"_crop_"+str(coord["x_start"])+"_"+str(coord["y_start"])+".png")
                pred_crop_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_img_"+str(idx)+"_crop_"+str(coord["x_start"])+"_"+str(coord["y_start"])+".png")
                save_img(batch["A"][..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]], model.hparams.minmax_scale_A[0],
                         model.hparams.minmax_scale_A[1], gt_crop_path)                
                save_img(pred[..., coord["x_start"]:coord["x_end"], coord["y_start"]:coord["y_end"]], model.hparams.minmax_scale_B[0],
                         model.hparams.minmax_scale_B[1], pred_crop_path)

            pred_stitch_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_stitch.png")
            save_img(pred, model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_stitch_path)

            pred = pred.cpu().numpy()
            grid = np.full_like(pred, model.hparams.minmax_scale_B[0])  # initialize array with min value
            grid_axis_0_overlap_start = np.unique([coord["x_start"] for coord in coords])[1:]  # remove first entry (0)
            grid_axis_1_overlap_start = np.unique([coord["y_start"] for coord in coords])[1:]  # remove first entry (0)
            grid_axis_0_overlap_end = np.unique([coord["x_end"] for coord in coords])[:-1]  # remove last entry (pred.shape[-2])
            grid_axis_1_overlap_end = np.unique([coord["y_end"] for coord in coords])[:-1]  # remove last entry (pred.shape[-1])

            grid_value_start = model.hparams.minmax_scale_B[0]+(model.hparams.minmax_scale_B[1]-model.hparams.minmax_scale_B[0])/2
            grid_value_end = model.hparams.minmax_scale_B[1]
            grid[...,grid_axis_0_overlap_start,:] = grid_value_start
            grid[...,grid_axis_1_overlap_start] = grid_value_start
            grid[...,grid_axis_0_overlap_end,:] = grid_value_end
            grid[...,grid_axis_1_overlap_end] = grid_value_end

            pred_with_grid = pred
            pred_with_grid[grid==grid_value_end] = grid_value_end
            pred_with_grid[grid==grid_value_start] = grid_value_start

            pred_stitch_grid_path = results_folder.joinpath(batch["A_stem"][0]+"_pred_stitch_grid.png")  
            save_img(pred_with_grid, model.hparams.minmax_scale_B[0], model.hparams.minmax_scale_B[1], pred_stitch_grid_path)
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
    Image.fromarray(img).save(save_path)


def create_patch_grid(img_shape, crop_size, overlap):
    """ Create coordinates for the prediction of a large image with overlapping patches and returns a list
    of dict with coords["x_start", "x_end", "x_new_start", "y_start", "y_end", "y_new_start"]

    Args:
        img_shape (list): shape of the image predictions are performed on
        crop_size (int): size of the patches
        overlap (int): overlap between the crops of the patches
    """
    coords = []
    img_shape = img_shape[-2:]  # only keep last two dimensions
    for x in np.arange(0, img_shape[0]-overlap, crop_size-overlap):
        if x == 0:  # first row
            x_new_start = 0
            x_start = x
            x_end = x+crop_size
        elif x+crop_size > img_shape[0]:  # last row when more overlap is needed
            x_new_start = x+overlap 
            x_start = img_shape[0]-crop_size
            x_end = img_shape[0]
        else:  # all other rows
            x_new_start = x+overlap
            x_start = x
            x_end = x+crop_size
        for y in np.arange(0, img_shape[1]-overlap, crop_size-overlap):
            if y == 0:
                y_new_start = 0
                y_start = y
                y_end = y+crop_size
            elif y+crop_size > img_shape[1]:  # last row when more overlap is needed
                y_new_start = y+overlap
                y_start = img_shape[1]-crop_size
                y_end = img_shape[1]
            else:
                y_new_start = y+overlap
                y_start = y
                y_end = y+crop_size
            coords.append({"x_start": x_start, "x_end": x_end, "x_new_start": x_new_start, "y_start": y_start, "y_end": y_end, "y_new_start": y_new_start})

    return coords


def start_inference(args):
    # resume previous project
    wandb.login()

    # restore checkpoint by downloading .ckpt file
    api = wandb.Api()
    artifact = api.artifact(args.ckpt_artifact, type='model')

    combinations = np.array(np.meshgrid(args.crop_size, args.overlap, args.keep_predicted, [args.num_images])).T.reshape(-1, 4)

    artifact_dir = artifact.download(root=Path(__file__).parent.joinpath("results", artifact.project, artifact.name))
    # load model from .ckpt file
    model = LitCycleGAN.load_from_checkpoint(Path(artifact_dir).joinpath("model.ckpt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for combination in combinations:
        results_folder = Path(__file__).parent.joinpath("results", artifact.project, artifact.name, "crop_"+str(combination[0]),
                                                        "overlap_"+str(combination[1]), "keep_predicted_"+str(combination[2]))
        results_folder.mkdir(exist_ok=True, parents=True)

        infer(model, device, results_folder, crop_size=combination[0], overlap=combination[1],
              keep_predicted=combination[2], num_images=combination[3])
    wandb.finish()

    # delete wandb cached artifacts
    wandb_cache_folder = Path(wandb.env.get_cache_dir())
    shutil.rmtree(wandb_cache_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ckpt_artifact', '-ca', default="moritzb/color_RGB_inpainting/model-2ikrfkzt:v0",
                        type=str, help="WandB checkpoint path.")
    parser.add_argument('--crop_size', '-cs', nargs='+', default=[512], type=int, help="Crop size of the patches")
    parser.add_argument('--overlap', '-o', nargs='+', default=[64], type=int, help='Overlap between patches')
    parser.add_argument('--keep_predicted', '-kp', nargs='+', default=[50], type=int,
                        help="Amount of the overlapping area in percent kept, since it has already been predicted.")
    parser.add_argument('--num_images', '-ni', default=10, type=int, help='Number of images to predict')
    args = parser.parse_args()
    start_inference(args)
