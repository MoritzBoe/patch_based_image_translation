import torch.nn as nn
import torch
from torch import Tensor
import numpy as np
from scipy.ndimage import distance_transform_edt


def disc_loss(prediction, target_is_real, loss_function):
    """Calculates the loss for the discriminator output. Is used to optimize the discriminator and to optimize
    the generator performance (feed fake image to discriminator and see if it is able to classify it as fake).

    Args:
        prediction (tensor): output from the discriminator
        target_is_real (bool): true if the input to the discriminator was a real image, false if the input was a fake one
        gan_loss ([type]): loss function used for the discriminator output on the generated or real images
        device ([type]): device of the performed calculations

    Raises:
        NotImplementedError: Raised if the loss requested is not implemented

    Returns:
        loss: (float) loss for the generator network with respect to the discriminator
    """
    if target_is_real:
        target = torch.tensor(1, dtype=prediction.dtype)
    else:
        target = torch.tensor(0, dtype=prediction.dtype)
    target = target.expand_as(prediction).to(prediction.device)
    if loss_function == "MSE":
        loss = nn.MSELoss()
    elif loss_function == "L1":
        loss = nn.L1Loss()
    elif loss_function == "BCE":
        loss = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"gan_loss unknown. Value passed: {loss_function}")
    return loss(prediction, target)


class DirectionalWeightedMSELoss(nn.Module):
    """Loss for images consisting of both domains A and B transferred by the generator AB to domain B
    loss(gen_AB(real_inpainting_A))

    Loss is only calculated for regions from domain B in real_inpainting_A. For 2D images this can be
    top, left or top and left. For 3D this can be top, left, front, and their combinations.

    This loss is the same as InpaintingLoss, but MSE is used as a loss metric and pixels are weighted
    according to their distance to the transition from domain A and B in real_inpainting_A.
    """
    def __init__(self, crop_size, network_G):
        """Initialization of the loss to set used metric (loss), crop size and image dimension.

        Args:
            crop_size (int or list[int]): Size of the crops processed during training. Int => same for each dim, ZXY otherwise.
            network_G (str): Name of the generator. Used to extract whether 3D images are processed.
        """
        super(DirectionalWeightedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="none")
        if len(crop_size) == 1:
            self.crop_size = [crop_size[0], crop_size[0], crop_size[0]]
        else:
            self.crop_size = crop_size
        if "3D" in network_G:
            self.dim = 3
        else:
            self.dim = 2
        self.weight_maps = self.get_weight_maps()  # inpainting 0=>top, 1=>left, 2 =>top/left

    def get_weight_maps(self):
        w_maps = []
        if self.dim == 2:
            # 0 (top)
            w_map = np.ones([self.crop_size[0]+1, self.crop_size[1]])
            w_map[-1:,:] = 0
            w_maps.append(distance_transform_edt(w_map)[:-1,:])
            # 1 (left)
            w_map = np.ones([self.crop_size[0], self.crop_size[1]+1])
            w_map[:,-1] = 0
            w_maps.append(distance_transform_edt(w_map)[:,:-1])
            # 2 (topleft)
            w_map = np.ones([self.crop_size[0]+1, self.crop_size[1]+1])
            w_map[-1,-1] = 0
            w_maps.append(distance_transform_edt(w_map)[:-1,:-1])
        if self.dim == 3:  # only use weightmap from left, top, front instead of multiple weight maps
            w_map = np.ones([self.crop_size[0]+1, self.crop_size[1]+1, self.crop_size[2]+1])
            w_map[-1,-1,-1] = 0
            w_maps.append(distance_transform_edt(w_map)[:-1,:-1,:-1])
        return w_maps

    def forward(self, input, target, inpainting_pixel, inpainting_mode, **kwargs):
        # for an easy/short implementation, parts not needed could be set to zero in weights.
        # since overlapping areas are small, it is better to crop the necessary parts before computing loss
        # ToDo: Speed comparison. If speed is equal => use short implementation
        
        # inpainting_pixel [Z, X, Y]
        if self.dim == 2:
            if (inpainting_mode == [True, False]).all():  # top [HxW]
                input = input[..., 0:inpainting_pixel[1], :]
                target = target[..., 0:inpainting_pixel[1], :]
                weights = self.weight_maps[0][0:inpainting_pixel[1], :]
                weights = weights/weights.max()
                weights = np.concatenate([weights[np.newaxis,...]]*input.shape[-3], axis=0)
                weights = np.concatenate([weights[np.newaxis,...]]*input.shape[-4], axis=0)
                weights = torch.tensor(weights, device=target.device)
                loss = (weights*self.loss(input, target)).mean()
            elif (inpainting_mode == [False, True]).all():  # left
                input = input[..., 0:inpainting_pixel[2]]
                target = target[..., 0:inpainting_pixel[2]]
                weights = self.weight_maps[1][:, 0:inpainting_pixel[2]]
                weights = weights/weights.max()
                weights = np.concatenate([weights[np.newaxis,...]]*input.shape[-3], axis=0)
                weights = np.concatenate([weights[np.newaxis,...]]*input.shape[-4], axis=0)
                weights = torch.tensor(weights, device=target.device)
                loss = (weights*self.loss(input, target)).mean()
            elif (inpainting_mode == [True, True]).all():  # top+left
                # left_part
                input_left = input[..., 0:inpainting_pixel[2]]
                target_left = target[..., 0:inpainting_pixel[2]]
                weights_left = self.weight_maps[2][:, 0:inpainting_pixel[2]]
                weights_left = weights_left/weights_left.max()
                weights_left = np.concatenate([weights_left[np.newaxis,...]]*input.shape[-3], axis=0)
                weights_left = np.concatenate([weights_left[np.newaxis,...]]*input.shape[-4], axis=0)
                weights_left = torch.tensor(weights_left, device=target.device)
                loss_left = (weights_left*self.loss(input_left, target_left)).mean()
                # top_right part
                input_tr = input[..., :inpainting_pixel[1], inpainting_pixel[2]:]
                target_tr = target[..., :inpainting_pixel[1], inpainting_pixel[2]:]
                # cut left part, normalize and afterwards cut top part this will result in matching weights for the top part of mode 2 (top and left)
                weights_tr = self.weight_maps[2][0:inpainting_pixel[1], :]
                weights_tr = weights_tr/weights_tr.max()  
                weights_tr = weights_tr[:, inpainting_pixel[2]:]
                weights_tr = np.concatenate([weights_tr[np.newaxis,...]]*input.shape[-3], axis=0)
                weights_tr = np.concatenate([weights_tr[np.newaxis,...]]*input.shape[-4], axis=0)
                weights_tr = torch.tensor(weights_tr, device=target.device)
                loss_tr = (weights_tr*self.loss(input_tr, target_tr)).mean()
                loss = loss_left + loss_tr
        elif self.dim == 3:
            batch_size = input.shape[0]
            color_channels = input.shape[1]
            size_z = self.weight_maps[0].shape[0]
            size_x = self.weight_maps[0].shape[1]
            size_y = self.weight_maps[0].shape[2]
            if (inpainting_mode == [True, False, False]).all():  # Front [ZxHxW]             inpainting_size
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()
            elif (inpainting_mode == [False, True, False]).all():  # top
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                       np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()
            elif (inpainting_mode == [False, False, True]).all():  # left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                       np.r_[0:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()
            elif (inpainting_mode == [True, True, False]).all():  # front, top
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()  # front
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()  # top without front
                indices = np.concatenate((indices, indices_2), axis=0)
            elif (inpainting_mode == [True, False, True]).all():  # front, left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()  # front
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[0:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()  # left without front
                indices = np.concatenate((indices, indices_2), axis=0)
            elif (inpainting_mode == [False, True, True]).all():  # top, left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                       np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()  # top
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                         np.r_[inpainting_pixel[1]:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()  # left without top
                indices = np.concatenate((indices, indices_2), axis=0)
            elif (inpainting_mode == [True, True, True]).all():  # front, top, left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()  # front
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()  # top without front
                indices_3 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[inpainting_pixel[1]:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()  # left without front and top
                indices = np.concatenate((indices, indices_2, indices_3), axis=0)

            weights = np.concatenate([self.weight_maps[0][np.newaxis,...]]*input.shape[1], axis=0)
            weights = np.concatenate([weights[np.newaxis,...]]*input.shape[0], axis=0)
            weights = weights.flatten()[indices]
            weights = weights/weights.max()

            weights = torch.tensor(weights, device=target.device)
            loss = (weights*self.loss(input.flatten()[indices], target.flatten()[indices])).mean()
        return loss


class InpaintingLoss(nn.Module):
    """Loss for images consisting of both domains A and B transferred by the generator AB to domain B
    loss(gen_AB(real_inpainting_A))

    Loss is only calculated for regions from domain B in real_inpainting_A. For 2D images this can be
    top, left or top and left. For 3D this can be top, left, front, and their combinations.
    """
    def __init__(self, network_G, loss):
        """Initialization of the loss to set used metric (loss), crop size and image dimension.

        Args:
            network_G (str): Name of the generator. Used to extract whether 3D images are processed.
            loss (pytorch loss function): Used loss function after extracting the regions where the loss is applied.
        """
        super(InpaintingLoss, self).__init__()
        self.loss = loss
        if "3D" in network_G:
            self.dim = 3
        else:
            self.dim = 2

    def forward(self, input, target, inpainting_pixel, inpainting_mode, **kwargs):
        # for an easy/short implementation, parts not needed could be set to zero in weights.
        # since overlapping areas are small, it is better to crop the necessary parts before computing loss
        # ToDo: Speed comparison. If speed is equal => use short implementation
        
        # inpainting_pixel [Z, X, Y]
        if self.dim == 2:
            if (inpainting_mode == [True, False]).all():  # top [HxW]
                input = input[..., 0:inpainting_pixel[1], :]
                target = target[..., 0:inpainting_pixel[1], :]
                loss = (self.loss(input, target)).mean()
            elif (inpainting_mode == [False, True]).all():  # left
                input = input[..., 0:inpainting_pixel[2]]
                target = target[..., 0:inpainting_pixel[2]]
                loss = (self.loss(input, target)).mean()
            elif (inpainting_mode == [True, True]).all():  # top+left
                # left_part
                input_left = input[..., 0:inpainting_pixel[2]]
                target_left = target[..., 0:inpainting_pixel[2]]
                loss_left = (self.loss(input_left, target_left)).mean()
                # top_right part
                input_tr = input[..., :inpainting_pixel[1], inpainting_pixel[2]:]
                target_tr = target[..., :inpainting_pixel[1], inpainting_pixel[2]:]
                # cut left part, normalize and afterwards cut top part this will result in matching weights for the top part of mode 2 (top and left)
                loss_tr = (self.loss(input_tr, target_tr)).mean()
                loss = loss_left + loss_tr
        elif self.dim == 3:
            batch_size = input.shape[0]
            color_channels = input.shape[1]
            size_z = input.shape[2]
            size_x = input.shape[3]
            size_y = input.shape[4]
            if (inpainting_mode == [True, False, False]).all():  # Front [ZxHxW]             inpainting_size
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()
            elif (inpainting_mode == [False, True, False]).all():  # top
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                       np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()
            elif (inpainting_mode == [False, False, True]).all():  # left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z], np.r_[0:size_x],
                                                       np.r_[0:inpainting_pixel[2]])), input.shape).flatten()
            elif (inpainting_mode == [True, True, False]).all():  # front, top
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()  # front
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()  # top without front
                indices = np.concatenate((indices, indices_2), axis=0)
            elif (inpainting_mode == [True, False, True]).all():  # front, left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()  # front
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[0:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()  # left without front
                indices = np.concatenate((indices, indices_2), axis=0)
            elif (inpainting_mode == [False, True, True]).all():  # top, left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                       np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()  # top
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:size_z],
                                                         np.r_[inpainting_pixel[1]:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()  # left without top
                indices = np.concatenate((indices, indices_2), axis=0)
            elif (inpainting_mode == [True, True, True]).all():  # front, top, left
                indices = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[0:inpainting_pixel[0]],
                                                       np.r_[0:size_x], np.r_[0:size_y])), input.shape).flatten()  # front
                indices_2 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[0:inpainting_pixel[1]], np.r_[0:size_y])), input.shape).flatten()  # top without front
                indices_3 = np.ravel_multi_index((np.ix_(np.r_[0:batch_size], np.r_[0:color_channels], np.r_[inpainting_pixel[0]:size_z],
                                                         np.r_[inpainting_pixel[1]:size_x], np.r_[0:inpainting_pixel[2]])), input.shape).flatten()  # left without front and top
                indices = np.concatenate((indices, indices_2, indices_3), axis=0)
            loss = (self.loss(input.flatten()[indices], target.flatten()[indices])).mean()
        return loss


class SideWeightedLoss(nn.Module):
    def __init__(self, weight, weight_side, loss):
        """Weight loss according to the side the prediction.
        Prediction can be lower than target or higher than target.
        This loss can be used to force prediction towards lower or higher than target
        This is done by adding a weight towards all values on the wrong side enforcing the network to predict values on the other
        side rather than on the side with the higher weights.

        Args:
            weight (float): Weight of the side you want to pull the predictions towards (>1)
            weight_side (str): Side of values, you want to pull the predictions towards (lower or higher)
            loss (pytorch loss function): Loss used (e.g. L1, MSE, SmoothL!) needs to be initialized with reduction="none"

        Returns:
            tensor: loss
        """
        super(SideWeightedLoss, self).__init__()
        self.weight = weight
        self.weight_side = weight_side
        self.loss = loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weighting = torch.ones_like(input, device=input.device)
        if self.weight_side == "lower":  # weight input > target more (goal is to pull the predictions towards lower)
            weighting[(input > target)] = self.weight
        else:
            weighting[(input < target)] = self.weight
        loss = weighting*self.loss(input, target)
        return loss


class Loss(nn.Module):
    def __init__(self, criterion, hparams):
        super(Loss, self).__init__()
        self.criterion = criterion
        self.pass_kwargs = False
        if criterion == "L1":
            self.loss = nn.L1Loss()
        elif criterion == "MSE":
            self.loss = nn.MSELoss()
        elif criterion == "BCE":
            self.loss = nn.BCELoss()
        elif criterion == "SmoothL1":
            self.loss = nn.SmoothL1Loss()
        elif criterion == "DweightedMSE":
            self.loss = DirectionalWeightedMSELoss(hparams["crop_size"], hparams["network_G"])
            self.pass_kwargs = True
        elif criterion == "InpaintingMSE":
            self.loss = InpaintingLoss(hparams["network_G"], loss=nn.MSELoss(reduction="none"))
            self.pass_kwargs = True
        elif criterion == "InpaintingL1":
            self.loss = InpaintingLoss(hparams["network_G"], loss=nn.L1Loss(reduction="none"))
            self.pass_kwargs = True
        elif criterion == "InpaintingBCE":
            self.loss = InpaintingLoss(hparams["network_G"], loss=nn.BCELoss(reduction="none"))
            self.pass_kwargs = True
        elif criterion == "InpaintingSideWeightedSmoothL1":
            self.loss = InpaintingLoss(hparams["network_G"],
                                       loss=SideWeightedLoss(5, weight_side="lower", loss=nn.SmoothL1Loss(reduction="none")))
            self.pass_kwargs = True
        elif criterion == "InpaintingSideWeightedL1":
            self.loss = InpaintingLoss(hparams["network_G"],
                                       loss=SideWeightedLoss(5, weight_side="lower", loss=nn.L1Loss(reduction="none")))
            self.pass_kwargs = True
        elif criterion == "InpaintingSideWeightedMSE":
            self.loss = InpaintingLoss(hparams["network_G"],
                                       loss=SideWeightedLoss(5, weight_side="lower", loss=nn.MSELoss(reduction="none")))
            self.pass_kwargs = True
        else:
            raise NotImplementedError(f"Loss {criterion} is not implemented.")

    def forward(self, input: Tensor, target: Tensor, **kwargs) -> Tensor:
        if self.pass_kwargs:
            return self.loss(input, target, **kwargs)
        else:
            return self.loss(input, target)
