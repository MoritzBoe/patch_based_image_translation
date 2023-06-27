from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import itertools
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from data_modules import get_data_module
from networks import define_patchGAN_discriminator, define_generator
from losses import disc_loss, Loss


class LitCycleGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Networks
        self.gen_AB = define_generator(self.hparams, "A")
        self.gen_BA = define_generator(self.hparams, "B")

        # module list needed to store networks in list
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/8775
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList
        self.disc_A = nn.ModuleList()
        for idx, layers in np.ndenumerate(self.hparams.n_layers_D_A):
            self.disc_A.append(define_patchGAN_discriminator(self.hparams, layers, "A"))
        self.disc_B = nn.ModuleList()
        for idx, layers in np.ndenumerate(self.hparams.n_layers_D_B):
            self.disc_B.append(define_patchGAN_discriminator(self.hparams, layers, "B"))

        # get loss criterions
        self.criterion_idt_A = Loss(self.hparams["criterion_idt_A"], self.hparams)
        self.criterion_idt_B = Loss(self.hparams["criterion_idt_B"], self.hparams)
        self.criterion_cycle_ABA = Loss(self.hparams["criterion_cycle_ABA"], self.hparams)
        self.criterion_cycle_BAB = Loss(self.hparams["criterion_cycle_BAB"], self.hparams)
        self.criterion_inpainting = Loss(self.hparams["criterion_inpainting"], self.hparams)
        self.criterion_disc = disc_loss

        # calculate scaling values for domain A and B, GAN output is [-1,1]
        # final scaling is performed with output*self.scaling_X[0]+scaling_X[1]
        self.scaling_A = [(self.hparams["minmax_scale_A"][1]-self.hparams["minmax_scale_A"][0])/2,
                          self.hparams["minmax_scale_A"][0]-(-1)*((self.hparams["minmax_scale_A"][1]-self.hparams["minmax_scale_A"][0])/2)]
        self.scaling_B = [(self.hparams["minmax_scale_B"][1]-self.hparams["minmax_scale_B"][0])/2,
                          self.hparams["minmax_scale_B"][0]-(-1)*((self.hparams["minmax_scale_B"][1]-self.hparams["minmax_scale_B"][0])/2)]

        print("CycleGAN initialized")

    def on_train_epoch_start(self):
        # set lambda_inpainting
        if self.hparams["inpainting_start_epoch"] > self.current_epoch:
            self.lambda_inpainting = 0  # not needed since inpainting images are not created, but to be consistent
        elif self.hparams["lambda_inpainting_ramp"]:
            self.lambda_inpainting = (self.hparams["lambda_inpainting"]*(self.current_epoch-self.hparams["inpainting_start_epoch"]) /
                                      (self.hparams["n_epochs_lr_fix"]+self.hparams["n_epochs_lr_decay"]-self.hparams["inpainting_start_epoch"]))
        else:
            self.lambda_inpainting = self.hparams["lambda_inpainting"]
        self.log("lambda_inpainting", self.lambda_inpainting)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            # Do optimization for generators

            # calculate network outputs also needed later for optimizer 1 (discriminator)
            self.real_A = batch["A"]
            self.real_B = batch["B"]
            self.fake_B = self.gen_AB(self.real_A)*self.scaling_B[0]+self.scaling_B[1]  # gen_AB(self.real_A) in [-1,1], self.fake_B in [minmax_scale_B[0], minmax_scale_B[1]]
            self.fake_A = self.gen_BA(self.real_B)*self.scaling_A[0]+self.scaling_A[1]  # gen_BA(self.real_B) in [-1,1], self.fake_A in [minmax_scale_A[0], minmax_scale_A[1]]
            self.cycle_ABA = self.gen_BA(self.fake_B)*self.scaling_A[0]+self.scaling_A[1]
            self.cycle_BAB = self.gen_AB(self.fake_A)*self.scaling_B[0]+self.scaling_B[1]

            # disable grad for the discriminator networks
            self.set_requires_grad([*self.disc_A, *self.disc_B], False)

            # identity loss
            if self.hparams["lambda_idt"] > 0:
                self.idt_A = self.gen_BA(self.real_A)*self.scaling_A[0]+self.scaling_A[1]
                self.idt_B = self.gen_AB(self.real_B)*self.scaling_B[0]+self.scaling_B[1]
                loss_idt_A = self.criterion_idt_A(self.idt_A, self.real_A)
                loss_idt_B = self.criterion_idt_A(self.idt_B, self.real_B)
            else:
                loss_idt_A = torch.tensor(0, device=self.device, dtype=torch.float32)
                loss_idt_B = torch.tensor(0, device=self.device, dtype=torch.float32)

            # cycle loss
            loss_cycle_ABA = self.criterion_cycle_ABA(self.cycle_ABA, self.real_A)
            loss_cycle_BAB = self.criterion_cycle_BAB(self.cycle_BAB, self.real_B)

            # generator loss calculated from discriminator output
            loss_gen = {}
            for idx, disc in enumerate(self.disc_A):
                loss = self.criterion_disc(disc(self.fake_A), True, self.hparams["criterion_disc"])
                loss_gen[f"loss_gen_BA_{idx}"] = loss  # store loss in dict to make it easy to display by logger

            for idx, disc in enumerate(self.disc_B):
                loss = self.criterion_disc(disc(self.fake_B), True, self.hparams["criterion_disc"])
                loss_gen[f"loss_gen_AB_{idx}"] = loss  # store loss in dict to make it easy to display by logger

            if self.current_epoch >= self.hparams["inpainting_start_epoch"]:
                if self.hparams["variable_inpainting"]:
                    inpainting_percent = np.random.randint(1, self.hparams["inpainting_percent"])
                else:
                    inpainting_percent = self.hparams["inpainting_percent"]
                # calculate inpainting pixels ZXY => -3 -2 -1
                inpainting_pixel = [round(inpainting_percent*self.real_A.shape[-3]/100), round(inpainting_percent*self.real_A.shape[-2]/100),
                                    round(inpainting_percent*self.real_A.shape[-1]/100)]
                    
                if self.hparams["disable_overlap_sampling"] is True:
                    inpainting_mode = [True for i in range(self.real_A.ndim-2)]  # True for each dim (2D [True, True], 3D [True, True, True])
                else:  # enable or disable inpainting for each dimension
                    n_dims_inpainting = np.random.randint(1, self.real_A.ndim-2+1)  # max is exclusive => +1
                    inpainting_mode = np.zeros(self.real_A.ndim-2).astype(bool)
                    choices = np.random.choice(self.real_A.ndim-2, n_dims_inpainting, replace=False)
                    inpainting_mode[choices] = True

                self.real_inpainting_A = self.generate_real_inpainting_A(self.real_A, self.fake_B, inpainting_pixel, inpainting_mode)
                self.fake_inpainting_B = self.gen_AB(self.real_inpainting_A)*self.scaling_B[0]+self.scaling_B[1]

                loss_inpainting_identity = self.criterion_inpainting(self.fake_inpainting_B, self.fake_B,
                                                                     inpainting_pixel=inpainting_pixel, inpainting_mode=inpainting_mode)

                # use discriminator to create overall image loss
                for idx, disc in enumerate(self.disc_B):
                    loss = self.criterion_disc(disc(self.fake_inpainting_B), True, self.hparams["criterion_disc"])
                    loss_gen[f"loss_gen_inpainting_{idx}"] = loss  # store loss in dict to make it easy to display by logger
            else:
                loss_inpainting_identity = torch.tensor(0, device=self.device, dtype=torch.float32)

            # Merge generator losses for all discriminators into one
            loss_gen_sum = torch.tensor(0, device=self.device, dtype=torch.float32)
            for loss in loss_gen.values():
                loss_gen_sum = loss_gen_sum + loss

            final_loss_generators = ((loss_idt_A+loss_idt_B)*self.hparams["lambda_idt"]
                                     + (loss_cycle_ABA+loss_cycle_BAB)*self.hparams["lambda_cycle"]
                                     + loss_inpainting_identity*self.lambda_inpainting+loss_gen_sum)
            losses = {"loss_generators": final_loss_generators,
                      "loss_idt_A": loss_idt_A, "loss_idt_B": loss_idt_B,
                      "loss_cycle_ABA": loss_cycle_ABA, "loss_cycle_BAB": loss_cycle_BAB,
                      "loss_inpainting_identity": loss_inpainting_identity, **loss_gen}
            return [final_loss_generators, losses]  # hand back loss separately to not log discriminator loss and generator loss for the same keyword ("loss")

        if optimizer_idx == 1:
            # Do optimization for discriminators

            # enable grad for the discriminator networks
            self.set_requires_grad([*self.disc_A, *self.disc_B], True)

            # loss for discriminators on real images
            loss_disc = {}
            for idx, disc in enumerate(self.disc_A):
                loss = self.criterion_disc(disc(self.real_A), True, self.hparams["criterion_disc"])
                loss_disc[f"loss_disc_realA_{idx}"] = loss  # store loss in dict to make it easy to display by logger

            for idx, disc in enumerate(self.disc_B):
                loss = self.criterion_disc(disc(self.real_B), True, self.hparams["criterion_disc"])
                loss_disc[f"loss_disc_realB_{idx}"] = loss  # store loss in dict to make it easy to display by logger

            # loss for discriminators on fake images detach to have no backprob errors for generator
            # already used in generator step and no detach would result in
            # RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed.
            # Specify retain_graph=True when calling backward the first time.
            # Since we do not want to retain the graph for the generators, detach() and enable requires_grad
            fake_A = self.fake_A.detach()
            fake_B = self.fake_B.detach()
            fake_A.requires_grad = True
            fake_B.requires_grad = True

            for idx, disc in enumerate(self.disc_A):
                loss = self.criterion_disc(disc(fake_A), False, self.hparams["criterion_disc"])
                loss_disc[f"loss_disc_fakeA_{idx}"] = loss  # store loss in dict to make it easy to display by logger

            for idx, disc in enumerate(self.disc_B):
                loss = self.criterion_disc(disc(fake_B), False, self.hparams["criterion_disc"])
                loss_disc[f"loss_disc_fakeB_{idx}"] = loss  # store loss in dict to make it easy to display by logger

            if self.current_epoch >= self.hparams["inpainting_start_epoch"]:
                fake_inpainting_B = self.fake_inpainting_B.detach()
                fake_inpainting_B.required_grad = True
                for idx, disc in enumerate(self.disc_B):
                    loss = self.criterion_disc(disc(fake_inpainting_B), False, self.hparams["criterion_disc"])
                    loss_disc[f"loss_disc_fake_inpainting_B_{idx}"] = loss

            # Merge generator losses for all discriminators into one
            loss_disc_sum = torch.tensor(0, device=self.device, dtype=torch.float32)
            for loss in loss_disc.values():
                loss_disc_sum = loss_disc_sum + loss

            losses = {"loss_discriminators": loss_disc_sum, **loss_disc}
            return [loss_disc_sum, losses]  # hand back loss separately to not log discriminator loss and generator loss for the same keyword ("loss")

    def generate_real_inpainting_A(self, real_A, fake_B, inpainting_pixel, inpainting_mode):
        """ generate the image real_inpainting_A from real_A and fake_B by replacing parts of real_A with parts of fake_B

        Args:
            real_A (tensor): image from domain A
            fake_B (tensor): image created by the GAN with gen_AB(real_A)
            inpainting_pixel (list[int]): Inpainting in pixel for each dim
            inpainting_mode (list of bool): bool for each dim whether to replace parts of real_A or not

        Returns:
            tensor: image consisting of parts of real_A and fake_B depending on inpainting mode and num_rows_cols_inpainting
        """

        # detach().clone() to not propagate loss back through generation of fake_B when calculating inpainting loss
        real_inpainting_A = real_A.detach().clone()

        for idx, region in enumerate(inpainting_mode):
            if region:
                if idx == 0:  # x (top)
                    real_inpainting_A[...,0:inpainting_pixel[1],:] = fake_B[...,0:inpainting_pixel[1],:].detach().clone()
                elif idx == 1:  # y (left)
                    real_inpainting_A[...,0:inpainting_pixel[2]] = fake_B[...,0:inpainting_pixel[2]].detach().clone()
                elif idx == 2:  # z (front)
                    real_inpainting_A[...,0:inpainting_pixel[0],:,:] = fake_B[...,0:inpainting_pixel[0],:,:].detach().clone()
        return real_inpainting_A

    def training_step_end(self, outs):
        # used for logging to work with DDP instead of logging in training_step
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=training_step#training-step
        # https://docs.wandb.ai/guides/integrations/lightning
        self.log_dict(outs[1], on_epoch=True, batch_size=self.hparams["batch_size"])
        return outs[0]  # route through the final loss to the optimizer

    def test_step(self, batch, batch_idx):
        """In test-mode, standard cycleGAN outputs are created.
        No inpainting images are created. we do this in separate scripts directly accessing the generators.

        Args:
            batch (dict of tensors): Images from domain A and B
            batch_idx (_type_): idx of batch

        Returns:
            dict of tensors : outputs of cycleGAN
        """
        # calculate results, images are uploaded in the WandbImageCallback
        outputs = batch
        # calculate network outputs
        outputs["fake_B"] = self.gen_AB(batch["A"])*self.scaling_B[0]+self.scaling_B[1]
        outputs["fake_A"] = self.gen_BA(batch["B"])*self.scaling_A[0]+self.scaling_A[1]
        outputs["cycle_ABA"] = self.gen_BA(outputs["fake_B"])*self.scaling_A[0]+self.scaling_A[1]
        outputs["cycle_BAB"] = self.gen_AB(outputs["fake_A"])*self.scaling_B[0]+self.scaling_B[1]
        if self.hparams["lambda_idt"] > 0:
            outputs["idt_A"] = self.gen_BA(batch["A"])*self.scaling_A[0]+self.scaling_A[1]
            outputs["idt_B"] = self.gen_AB(batch["B"])*self.scaling_B[0]+self.scaling_B[1]
        return outputs

    def configure_optimizers(self):
        # iterators from *.parameters are chained together with itertools
        optimizer_G = torch.optim.Adam(itertools.chain(self.gen_AB.parameters(), self.gen_BA.parameters()),
                                       self.hparams["lr_G"], betas=(0.5, 0.999))

        # parameters of all discriminator are stored in list and lists are unpacked for itertools.chain.
        # afterwards all parameters are handed to the optimizer
        optimizer_D = torch.optim.Adam(itertools.chain(*[disc.parameters() for disc in self.disc_A],
                                                       *[disc.parameters() for disc in self.disc_B]),
                                       self.hparams["lr_D"], betas=(0.5, 0.999))
        optimizers = [optimizer_G, optimizer_D]

        def lambda_decay(epoch):
            if self.hparams["mode"] == "start_from_ckpt":
                # start with lr of 0 for n_epochs_lr_fix and go up to lr for n_epochs_lr_fix+n_epochs_lr_decay/2 and then back to zero until n_epochs_lr_fix + n_epochs_lr_decay is reached
                # + 1 because current_epoch starts with 0
                if self.trainer.current_epoch <= self.hparams["n_epochs_lr_fix"] + self.hparams["n_epochs_lr_decay"]/2:
                    decay = (self.trainer.current_epoch + 1 - self.hparams["n_epochs_lr_fix"]) / (self.hparams["n_epochs_lr_decay"]/2)
                else:
                    decay = 1.0 - (self.trainer.current_epoch + 1 - self.hparams["n_epochs_lr_fix"] - (self.hparams["n_epochs_lr_decay"]/2)) / (self.hparams["n_epochs_lr_decay"]/2)
            else:
                decay = 1.0 - max(0, self.trainer.current_epoch - self.hparams["n_epochs_lr_fix"]) / float(self.hparams["n_epochs_lr_decay"] + 1)
            return decay

        scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_decay)
        scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_decay)
        schedulers = [scheduler_G, scheduler_D]

        return optimizers, schedulers

    def set_requires_grad(self, nets, requires_grad=False):
        """set requires_grad=False for all networks given

        Args:
            nets (nn.module or list of nn.module): pytorch networks
            requires_grad (bool, optional): Set parameters of nets to requires_grad. Defaults to False.
        """

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_channels_A', type=int, default=1, help="Number of channels of images in domain A")
        parser.add_argument('--num_channels_B', type=int, default=1, help="Number of channels of images in domain B")
        parser.add_argument('--crop_size', type=int, default=256, nargs='*', help="Size of the random crop during augmentation. When multiple values are passed, dims have to be passed as Z X Y.")
        parser.add_argument('--network_G',type=str, default="resnet", help="Generator network [unet, resnet, resnet3D]")
        parser.add_argument('--network_D', type=str, default="patchGAN", help="Diskriminator network [patchGAN, patchGAN3D]")
        parser.add_argument('--num_resnet_blocks_3d', type=int, default=9, help="Number of ResNet blocks for resnet3D")
        parser.add_argument('--generator_norm', type=str, default="instance", help="Normalization layer of the generator [instance, batch, spectral]")
        parser.add_argument('--discriminator_norm', type=str, default="instance", help="Normalization layer of the discriminator [instance, batch, spectral]")
        parser.add_argument('--ndf', type=int, default=64, help="Number of feature maps in the first layer for the discriminator networks")
        parser.add_argument('--ngf', type=int, default=64, help="Number of feature maps in the first layer for the generator networks")
        parser.add_argument('--n_layers_D_A', type=int, default=3, nargs='*', help="Number of layers for the discriminators A")
        parser.add_argument('--n_layers_D_B', type=int, default=3, nargs='*', help="Number of layers for the discriminators B")
        parser.add_argument('--lr_G', type=float, default=0.0002, help="Learning rate for the generators")
        parser.add_argument('--lr_D', type=float, default=0.0002, help="Learning rate for the discriminators")
        parser.add_argument('--n_epochs_lr_fix', type=int, default=200, help="Number of epochs with the initial learning rate")
        parser.add_argument('--n_epochs_lr_decay', type=int, default=200, help="Number of epochs where the learning rate decays from initial to zero")

        parser.add_argument('--criterion_idt_A', type=str, default="L1", help="Loss criterion for identity loss on images A [L1, MSE, BCE]. BCE is only valid for binary input images.")
        parser.add_argument('--criterion_idt_B', type=str, default="L1", help="Loss criterion for identity loss on images B [L1, MSE, BCE]. BCE is only valid for binary input images.")
        parser.add_argument('--criterion_cycle_ABA', type=str, default="L1", help="Loss criterion for cycle consistency loss on images ABA [L1, MSE, BCE]. BCE is only valid for binary input images.")
        parser.add_argument('--criterion_cycle_BAB', type=str, default="L1", help="Loss criterion for cycle consistency loss on images BAB [L1, MSE, BCE]. BCE is only valid for binary input images.")
        parser.add_argument('--criterion_inpainting', type=str, default="L1", help="Loss criterion for inpainting loss on images fake_inpainting_B [L1, MSE, BCE, DweightedMSE]. BCE is only valid for binary input images.")
        parser.add_argument('--criterion_disc', type=str, default="MSE", help="Loss criterion for the discriminator output [L1, MSE, BCE]")
        parser.add_argument('--disable_overlap_sampling', '-dos', default=False, action='store_true', help="Disable overlap sampling for the inpainting mode. When disabled always use left and top for 2D and front, left and top for 3D.")

        parser.add_argument('--lambda_idt', type=float, default=5, help="Scaling factor for identity loss criterion.")
        parser.add_argument('--lambda_cycle', type=float, default=10, help="Scaling factor for cycle loss criterion.")
        parser.add_argument('--lambda_inpainting', type=float, default=10, help="Scaling factor for inpainting loss criterion.")
        parser.add_argument('--lambda_inpainting_ramp', '-lir', default=False, action='store_true', help="Ramp lambda_inpainting from 0 (epoch=inpainting_start_epoch) to lambda_inpainting (epoch=n_epochs_lr_fix+n_epochs_lr_decay)")

        parser.add_argument('--inpainting_percent', type=int, default=25, help="Part of real_A, that is impainted with fake_B to learn inpainting. Inpainting is done in percent to the full image size")
        parser.add_argument('--inpainting_start_epoch', type=int, default=200, help="First epoch where the inpainting is performed. Remember, epoch starts at 0.")
        parser.add_argument('--variable_inpainting', '-vi', default=False, action='store_true', help="Vary inpainting percent during training.")

        parser.add_argument('--minmax_scale_A', type=int, default=[-1, 1], nargs=2, help="Minimum and maximum domain A is scaled.")
        parser.add_argument('--minmax_scale_B', type=int, default=[-1, 1], nargs=2, help="Minimum and maximum domain A is scaled.")
        return parser


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of LitCycleGAN.
    For 3D images, we extract the middle slice of z

    Images are stacked into a mosaic in the following way:
    real_A | fake_B | cycle_ABA | idt_A | real_inpainting_A
    real_B | fake_A | cycle_BAB | idt_B | real_inpainting_B
    """

    def __init__(self, num_val_samples=2, val_frequence=1):
        super().__init__()
        self.num_val_samples = num_val_samples
        self.val_frequence = val_frequence
        self.test_step = 0

    def on_train_epoch_end(self, trainer, pl_module):
        if self.num_val_samples == 0:
            return
        if (trainer.current_epoch % self.val_frequence) != 0:
            return
        outputs = []
        for idx in range(self.num_val_samples):
            sample = trainer.train_dataloader.dataset.datasets.__getitem__(idx)
            # add batchsize dimension since we did not use the dataloader
            output = {}
            output["A"] = torch.unsqueeze(sample["A"],0).to(device=pl_module.device)
            output["B"] = torch.unsqueeze(sample["B"], 0).to(device=pl_module.device)
            output["fake_B"] = pl_module.gen_AB(output["A"])*trainer.model.scaling_B[0]+trainer.model.scaling_B[1]
            output["fake_A"] = pl_module.gen_BA(output["B"])*trainer.model.scaling_A[0]+trainer.model.scaling_A[1]
            output["cycle_ABA"] = pl_module.gen_BA(output["fake_B"])*trainer.model.scaling_A[0]+trainer.model.scaling_A[1]
            output["cycle_BAB"] = pl_module.gen_AB(output["fake_A"])*trainer.model.scaling_B[0]+trainer.model.scaling_B[1]
            if pl_module.hparams["lambda_idt"] > 0:
                output["idt_A"] = pl_module.gen_BA(output["A"])*trainer.model.scaling_A[0]+trainer.model.scaling_A[1]
                output["idt_B"] = pl_module.gen_AB(output["B"])*trainer.model.scaling_B[0]+trainer.model.scaling_B[1]

            # inpainting images only performed in num_channels_A==num_channels_B otherwise does not work
            if pl_module.hparams["num_channels_A"] == pl_module.hparams["num_channels_B"]: 
                # create slicing for 2D or 3D image. First two dims are batch_size, color_channel
                n_dim = output["A"].ndim
                
                start = np.zeros(n_dim, dtype=int)
                for i in range(2, n_dim):
                    start[i] = round(output["A"].shape[i]*pl_module.hparams["inpainting_percent"]/100)
                end = np.full((n_dim), output["A"].shape)
                slicing = tuple(map(slice, start, end))

                output["real_inpainting_A"] = output["fake_B"].detach().clone()
                output["real_inpainting_A"][slicing] = output["A"][slicing]
                output["fake_inpainting_B"] = pl_module.gen_AB(output["real_inpainting_A"])*trainer.model.scaling_B[0]+trainer.model.scaling_B[1]
            outputs.append(output)
        self.create_logging_image(trainer, outputs, mode="val", step=trainer.global_step)

    def create_logging_image(self, trainer, outputs, mode, step):
        # Merge Images to scheme and removes the images not present (e.g. when idt_loss is 0 or num_channels_A and num_channels_B differ and no inpainting is created):
        # real_A | fake_B | cycle_ABA | idt_A | real_inpainting_A
        # real_B | fake_A | cycle_BAB | idt_B | fake_inpainting_B
        top_keys_order = ["A", "fake_B", "cycle_ABA", "idt_A", "real_inpainting_A"]
        bottom_keys_order = ["B", "fake_A", "cycle_BAB", "idt_B", "fake_inpainting_B"]
        top_keys_order = [top_key for top_key in top_keys_order if top_key in outputs[0].keys()]
        bottom_keys_order = [bottom_key for bottom_key in bottom_keys_order if bottom_key in outputs[0].keys()]

        # when one domain is grayscale and one is RGB, the grayscale outputs have to be converted to RGB for logging
        if trainer.model.hparams["num_channels_A"] != trainer.model.hparams["num_channels_B"]:
            outputs = self.convert_grayscale_to_rgb(outputs)

        outputs = self.scale_outputs(outputs, trainer.model.hparams["minmax_scale_A"], trainer.model.hparams["minmax_scale_B"])
        if outputs[0]["A"].ndim == 5:  # batch_size+RGB+3D = 5
            outputs = self.get_plane_from_3D(outputs)
        combinations = []

        for i in range(0, len(outputs)):
            top_row = torch.cat([outputs[i][top_key] for top_key in top_keys_order], dim=-1)  # merge horizontally
            bottom_row = torch.cat([outputs[i][bottom_key] for bottom_key in bottom_keys_order], dim=-1)  # merge horizontally
            combination = torch.cat([top_row, bottom_row], dim=-2)  # merge vertically
            # create a uint8 numpy array to get images in range [0,255] to work with wandb. Values <0 and >255 are clipped to 0 and 255
            combination = combination.detach().cpu().numpy()
            combination = combination.astype(np.uint8)
            # move color axis to the back to enable wandb to infer colormode from channels
            combination = np.moveaxis(combination, -3, -1)
            combinations.append(combination)

        caption = "Top: " + " | ".join(top_keys_order) + "  Bottom: " + " | ".join(bottom_keys_order)
        trainer.logger.experiment.log({mode+"/examples": [wandb.Image(combination, caption=caption) for combination in combinations]})

    def convert_grayscale_to_rgb(self, outputs):
        for output in outputs:
            for key in output.keys():
                if output[key].shape[-3] == 1:
                    output[key] = output[key].expand(-1, 3, -1, -1)  # expand the grayscale dim to 3
        return outputs

    def scale_outputs(self, outputs, input_norm_A, input_norm_B):
        norm_A = ["A", "fake_A", "cycle_ABA", "idt_A"]
        norm_B = ["B", "fake_B", "cycle_BAB", "idt_B", "fake_inpainting_B"]
        norm_inpainting = ["real_inpainting_A"]  # scale with global min and global max

        for output in outputs:
            for norm in norm_A:
                if norm in output.keys():  # check if output if present (e.g. no idt_A for --lambda_idt = 0)
                    output[norm] = (output[norm]-input_norm_A[0])*255/(input_norm_A[1]-input_norm_A[0])  # (X-Xmin)*255/(Xmax-Xmin)
            for norm in norm_B:
                if norm in output.keys():
                    output[norm] = (output[norm]-input_norm_B[0])*255/(input_norm_B[1]-input_norm_B[0])  # (X-Xmin)*255/(Xmax-Xmin)
            input_norm_inpainting = (np.min([input_norm_A[0], input_norm_B[0]]), np.max([input_norm_A[1], input_norm_B[1]]))
            for norm in norm_inpainting:
                output[norm] = (output[norm]-input_norm_inpainting[0])*255/(input_norm_inpainting[1]-input_norm_inpainting[0])  # (X-Xmin)*255/(Xmax-Xmin)
        return outputs

    def get_plane_from_3D(self, outputs):
        ret = []
        for idx, output in enumerate(outputs):
            ret.append({})
            for key in output.keys():
                ret[idx][key] = output[key][:,:,int(outputs[0]["A"].shape[2]/2),:,:]
        return ret


def train(args):
    # log into wandb and create logger
    wandb.login()
    run = wandb.init(project=args.project, resume="never", entity=args.entity)  # ensure, that the id is unique and a new run is started
    wandb_logger = WandbLogger(project=args.project, log_model=True)

    model = LitCycleGAN(**vars(args))
    data = get_data_module(args.data_module, batch_size=args.batch_size, num_workers=args.num_workers, crop_size=args.crop_size,
                           minmax_scale_A=args.minmax_scale_A, minmax_scale_B=args.minmax_scale_B)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="logs/"+args.project+"/"+run.name,
                                                       every_n_epochs=args.n_epochs_lr_fix+args.n_epochs_lr_decay)  # save only final model
    # use from_argparse_args instead of pl.Trainer() to be able to include arbitrary trainer args,
    # while overwriting them with the explicitly given ones (e.g. gpus, max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, gpus=-1,
                                            max_epochs=args.n_epochs_lr_fix+args.n_epochs_lr_decay,
                                            callbacks=[WandbImageCallback(args.num_val_samples, args.val_frequence),
                                                       pl.callbacks.LearningRateMonitor(logging_interval='step'),
                                                       pl.callbacks.RichProgressBar(refresh_rate_per_second=1), checkpoint_callback])
    trainer.fit(model, data)

    # finish wandb run
    wandb.finish()


def start_from_ckpt(args):
    # Please note, that the checkpoint hparams are overwritten by argparse arguments
    # this is also true for default argparse arguments!
    # Please note, that we start from a checkpoint and only initialize the networks with the correct weights.
    # We do not resume rom a checkpoint, where epoch / global steps / optimizer / scheduler options are also loaded

    # log into wandb and create logger
    wandb.login()
    run = wandb.init(project=args.project, resume="never", entity=args.entity)  # ensure, that the id is unique and a new run is started
    wandb_logger = WandbLogger(project=args.project, log_model=True)

    # download artifact
    artifact = run.use_artifact(args.ckpt_artifact, type='model')
    artifact_dir = artifact.download(root=run.dir)

    # create model and trainer
    model = LitCycleGAN.load_from_checkpoint(checkpoint_path=Path(artifact_dir).joinpath("model.ckpt"), **vars(args))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="logs/"+args.project+"/"+run.name,
                                                       every_n_epochs=args.n_epochs_lr_fix+args.n_epochs_lr_decay)  # save only final model
    # use from_argparse_args instead of pl.Trainer() to be able to include arbitrary trainer args,
    # while overwriting them with the explicitly given ones (e.g. gpus, max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, gpus=-1,
                                            max_epochs=args.n_epochs_lr_fix+args.n_epochs_lr_decay,
                                            callbacks=[WandbImageCallback(args.num_val_samples, args.val_frequence),
                                                       pl.callbacks.LearningRateMonitor(logging_interval='step'),
                                                       pl.callbacks.RichProgressBar(refresh_rate_per_second=1), checkpoint_callback])
    data = get_data_module(args.data_module, batch_size=args.batch_size, num_workers=args.num_workers, crop_size=args.crop_size,
                           minmax_scale_A=args.minmax_scale_A, minmax_scale_B=args.minmax_scale_B)

    # start training from checkpoint
    trainer.fit(model, data)  # add to resume from checkpoint ckpt_path=Path(artifact_dir).joinpath("model.ckpt")
    wandb.finish()


if __name__ == "__main__":
    print(f"Cuda avail: {torch.cuda.is_available()}")
    print(f"GPU IDs avail: {torch.cuda.device_count()}")
    # parse args
    parser = ArgumentParser()
    parser.add_argument('--mode', default="train", type=str, help="Define mode (train, start_from_ckpt")
    parser.add_argument('--project', default="test", type=str, help="Name of the project you are working on")
    parser.add_argument('--batch_size', default=5, type=int, help="Batchsize fed into the network")
    parser.add_argument('--num_workers', default=10, type=int, help="Number of workers for the dataloader")
    parser.add_argument('--data_module', default="BenchmarkColorRGBDataModule", type=str, help="Name of the datamodule used. Datamodule needs to be implemented in data_modules.py")
    parser.add_argument('--ckpt_artifact', type=str, help="Artifact path in wandb artifact API (user/project/Version). Needed for mode=start_from_local_ckpt")
    parser.add_argument('--num_val_samples', type=int, default=2, help='Number of images saved for validation')
    parser.add_argument('--val_frequence', type=int, default=1, help='Frequence validation image is created.')
    parser.add_argument('--entity', type=str, help="Entity used for wandb.")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitCycleGAN.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "start_from_ckpt":
        start_from_ckpt(args)
    else:
        print(f"Mode {args.mode} not implemented yet.")
