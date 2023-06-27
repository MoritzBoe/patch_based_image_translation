from torch import nn
import torch

"""
The networks are adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix  (accessed: 19.10.2022)
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def conv_norm(conv_layer, norm, num_features=None):
    if norm == "instance":
        return conv_layer, nn.InstanceNorm2d(num_features)
    elif norm == "batch":
        return conv_layer, nn.BatchNorm2d(num_features)
    elif norm == "spectral":
        return [nn.utils.spectral_norm(conv_layer)]
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm)


def conv_norm_3d(conv_layer, norm, num_features=None):
    if norm == "instance":
        return conv_layer, nn.InstanceNorm3d(num_features)
    elif norm == "batch":
        return conv_layer, nn.BatchNorm3d(num_features)
    elif norm == "spectral":
        return [nn.utils.spectral_norm(conv_layer)]
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, opt, input_nc, output_nc, num_downs=8, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            opt["ngf"](int)       -- the number of filters in the last conv layer
            opt["generator_norm"](str)      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        norm_layer = opt["generator_norm"]
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(opt["ngf"] * 8, opt["ngf"] * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with opt["ngf"]* 8 filters
            unet_block = UnetSkipConnectionBlock(opt["ngf"] * 8, opt["ngf"] * 8, norm_layer=norm_layer, input_nc=None,
                                                 submodule=unet_block, use_dropout=use_dropout)
        # gradually reduce the number of filters from opt["ngf"]* 8 to ngf
        unet_block = UnetSkipConnectionBlock(opt["ngf"] * 4, opt["ngf"] * 8, norm_layer=norm_layer, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(opt["ngf"] * 2, opt["ngf"] * 4, norm_layer=norm_layer, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(opt["ngf"], opt["ngf"] * 2, norm_layer=norm_layer, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, opt["ngf"], norm_layer=norm_layer, input_nc=input_nc,
                                             submodule=unet_block, outermost=True)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, norm_layer, input_nc=None,
                 submodule=None, outermost=False, innermost=False, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == "instance"
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer  # inner_nc
        uprelu = nn.ReLU(True)
        upnorm = norm_layer  # outer_nc

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, *conv_norm(upconv, upnorm, outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, *conv_norm(downconv, downnorm, inner_nc)]
            up = [uprelu, *conv_norm(upconv, upnorm, outer_nc)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


def define_unet(opt, input_image_domain):
    """input_image_domain (str) -- input domain of the generator (A or B)"""
    if "3D" in opt["network_G"]:
        if input_image_domain == "A":
            net = UNet3DPixelshuffle(opt, opt["num_channels_A"], opt["num_channels_B"])
        else:
            net = UNet3DPixelshuffle(opt, opt["num_channels_B"], opt["num_channels_A"])        
    else:
        if input_image_domain == "A":
            net = UnetGenerator(opt, opt["num_channels_A"], opt["num_channels_B"])
        else:
            net = UnetGenerator(opt, opt["num_channels_B"], opt["num_channels_A"])
    return net


def define_resnet(opt, input_image_domain):
    """input (str) -- input domain of the generator (A or B)"""
    if "3D" in opt["network_G"]:
        if input_image_domain == "A":
            net = ResnetGenerator3D(opt, opt["num_channels_A"], opt["num_channels_B"])
        else:
            net = ResnetGenerator3D(opt, opt["num_channels_B"], opt["num_channels_A"]) 
    else:
        if input_image_domain == "A":
            net = ResnetGenerator(opt, opt["num_channels_A"], opt["num_channels_B"])
        else:
            net = ResnetGenerator(opt, opt["num_channels_B"], opt["num_channels_A"])
    return net


def define_generator(opt, input_image_domain):
    """ picks generator

    Args:
        opt: options including opt["network_G"]
        input_image_domain (str): Domain A or B

    Returns:
        net: network
    """
    if "resnet" in opt["network_G"]:
        return define_resnet(opt, input_image_domain)
    elif "unet" in opt["network_G"]:
        return define_unet(opt, input_image_domain)
    else:
        raise ValueError(f"network type {opt['network_G']} unknown")


class ResnetGenerator(nn.Module):
    """Create a generator

    Parameters:
        opt["input_nc"] (int) -- the number of channels in input images
        opt["output_nc"] (int) -- the number of channels in output images
        opt.ngf (int) -- the number of filters in the last conv layer
        #netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        opt.norm (str) -- the name of normalization layers used in the network: instance
        #use_dropout (bool) -- if use dropout layers.
        #init_type (str)    -- the name of our initialization method.
        #init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        #gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    def __init__(self, opt, input_nc, output_nc):
        super(ResnetGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 *conv_norm(nn.Conv2d(input_nc, opt["ngf"], kernel_size=7, padding=0, bias=True), opt["generator_norm"], opt["ngf"]),
                 nn.LeakyReLU(0.2, True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [*conv_norm(nn.Conv2d(opt["ngf"] * mult, opt["ngf"] * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                                 opt["generator_norm"], opt["ngf"] * mult * 2),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** n_downsampling
        for i in range(9):
            model += [ResnetBlock(opt["ngf"] * mult, opt)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [*conv_norm(nn.ConvTranspose2d(opt["ngf"] * mult, int(opt["ngf"] * mult / 2),
                                                    kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                                 opt["generator_norm"], int(opt["ngf"] * mult / 2)),
                      nn.LeakyReLU(0.2, True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(opt["ngf"], output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """ Define a Resnet block"""

    def __init__(self, dim, opt):
        super(ResnetBlock, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       *conv_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), self.opt["generator_norm"], dim),
                       nn.LeakyReLU(0.2, True),
                       nn.ReflectionPad2d(1),
                       *conv_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True), self.opt["generator_norm"], dim)]

        return nn.Sequential(*conv_block)  # unpacks the list into positional arguments

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


def define_patchGAN_discriminator(opt, n_layers, input_image_domain):
    """input_image_domain (str) -- input domain of the discriminator (A or B)"""
    if not hasattr(opt, 'network_D'):  # fix to enable inference on older lit_cyclegan versions
        if input_image_domain == "A":
            net = PatchGANDiscriminator(opt, opt["num_channels_A"], n_layers=n_layers)
        else:
            net = PatchGANDiscriminator(opt, opt["num_channels_B"], n_layers=n_layers)        
    elif "3D" in opt["network_D"]:
        if input_image_domain == "A":
            net = PatchGANDiscriminator3D(opt, opt["num_channels_A"], n_layers=n_layers)
        else:
            net = PatchGANDiscriminator3D(opt, opt["num_channels_B"], n_layers=n_layers)
    else:
        if input_image_domain == "A":
            net = PatchGANDiscriminator(opt, opt["num_channels_A"], n_layers=n_layers)
        else:
            net = PatchGANDiscriminator(opt, opt["num_channels_B"], n_layers=n_layers)
    return net


class PatchGANDiscriminator(nn.Module):
    """Create a PatchGAN discriminator"""

    def __init__(self, opt, input_nc, n_layers=3):
        """ Construct the PatchGAN discriminator

        Parameters:
            input_nc (int)     -- the number of channels in input images
            ndf (int)          -- the number of filters in the first conv layer
            norm (str)         -- the type of normalization layers used in the network.

        Returns a discriminator
        """
        super(PatchGANDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.n_layers = n_layers

        model = [nn.Conv2d(self.input_nc, opt["ndf"], kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2, True)]

        # PatchGAN discriminator with receptive field of 70
        for i in range(self.n_layers-1):  # add downsampling layers
            mult = 2 ** i
            model += [*conv_norm(nn.Conv2d(opt["ndf"] * mult, opt["ndf"] * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                                 opt["discriminator_norm"], opt["ndf"] * mult * 2), nn.LeakyReLU(0.2, True)]

        model += [*conv_norm(nn.Conv2d(opt["ndf"] * (2**(self.n_layers-1)), opt["ndf"] * (2**self.n_layers), kernel_size=4,
                                       stride=1, padding=1, bias=True), opt["discriminator_norm"], opt["ndf"] * (2**self.n_layers)),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv2d(opt["ndf"] * (2**self.n_layers), 1, kernel_size=4, stride=1, padding=1, bias=True)]  # 1 channel prediction output

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ResnetGenerator3D(nn.Module):
    """Create a generator for 3D data

    Parameters:
        opt["input_nc"] (int) -- the number of channels in input images
        opt["output_nc"] (int) -- the number of channels in output images
        opt.ngf (int) -- the number of filters in the last conv layer
        #netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        opt.norm (str) -- the name of normalization layers used in the network: instance
        #use_dropout (bool) -- if use dropout layers.
        #init_type (str)    -- the name of our initialization method.
        #init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        #gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        opt["num_resnet_blocks_3d"] (int) -- Number of ResNet blocks to use

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """

    def __init__(self, opt, input_nc, output_nc):
        super(ResnetGenerator3D, self).__init__()

        model = [nn.ReflectionPad3d(3),
                 *conv_norm_3d(nn.Conv3d(input_nc, opt["ngf"], kernel_size=7, padding=0, bias=True), opt["generator_norm"], opt["ngf"]),
                 nn.LeakyReLU(0.2, True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [*conv_norm_3d(nn.Conv3d(opt["ngf"] * mult, opt["ngf"] * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                                    opt["generator_norm"], opt["ngf"] * mult * 2),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** n_downsampling
        for i in range(opt["num_resnet_blocks_3d"]):
            model += [ResnetBlock3D(opt["ngf"] * mult, opt)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [*conv_norm_3d(nn.ConvTranspose3d(opt["ngf"] * mult, int(opt["ngf"] * mult / 2), kernel_size=3, stride=2, padding=1,
                                                       output_padding=1, bias=True), opt["generator_norm"], int(opt["ngf"] * mult / 2)),
                      nn.LeakyReLU(0.2, True)]
        model += [nn.ReflectionPad3d(3),
                  nn.Conv3d(opt["ngf"], output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock3D(nn.Module):
    """ Define a 3D Resnet block"""

    def __init__(self, dim, opt):
        super(ResnetBlock3D, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.ReflectionPad3d(1),
                       *conv_norm_3d(nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=True), self.opt["generator_norm"], dim),
                       nn.LeakyReLU(0.2, True),
                       nn.ReflectionPad3d(1),
                       *conv_norm_3d(nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=True), self.opt["generator_norm"], dim)]

        return nn.Sequential(*conv_block)  # unpacks the list into positional arguments

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


class PatchGANDiscriminator3D(nn.Module):
    """Create a PatchGAN discriminator"""

    def __init__(self, opt, input_nc, n_layers=3):
        """ Construct the 3D PatchGAN discriminator

        Parameters:
            input_nc (int)     -- the number of channels in input images
            ndf (int)          -- the number of filters in the first conv layer
            norm (str)         -- the type of normalization layers used in the network.

        Returns a discriminator
        """
        super(PatchGANDiscriminator3D, self).__init__()
        self.input_nc = input_nc
        self.n_layers = n_layers

        model = [nn.Conv3d(self.input_nc, opt["ndf"], kernel_size=4, stride=2, padding=1, bias=True),
                 nn.LeakyReLU(0.2, True)]

        # PatchGAN discriminator with receptive field of 70
        for i in range(self.n_layers-1):  # add downsampling layers
            mult = 2 ** i
            model += [*conv_norm_3d(nn.Conv3d(opt["ndf"] * mult, opt["ndf"] * mult * 2, kernel_size=4, stride=2, padding=1, bias=True),
                                    opt["discriminator_norm"],
                                    opt["ndf"] * mult * 2),
                      nn.LeakyReLU(0.2, True)]

        model += [*conv_norm_3d(nn.Conv3d(opt["ndf"] * (2**(self.n_layers-1)), opt["ndf"] * (2**self.n_layers),
                                          kernel_size=4, stride=1, padding=1, bias=True),
                                opt["discriminator_norm"],
                                opt["ndf"] * (2**self.n_layers)),
                  nn.LeakyReLU(0.2, True),
                  nn.Conv3d(opt["ndf"] * (2**self.n_layers), 1, kernel_size=4, stride=1, padding=1, bias=True)]  # 1 channel prediction output

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


"""
# 3D Image Data Synthesis.
# Copyright (C) 2021 D. Eschweiler, M. Rethwisch, M. Jarchow, S. Koppers, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please refer to the documentation for more information about the software
# as well as for installation instructions.
#
"""


class UNet3DPixelshuffle(nn.Module):
    """Implementation of the 3D U-Net architecture.
    """
    def __init__(self, opt, in_channels, out_channels):
        super(UNet3DPixelshuffle, self).__init__()
        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        self.feat_channels = opt["ngf"] # 16
        self.norm_method = opt["generator_norm"] # instance | batch | none
        
        if self.norm_method == 'instance':
            self.norm = nn.InstanceNorm3d
        elif self.norm_method == 'batch':
            self.norm = nn.BatchNorm3d
        elif self.norm_method == 'none':
            self.norm = nn.Identity
        else:
            raise ValueError('Unknown normalization method "{0}". Choose from "instance|batch|none".'.format(self.norm_method))
        
        
        # Define layer instances       
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels, self.feat_channels//2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels//2, self.feat_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d1 = nn.Sequential(
            nn.Conv3d(self.feat_channels, self.feat_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.c2 = nn.Sequential(
            nn.Conv3d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels, self.feat_channels*2, kernel_size=3, padding=1),
            self.norm(self.feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d2 = nn.Sequential(
            nn.Conv3d(self.feat_channels*2, self.feat_channels*2, kernel_size=4, stride=2, padding=1),
            self.norm(self.feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.c3 = nn.Sequential(
            nn.Conv3d(self.feat_channels*2, self.feat_channels*2, kernel_size=3, padding=1),
            self.norm(self.feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels*2, self.feat_channels*4, kernel_size=3, padding=1),
            self.norm(self.feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d3 = nn.Sequential(
            nn.Conv3d(self.feat_channels*4, self.feat_channels*4, kernel_size=4, stride=2, padding=1),
            self.norm(self.feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.c4 = nn.Sequential(
            nn.Conv3d(self.feat_channels*4, self.feat_channels*4, kernel_size=3, padding=1),
            self.norm(self.feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels*4, self.feat_channels*8, kernel_size=3, padding=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u1 = nn.Sequential(
            nn.Conv3d(self.feat_channels*8, self.feat_channels*8, kernel_size=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelShuffle3d(2),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c5 = nn.Sequential(
            nn.Conv3d(self.feat_channels*5, self.feat_channels*8, kernel_size=3, padding=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels*8, self.feat_channels*8, kernel_size=3, padding=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u2 = nn.Sequential(
            nn.Conv3d(self.feat_channels*8, self.feat_channels*8, kernel_size=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelShuffle3d(2),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c6 = nn.Sequential(
            nn.Conv3d(self.feat_channels*3, self.feat_channels*8, kernel_size=3, padding=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels*8, self.feat_channels*8, kernel_size=3, padding=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u3 = nn.Sequential(
            nn.Conv3d(self.feat_channels*8, self.feat_channels*8, kernel_size=1),
            self.norm(self.feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelShuffle3d(2),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c7 = nn.Sequential(
            nn.Conv3d(self.feat_channels*2, self.feat_channels, kernel_size=3, padding=1),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.out = nn.Sequential(
            nn.Conv3d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            self.norm(self.feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.feat_channels, out_channels, kernel_size=1)
            )
       
        self.out_fcn = nn.Tanh()

    def forward(self, img):
        
        c1 = self.c1(img)
        d1 = self.d1(c1)
        
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        
        c4 = self.c4(d3)
        
        u1 = self.u1(c4)
        c5 = self.c5(torch.cat((u1,c3),1))
        
        u2 = self.u2(c5)
        c6 = self.c6(torch.cat((u2,c2),1))
        
        u3 = self.u3(c6)
        c7 = self.c7(torch.cat((u3,c1),1))
        
        out = self.out(c7)
        if not self.out_fcn is None:
            out = self.out_fcn(out)
        
        return out
        
class PixelShuffle3d(nn.Module):
    '''
    reference: http://www.multisilicon.com/blog/a25332339.html
    '''
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale
    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3
        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, nOut, out_depth, out_height, out_width)