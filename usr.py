
from interface import OP
import os.path
import cv2
import numpy as np
from tqdm import tqdm
import torch

from USRNet.utils import utils_deblur
from USRNet.utils import utils_sisr as sr
from USRNet.utils import utils_image as util

from USRNet.models.network_usrnet import USRNet as net

class USR(OP):
    ''' 
    parameters
        scale_factor: int, scale factor , only from {1,2,3,4}
        chip: bool,
            whether to remove the noise
        outputdir: str,
            where will the output be saved
    '''
    def __init__(self, scale_factor, chip, outputdir):
        super(USR, self).__init__(outputdir)
        model_name = 'usrgan'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
        self._sf = scale_factor
        self._n_channels = 3  # 3 for color image, 1 for grayscale image
        # ----------------------------------------
        # set noise level and kernel
        # ----------------------------------------
        if chip:
            noise_level_img = 15       # noise level for LR image, 15 for chip
            kernel_width_default_x1234 = [0.6, 0.9, 1.7, 2.2] # Gaussian kernel widths for x1, x2, x3, x4
        else:
            noise_level_img = 2       # noise level for LR image, 0.5~3 for clean images
            kernel_width_default_x1234 = [0.4, 0.7, 1.5, 2.0] # default Gaussian kernel widths of clean/sharp images for x1, x2, x3, x4

        noise_level_model = noise_level_img/255.  # noise level of model
        kernel_width = kernel_width_default_x1234[self._sf-1]

        # set your own kernel width
        k = utils_deblur.fspecial('gaussian', 25, kernel_width)
        k = sr.shift_pixel(k, self._sf)  # shift the kernel
        k /= np.sum(k)
        kernel = util.single2tensor4(k[..., np.newaxis])

        #load model
        model_path = os.path.join('model_weights', model_name+'.pth')
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'tiny' in model_name:
            model = net(n_iter=6, h_nc=32, in_nc=4, out_nc=3, nc=[16, 32, 64, 64],
                        nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
        else:
            model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
                        nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")

        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for key, v in model.named_parameters():
            v.requires_grad = False

        self._model = model.to(self._device)
        sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])
        self._sigma = sigma.to(self._device)
        self._kernel = kernel.to(self._device)

    def exec(self, param_dict):
        images = os.listdir(param_dict['inputdir'])
        for img in tqdm(images):
            filename = os.path.join(param_dict['inputdir'], img)
            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_L = util.imread_uint(filename, n_channels=self._n_channels)
            img_L = util.uint2single(img_L)

            w, h = img_L.shape[:2]

            # boundary handling
            boarder = 8     # default setting for kernel size 25x25
            img = cv2.resize(img_L, (self._sf*h, self._sf*w), interpolation=cv2.INTER_NEAREST)
            img = utils_deblur.wrap_boundary_liu(img, [int(np.ceil(self._sf*w/boarder+2)*boarder), int(np.ceil(self._sf*h/boarder+2)*boarder)])
            img_wrap = sr.downsample_np(img, self._sf, center=False)
            img_wrap[:w, :h, :] = img_L
            img_L = img_wrap

            img_L = util.single2tensor4(img_L)
            img_L = img_L.to(self._device)

            # (2) img_E
            img_E = self._model(img_L, self._kernel, self._sf, self._sigma)
            img_E = util.tensor2uint(img_E)[:self._sf*w, :self._sf*h, ...]
            util.imsave(img_E, os.path.join(self.outputdir, '{}.png'.format(img_name) ) )
