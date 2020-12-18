import time
import os
from torch.autograd import Variable
import torch
import random
import numpy as np
from tqdm import tqdm

from DAIN import networks
from scipy.misc import imread, imsave
from DAIN.AverageMeter import  *
import shutil
from interface import OP

class DAIN(OP):
    '''
    parameter:
        inputdir: str. 
            it should be a folder which contains all the frame of a video named by integer
        outputdir: str.
            a folder for saving the output
        save_which: int, 0 or 1
            choose which result to save: 0 ==> interpolated, 1==> rectified
    '''
    def __init__(self, timestep, outputdir, save_which):
        super(DAIN, self).__init__(outputdir)
        self._timestep = timestep
        self._save_which = save_which

    def exec(self, param):
        inputdir = param['inputdir']
        assert os.path.exists(inputdir), 'No such file:{}'.format(inputdir)

        torch.backends.cudnn.benchmark = True # to speed up the

        model = networks.__dict__['DAIN_slowmotion'](    channel=3,
                                            filter_size=4 ,
                                            timestep=self._timestep,
                                            training=False)

        if torch.cuda.is_available():
            model = model.cuda()

        weight_file = 'model_weights/dain.pth'
        assert os.path.exists(weight_file), 'No such file:{}'.format(weight_file)
        print("The testing model weight is: " + weight_file)
        if not torch.cuda.is_available():
            pretrained_dict = torch.load(weight_file, map_location=lambda storage, loc: storage)
            # model.load_state_dict(torch.load(weight_file, map_location=lambda storage, loc: storage))
        else:
            pretrained_dict = torch.load(weight_file)
            # model.load_state_dict(torch.load(weight_file))

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. release the pretrained dict for saving memory
        pretrained_dict = []


        model = model.eval() # deploy mode
        use_cuda=torch.cuda.is_available()

        inputfiles = os.listdir(inputdir)
        

        tot_timer = AverageMeter()
        proc_timer = AverageMeter()
        end = time.time()
        output_pos = 1
        for input_pos in tqdm(range(1,len(inputfiles)) ): 
            arguments_strFirst = os.path.join(inputdir, '{}.png'.format(input_pos))
            arguments_strSecond = os.path.join(inputdir, '{}.png'.format(input_pos+1))
            dtype = torch.cuda.FloatTensor
            X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
            X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))
            assert (X0.size(0) == X1.size(0) == 3)

            intWidth = X0.size(2)
            intHeight = X0.size(1)

            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

            torch.set_grad_enabled(False)
            X0 = Variable(torch.unsqueeze(X0,0))
            X1 = Variable(torch.unsqueeze(X1,0))
            X0 = pader(X0)
            X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()
            proc_end = time.time()
            y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
            y_ = y_s[self._save_which]

            proc_timer.update(time.time() -proc_end)
            tot_timer.update(time.time() - end)
            end  = time.time()
            # print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
            if use_cuda:
                if not isinstance(y_, list):
                    y_ = y_.data.cpu().numpy()
                else:
                    y_ = [item.data.cpu().numpy() for item in y_]
            else:
                if not isinstance(y_, list):
                    y_ = y_.data.numpy()
                else:
                    y_ = [item.data.numpy() for item in y_]

            y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                                    intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]

            shutil.copy(arguments_strFirst, os.path.join(self.outputdir,"{}.png".format(output_pos)))
            output_pos  = output_pos + 1
            for item in y_:
                arguments_strOut = os.path.join(self.outputdir,"{}.png".format(output_pos))
                output_pos = output_pos + 1
                imsave(arguments_strOut, np.round(item).astype(np.uint8))
        # copy the last frame as the network output
        numFrames = int(1.0 / self._timestep)
        for _ in range( numFrames):
            shutil.copy(arguments_strSecond, os.path.join(self.outputdir, "{}.png".format(output_pos)))
            output_pos = output_pos + 1
        param['fps'] = int(1/self._timestep) * param['fps']


         