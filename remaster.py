
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse
import subprocess
import DeepRemaster.utils as utils

from interface import OP

class Remaster(OP):
    def __init__(self, refer_dir, outputdir, mindim=320, disable_restore=False):
        super(Remaster, self).__init__(outputdir)
        self._refer_dir = refer_dir
        self._disable_colorization = False
        self._disable_restore = disable_restore
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mindim = mindim
        self._refer_block = {'sec':None,'ref_imgs':None}

        if self._refer_dir!='none':
            assert os.path.exists(self._refer_dir), 'No such directory:{}'.format(self._refer_dir)
            import glob
            ext_list = ['png','jpg','bmp']
            reference_files = []
            for ext in ext_list:
                reference_files += glob.glob( self._refer_dir+'/*.'+ext, recursive=True )
            assert len(reference_files)>0, 'The directory is empty:{}'.format(self._refer_dir)
            self._all_refs = []
            for rf in reference_files:
                h,m,s = os.path.basename(rf).split('.')[0].split(':')
                time = int(h)*3600 + int(m)*60 + int(s)
                self._all_refs.append((rf, time))
            self._all_refs.sort(key=lambda x: x[1])

        # Load remaster network
        state_dict = torch.load( 'model_weights/remasternet.pth.tar' )
        if not self._disable_restore:
            modelR = __import__( 'DeepRemaster.model.remasternet', fromlist=['NetworkR'] ).NetworkR()
            modelR.load_state_dict( state_dict['modelR'] )
            self._modelR = modelR.to(self._device)
            self._modelR.eval()
        if not self._disable_colorization:
            modelC = __import__( 'DeepRemaster.model.remasternet', fromlist=['NetworkC'] ).NetworkC()
            modelC.load_state_dict( state_dict['modelC'] )
            self._modelC = modelC.to(self._device)
            self._modelC.eval()
        
    def _find_refer_block(self, proc_begin, proc_end):

        a = [1 if proc_begin-i[1] > 0 else 0 for i in self._all_refs]
        start_indx = max(sum(a)-1, 0)
        b = [1 if proc_end-i[1] > 0 else 0 for i in self._all_refs]
        end_indx  = min(len(self._all_refs), sum(b)+1)
        return start_indx, end_indx  # start from 0

    def _load_refer(self, proc_begin, proc_end):
        '''[proc_begin, proc_end] 闭区间
        parameter:
            proc_begin:float, unit second
            proc_end:float, unit second
        '''
        if (self._refer_block['sec'] is not None) and (self._refer_block['sec'][-1] >= proc_end):
            return self._refer_block['ref_imgs']
        if self._refer_dir!='none':
            i,j = self._find_refer_block(proc_begin, proc_end)
            aspect_mean = 0
            minedge_dim = 256
            refs = []
            self._refer_block['sec'] = []
            for v, t in self._all_refs[i:j]:
                refimg = Image.open( v ).convert('RGB')
                w, h = refimg.size
                aspect_mean += w/h
                refs.append( refimg )
                self._refer_block['sec'].append(t)
            aspect_mean /= len(self._refer_block['sec'])
            target_w = int(256*aspect_mean) if aspect_mean>1 else 256
            target_h = 256 if aspect_mean>=1 else int(256/aspect_mean)
            refimgs = torch.FloatTensor(len(self._refer_block['sec']), 3, target_h, target_w)
            for i, v in enumerate(refs):
                refimg = utils.addMergin( v, target_w=target_w, target_h=target_h )
                refimgs[i] = transforms.ToTensor()( refimg )
            self._refer_block['ref_imgs'] = refimgs.view(1, refimgs.size(0), refimgs.size(1), refimgs.size(2), refimgs.size(3)).to( self._device )
            return self._refer_block['ref_imgs']
    
        return None

    def exec(self, param_dict):

        # Load video
        indir = param_dict['inputdir']
        images_file = os.listdir(indir)
        nframes = len(images_file)
        assert nframes >0, 'there are no files in this directory:{}'.format(indir)
        v_w, v_h = Image.open(os.path.join(indir, images_file[0]) ).size
        minwh = min(v_w,v_h)
        scale = 1
        if (-1 != self._mindim) and (minwh != self._mindim):
            scale = self._mindim / minwh
        t_w = round(v_w*scale/16.)*16
        t_h = round(v_h*scale/16.)*16
        pbar = tqdm(total=nframes)
        block = 5

        # Process 
        with torch.no_grad():
            it = 0
            while True:
                frame_pos = it*block
                if frame_pos >= nframes:
                    break
                if block >= nframes-frame_pos:
                    proc_g = nframes-frame_pos
                else:
                    proc_g = block

                input = None
                gtC = None
                for i in range(proc_g):
                    index = frame_pos + i + 1
                    frame = cv2.imread(os.path.join(indir, '{}.png'.format(index)))
                    frame = cv2.resize(frame, (t_w, t_h))
                    nchannels = frame.shape[2]
                    if nchannels == 1 or not self._disable_colorization:
                        frame_l = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        frame_l = torch.from_numpy(frame_l).view( frame_l.shape[0], frame_l.shape[1], 1 )
                        frame_l = frame_l.permute(2, 0, 1).float() # HWC to CHW
                        frame_l /= 255.
                        frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
                    elif nchannels == 3:
                        frame = frame[:,:,::-1] ## BGR -> RGB
                        frame_l, frame_ab = utils.convertRGB2LABTensor( frame )
                        frame_l = frame_l.view(1, frame_l.size(0), 1, frame_l.size(1), frame_l.size(2))
                        frame_ab = frame_ab.view(1, frame_ab.size(0), 1, frame_ab.size(1), frame_ab.size(2))

                    input = frame_l if i==0 else torch.cat( (input, frame_l), 2 )
                    if nchannels==3 and self._disable_colorization:
                        gtC = frame_ab if i==0 else torch.cat( (gtC, frame_ab), 2 )
                
                input = input.to( self._device )

                # Perform restoration
                if not self._disable_restore:
                    output_l = self._modelR( input ) # [B, C, T, H, W]
                else:
                    output_l = input

                # Save restoration output without colorization when using the option [--disable_colorization]
                if self._disable_colorization:
                    for i in range( proc_g ):
                        index = frame_pos + i + 1 # frame id start from 1
                        out_fname = os.path.join(self.outputdir, '%d.png'%(index))
                        if nchannels==3:
                            out_l = output_l.detach()[0,:,i].cpu()
                            out_ab = gtC[0,:,i].cpu()
                            out = torch.cat((out_l, out_ab),dim=0).detach().numpy().transpose((1, 2, 0))
                            out = Image.fromarray( np.uint8( utils.convertLAB2RGB( out )*255 ) )
                            out.save( out_fname )
                        else:
                            save_image( output_l.detach()[0,:,i], out_fname, nrow=1 )
                # Perform colorization
                else:
                    start_t = frame_pos/float(param_dict['fps'])
                    end_t = (frame_pos + proc_g-1)/ float(param_dict['fps'])
                    refimgs = self._load_refer(start_t, end_t)
                    # print(start_t, end_t, self._refer_block['sec'])
                    if refimgs is None:
                        output_ab = self._modelC( output_l )
                    else:
                        output_ab = self._modelC( output_l, refimgs )
                    output_l = output_l.detach().cpu()
                    output_ab = output_ab.detach().cpu()
                    
                    # Save output frames of restoration with colorization
                    for i in range( proc_g ):
                        index = frame_pos + i + 1 
                        out_l = output_l[0,:,i,:,:]
                        out_c = output_ab[0,:,i,:,:]
                        output = torch.cat((out_l, out_c), dim=0).numpy().transpose((1, 2, 0))
                        output = Image.fromarray( np.uint8( utils.convertLAB2RGB( output )*255 ) )
                        output.save( os.path.join(self.outputdir, '%d.png'%index ) )

                it = it + 1
                pbar.update(proc_g)
        
        pbar.close()