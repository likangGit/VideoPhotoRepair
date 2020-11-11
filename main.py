import os
import re
import sys
sys.path += ['DAIN','USRNet']
import shutil
import warnings
warnings.filterwarnings('ignore')
import subprocess
import argparse
from dain import DAIN
from ffmpeg import ToVideo, ToFrame
from usr import USR


def printInfo(message):
    info = '======================{}========================='.format(message)
    print(info)

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='The file you want to process, \
                    It should be a file or a folder')
parser.add_argument('--operation', type=str, nargs='+', 
                    help="what perations you want execute. You can choose one,two or three \
                    items from choices,regardless of their priority. default:[dain,usr,remaster]",
                    default=['dain', 'usr', 'remaster'], choices=['dain', 'usr', 'remaster'])
parser.add_argument('--clc',help='whether to clean the results folder', action='store_true',default=False)

dain_group = parser.add_argument_group(title='DAIN option')
usr_group = parser.add_argument_group(title='USRNet option')
dremaster_group = parser.add_argument_group(title='DeepRemaster group')
dain_group.add_argument('--multi', type=int, help="what's the desired multiple during interpolation",
                    default=2, choices=[2,4,8,16])
dain_group.add_argument('--save_which', type=int, help="choose which result to save: 0 ==> interpolated, 1==> rectified",
                    default=1, choices=[0, 1])

usr_group.add_argument('--scale', help='scale factor , only from {1,2,3,4}',type=int, default=1,choices=[1,2,3,4] )
usr_group.add_argument('--chip', help='whether to remove noise', action='store_true', default=False)

dremaster_group.add_argument('--refer_dir',  type=str, default='none', help='Path to the reference image directory')
dremaster_group.add_argument('--mindim',     type=int,   default='320',    help='Length of minimum image edges')
args = parser.parse_args()

assert os.path.exists(args.input), "No such file:{}".format(args.input)
assert len(args.operation) <= 3, "Too many operation. The number of operatoin should be less than or equal 3. "

result_folder = 'results'
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
else:
    if args.clc:
        shutil.rmtree(result_folder)
        os.mkdir(result_folder)

op_dict = {'toframe':ToFrame, 'dain':DAIN, 'usr':USR, 'remaster': '', 'tovideo':ToVideo}
op_init_dict = {'toframe':{'outputdir': os.path.join(result_folder, 'frame')},
                'dain':{'timestep':1/float(args.multi),
                        'outputdir':os.path.join(result_folder, 'dain'),
                        'save_which':args.save_which}, 
                'usr':{'outputdir': os.path.join(result_folder, 'usr'),
                        'scale_factor': args.scale,
                        'chip': args.chip
                        },
                'remaster':{'outputdir':os.path.join(result_folder, 'remaster')
                        },
                'tovideo':{'outputdir':result_folder}
                }
exec_param = {'inputdir':args.input, 'fps': -1}
if os.path.isfile(args.input):
    ops = ['toframe',] + args.operation + ['tovideo',]
else:
    ops = args.operation

for op in ops:
    printInfo('{} operation start'.format(op))

    param = op_init_dict[op]
    op_instance = op_dict[op](**param)
    op_instance.exec(exec_param)
    exec_param['inputdir'] = op_instance.outputdir

    printInfo('{} operation finish!'.format(op))


printInfo('all done')




