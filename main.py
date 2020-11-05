import os
import re
import sys
sys.path.append('DAIN')
import warnings
warnings.filterwarnings('ignore')
import subprocess
import argparse
from dain import DAIN



def printInfo(message):
    info = '======================{}========================='.format(message)
    print(info)

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='The file you want to process, \
                    It should be a file')
parser.add_argument('--operation', type=str, nargs='+', 
                    help="what perations you want execute. You can choose one,two or three \
                    items from choices,regardless of their priority. default:[dain,usr,remaster]",
                    default=['dain', 'usr', 'remaster'], choices=['dain', 'usr', 'remaster'])
dain_group = parser.add_argument_group(title='DAIN option')
usr_group = parser.add_argument_group(title='USRNet option')
dremaster_group = parser.add_argument_group(title='DeepRemaster group')
dain_group.add_argument('--multi', type=int, help="what's the desired multiple during interpolation",
                    default=2, choices=[2,4,8,16])
dain_group.add_argument('--save_which', type=int, help="choose which result to save: 0 ==> interpolated, 1==> rectified",
                    default=1, choices=[0, 1])
args = parser.parse_args()

assert os.path.exists(args.input), "No such file:{}".format(args.input)
assert len(args.operation) <= 3, "Too many operation. The number of operatoin should be less than or equal 3. "

temp_folder = 'temp'
if not os.path.exists(temp_folder):
    os.mkdir(temp_folder)

# 1. export frame from video by using ffmpeg command
printInfo('frame export start')
temp_frame_folder = os.path.join(temp_folder, 'frame')
if not os.path.exists(temp_frame_folder):
    os.mkdir(temp_frame_folder)
fps = -1
cmd = 'ffmpeg -i {} -f image2 {}'.format(args.input, os.path.join(temp_frame_folder, '%d.png'))
p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
while p.poll() is None:
    line = p.stdout.readline().rstrip()
    end = '\r' if (('frame=' in line) and ('fps=' in line) and ('time=' in line)) else '\n'
    print(line, end=end)
    if ('Stream' in line) and ('tbr' in line):
        matchObj = re.match(r'Stream .*kb/s, (\d+) fps,.* tbr,.*', line.strip() )
        if matchObj:
            fps = round(float(matchObj.group(1)))
printInfo('frame export finish!')
assert fps != -1, 'Get fps failed'
if 'dain' in args.operation:
    fps = args.multi * fps

# 2. execute operation
op_dict = {'dain':DAIN, 'usr':'', 'remaster': ''}
op_param_dict = {'dain':{'timestep':1/float(args.multi),
                        'outputdir':os.path.join(temp_folder, 'dain'),
                        'save_which':args.save_which}, 
                'usr':{'outputdir':os.path.join(temp_folder, 'usr')
                        },
                'remaster':{'outputdir':os.path.join(temp_folder, 'remaster')
                        }
                }
inputdir = temp_frame_folder
for op in args.operation:
    printInfo('{} operation start'.format(op))
    param = op_param_dict[op]
    op_instance = op_dict[op](**param)
    op_instance.inputdir = inputdir
    op_instance.exec()
    inputdir = op_instance.outputdir
    printInfo('{} operation finish!'.format(op))

# use the frame create a video
printInfo('create video start')
cmd = 'ffmpeg -r {} -f image2 -i {} result.mp4 -y'.format(fps, os.path.join(inputdir, '%08d.png'))
p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
while p.poll() is None:
    line = p.stdout.readline().rstrip()
    end = '\r' if (('frame=' in line) and ('fps=' in line) and ('time=' in line)) else '\n'
    print(line, end=end)
printInfo('all done')




