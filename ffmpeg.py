import os
import re
import subprocess
from interface import OP

class ToFrame(OP):
    def __init__(self, outputdir):
        super(ToFrame, self).__init__(outputdir)
    
    def exec(self, param_dict):
        fps = -1
        cmd = 'ffmpeg -i {} -f image2 {}'.format(param_dict['inputdir'], os.path.join(self.outputdir, '%d.png'))
        p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        while p.poll() is None:
            line = p.stdout.readline().rstrip()
            end = '\r' if (('frame=' in line) and ('fps=' in line) and ('time=' in line)) else '\n'
            print(line, end=end)
            if ('Stream' in line) and ('tbr' in line):
                matchObj = re.match(r'Stream .*kb/s, (\d+) fps,.* tbr,.*', line.strip() )
                if matchObj:
                    fps = round(float(matchObj.group(1)))
        assert fps != -1, 'Get fps failed'
        param_dict['fps'] = fps

class ToVideo(OP):
    def __init__(self, outputdir):
        super(ToVideo, self).__init__(outputdir)

    def exec(self, param_dict):
        cmd = 'ffmpeg -r {} -f image2 -i {} -vcodec libx264 {} -y'.format(
            param_dict['fps'],
            os.path.join(param_dict['inputdir'], '%d.png'),
            os.path.join(self.outputdir, 'result.mp4')
            )
        print(cmd)
        subprocess.call(cmd, shell=True)
        # p = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
        # while p.poll() is None:
        #     line = p.stdout.readline().rstrip()
        #     end = '\r' if (('frame=' in line) and ('fps=' in line) and ('time=' in line)) else '\n'
        #     print(line, end=end)