# 老视频、老照片修复工程
工程结合DAIN、USRNet、DeepRemaster三个工程分别实现插帧、提升分辨率以及黑白视频上色。

## 安装
```bash
./install.sh
```
## 运行
1. 激活虚拟环境
    ```bash
    conda activate VideoPhotoRepair
    ```
1. 操作注意事项
   - 使用`python main.py -h`命令查看帮助
   - refer_dir路径下的参考图片请参考demo中的**hh:mm:ss.png**方式命名。h、m、s分别代表时、分、秒。该时间是指在要处理的视频中那一段需要参考该图片，并按照该段视频时间中点命名参考图片。即按照视频中与参考图片最相似的帧的时间为参考图片命名。
   - 如果input是文件夹模式，请保证文件夹内的视频帧序列命名方式按'%d.png'方式命名，并从1开始。