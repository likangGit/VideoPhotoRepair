#!/bin/sh
workdir=$(cd $(dirname $0); pwd)

# #1. install ffmpeg
# apt install ffmpeg
# #2. create env
# conda create -f env.yaml
# #3. build package of DAIN
# cd ${workdir}/DAIN/my_package
# cp build.sh build_tmp.sh
# sed -i "s/pytorch1.0.0/VideoPhotoRepair/g" build_tmp.sh
# output=`./build_tmp.sh`
# cd ../PWCNet/correlation_package_pytorch1_0
# cp build.sh build_tmp.sh
# sed -i "s/pytorch1.0.0/VideoPhotoRepair/g" build_tmp.sh
# output=`./build_tmp.sh`

cd ${workdir}/DAIN
mkdir model_weights
cd model_weights
wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth

