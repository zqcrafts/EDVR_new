# DVD数据集
# wget https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip
# unzip DeepVideoDeblurring_Dataset.zip

# REDS数据集
# wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_sharp.zip
# wget -c http://data.cv.snu.ac.kr:8008/webdav/dataset/REDS/train_blur.zip
# unzip train_sharp.zip
# unzip train_blur.zip

# 环境
conda create --name deblur python=3.6.8
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install pyyaml
pip install opencv-python
pip install scikit-image

git clone git@github.com:zqcrafts/EDVR_new.git

cd EDVR
python create_txt.py --input ../data/REDS/train/train_blur --target ../data/REDS/train/train_sharp --output ../data/REDS/train
python create_txt.py --input ../data/REDS/val/val_blur --target ../data/REDS/val/val_sharp --output ../data/REDS/val
python test.py -opt options/test/test_EDVR_L_deblur_REDS_zehui.yml
#python train.py -opt options/train/train_EDVR_M_zehui.yml

