#### simple usage
```
cd openAPD

pip install -v -e .
```
#### prepare datasets
```
LEVIR-CD：https://justchenhao.github.io/LEVIR/

DSIFN-CD: https://www.dropbox.com/s/1lr4m70x8jdkdr0/DSIFN-CD-256.zip?dl=0

WHU-CD：http://study.rsgis.whu.edu.cn/pages/download/
```
establish soft connection
```
ln -s /datasets_address  /openAPD/data/LEVIR-CD
```
train
```
python tools/train.py configs/pcam/pcam_r18_512x512_60k_levircd.py  --work-dir ./pcam_r18_levir_workdir  --gpu-id 0  --seed 307
```
infer
```
# get .png results
python tools/test.py configs/pcam/pcam_r18_512x512_60k_levircd.py  pcam_r18_levir_workdir/latest.pth --format-only --eval-options "imgfile_prefix=tmp_infer"
# get metrics
python tools/test.py configs/pcam/pcam_r18_512x512_60k_levircd.py  pcam_r18_levir_workdir/latest.pth --eval mFscore mIoU
```

