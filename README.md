# Preparation
## environment
```shell
pip install -r requirements.txt
```
## dataset
The train data is from ImageNet, to download it you can go [`ImageNet`](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)
It is recommaned to put the data file in the dataset folder like
```
dataset/
    imagenet_val/
        pic1
        pic2
        ...
```
```
python data_split.py # Run this command when the data is ready, which is to partition the data set
```

# Train & Evaluation
There are a number of options that can be set, most of which can be used by default, which you can view in train files.
## for train
```
python train_simclr.py --num <number of train data> # if you want to train for self-supervised learning part

python train_simclr_cls.py --pth <model_ckp_path(self-supervised)> # training the classifier for for supervised learning

python train_rn18.py # finetune the ResNet model    "--pretrained" will load the weight pretrained in ImageNet

```


## for evaluation
```
python eval.py --file <model_ckp_path> # test the self-supervised learning method
```
The weights after model training can be downloaded [`here`](https://drive.google.com/drive/folders/1ghCX_HGWdNnL-1fp6scA7UgySNa11io7?usp=drive_link)


