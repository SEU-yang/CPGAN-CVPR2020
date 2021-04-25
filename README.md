# CPGAN-CVPR2020

Dear friends, Thank you for keep tracking in this implementation of CPGAN (CVPR 2020 Oral Paper)
! https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Copy_and_Paste_GAN_Face_Hallucination_From_Shaded_Thumbnails_CVPR_2020_paper.html

## Prerequisites:

- Python 3.7
- Pytorch 1.5 or newer
- matplotlib
- skimage

## Train: 

python train.py

Change the option in train.py to set the dataset's directory. I am using Multi-PIE and CelebA as training sets. 

## Test

python test.py

Pre-trained models: 

`Multi-PIE` [BaiduYun] (Link: https://pan.baidu.com/s/1eve7GN_VXXJk5fxyT6LE9g)  Password: 1234

`CelebA`
Link: https://pan.baidu.com/s/1MPXR9Jb--c-CXu47YWmCIg Password: 1234


## Citation

If you find CPGAN useful in your research, please consider citing:
```
@inproceedings{zhang2020copy,
  title={Copy and Paste GAN: Face Hallucination from Shaded Thumbnails},
  author={Zhang, Yang and Tsang, Ivor W and Luo, Yawei and Hu, Chang-Hui and Lu, Xiaobo and Yu, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7355--7364},
  year={2020}
}
```
