# CPGAN-CVPR2020

Dear friends, Thank you for keep tracking in this implementation of CPGAN (CVPR 2020 Oral Paper)

# Prerequisites:

Python 3.6

Pytorch 1.0 or newer (Pytorch > 0.4 should be ok)

matplotlib

skimage

# Train: 

python train.py

Change the option in Train.py to set the dataset's directory. I am using Multi-PIE as the training set. 


# Test

python test.py

# Citation

If you find CPGAN useful in your research, please consider citing:

@inproceedings{zhang2020copy,
  title={Copy and Paste GAN: Face Hallucination from Shaded Thumbnails},
  author={Zhang, Yang and Tsang, Ivor W and Luo, Yawei and Hu, Chang-Hui and Lu, Xiaobo and Yu, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7355--7364},
  year={2020}
}
