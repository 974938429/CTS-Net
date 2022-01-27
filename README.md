# CTS-Net
The links of the CTS-Net parameters is (on BaiduNetdisk) : https://pan.baidu.com/s/1NEhBMa2DEf6q5Pw4ME6xFg 
Extraction code:x3qw

And HS dataset is (on BaiduNetdisk) : https://pan.baidu.com/s/1L6ZKSJcPUg1KNX9hc606cw?pwd=sqrd
Extraction code:sqrd
We generate 2.8w images on COCO and Casia I (casia_au_and_casia_template_after_divide, coco_casia_template_after_divide, coco_cm and coco_sp), 2.8w images on COD10K and STEX (COD10K and periodic_texture), and 1.6W images are post-processing (COD10K_new_tampered is processed by Photoshop, blur_data is processed by Gaussian blur, and casia_au_and_casia_jpegcompress is processed JPEG compression), and 7.2w images in total. Negative is image without splicing and corresponding ground truth is all-black image, you can use negative_gt.bmp as well. In the public_dataset, we use Casia and Coverage for testing. 

# Notation:
Please download the parameters of model first, and put the 'parameters' folder same with other folders, and then run the PY file in the test folder. During the running process, the results of the two stages can be presented, and the final results can be found in the 'results' folder.

# Feathering effect
![image](https://user-images.githubusercontent.com/73570008/151310727-02e5af0a-afdc-43d1-96b7-d25a1a961ce1.png)
In the above image, the spider is cropped by Photoshop. When you zoom in locally, there is a translucent edge around contour. When you paste the spider in image, the translucent edge will overlap with the authentic pixels at the corresponding position. If neighboring pixels have similar colors, the translucent edge will weaken the unnatural transition between two parts, which is the feathering effect.
CTS-Net detects splicing trace with double-edge representation, in other words, ours focuses on the unnatural transition of pixel mutation. Therefore, splicing image processed by Photoshop is a great challenge for CTS-Net. We code to utilize Photoshop software and synthesize 3037 training data images and 2026 test images to try CTS-Net as hard samples.

 Scheme  | F1  | Precision | Recall
 ---- | ----- | ------  
 

CTS-Net (with PS samples) | 0.3324 | 0.4591 | 0.3082

CTS-Net | 0.2969 | 0.4187 | 0.2716

![image](https://user-images.githubusercontent.com/73570008/151317428-61d763dc-6b0b-4355-af73-95eb45a7fd76.png)

We also tested on Casia II, as shown in above table, CTS-Net trained by Photoshop processed samples performed better in F1, Precision and Precision than before.
# More results on Columbia and In-The-Wild
