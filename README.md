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

![image](https://user-images.githubusercontent.com/73570008/151317428-61d763dc-6b0b-4355-af73-95eb45a7fd76.png)

We also tested on Casia II, as shown in above table, CTS-Net trained by Photoshop processed samples performed better in F1, Precision and Precision than before.

# More results on Columbia and In-The-Wild dataset
## Columbia
![image](https://user-images.githubusercontent.com/73570008/151368152-55c863bc-25b7-4d3e-8f4d-d767f48a089f.png)

## In-The-Wild
As shown in the image below, false positives occur in tampered band prediction but not in tampered edges prediction. It shows the refined stage discriminates the tampered band prediction more strictly to avoid false detection as much as possible.
![image](https://user-images.githubusercontent.com/73570008/151360172-e3f5f368-0f36-4422-b532-3c956d34dea2.png)

# HS dataset
We generate a deceptive and highly simulated dataset (HS dataset) for tampered edges detection task. The size of every image is 320*320. Thank to the special annotation, HS dataset can not only be used for tampered edges detection task, but also can be applied to the task of tampered area detection. However, there is ambiguity problem in the latter task, we do not recommend using HS dataset in that way.
![image](https://user-images.githubusercontent.com/73570008/151368997-26baaaea-5a0f-410a-a695-b5c867045c64.png)
As shown in above image, there are four parts in the corresponding mask. Part A, B, C and D has the value of 50, 255, 100 and 0 respectively. Part B is the inner edge, and C is the outer edge, they jointly form the tampered edges. In CTS-Net, we take them as detection target and generate ground truth from that, set area B and area C to 255, and other parts to 0. 
In addition, if you want to use HS dataset in tampered area detection and localization task, though we do not recommend that, the only thing you need to do is setting part A and B to 255, other parts to 0. We believe HS dataset with deceptive and highly simulated samples will help in your work on forgery detection task.

