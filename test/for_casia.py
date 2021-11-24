import torch
import torch.optim as optim
import torch.utils.data.dataloader
import os, sys

sys.path.append('../')
sys.path.append('./')
import argparse
from model.unet_two_stage_model_0719_new8map import UNetStage1 as Net1
from model.unet_two_stage_model_0719_new8map import UNetStage2 as Net2
from PIL import Image
import rich
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
import numpy as np

parser = argparse.ArgumentParser(description='CTS-Net testing for COVERAGE samples')
parser.add_argument('--resume', default=[
    '../parameters/the_coarse_stage_casia_finetune.pth',
    '../parameters/the_refined_stage_casia_finetune.pth'
], type=list, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--src_path',default='../samples/casia')
parser.add_argument('--save_path',default='../results/casia')
args = parser.parse_args()

def main():
    #load the parameters of model
    model1 = Net1()
    model2 = Net2()
    if torch.cuda.is_available():
        model1.cuda()
        model2.cuda()
        if isfile(args.resume[0]) and isfile(args.resume[1]):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint1 = torch.load(args.resume[0])
            checkpoint2 = torch.load(args.resume[1])
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print('The model does not load any available parameters!!!')
    else:
        model1.cpu()
        model2.cpu()
        if isfile(args.resume[0]) and isfile(args.resume[1]):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint1 = torch.load(args.resume[0],map_location=torch.device('cpu'))
            checkpoint2 = torch.load(args.resume[1],map_location=torch.device('cpu'))
            model1.load_state_dict(checkpoint1['state_dict'])
            model2.load_state_dict(checkpoint2['state_dict'])
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print('The model does not load any available parameters!!!')

    #load coverage samples
    if os.path.exists(args.src_path):
        for index,item in enumerate(os.listdir(args.src_path)):
            src_dir=os.path.join(args.src_path,item)
            img=Image.open(src_dir).convert('RGB')
            if len(img.split()) != 3:
                rich.print(src_dir, 'error')
                continue
            img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.47, 0.43, 0.39), (0.27, 0.26, 0.27)),
            ])(img)
            img=img.unsqueeze(0)
            if torch.cuda.is_available():
                img=img.cuda()
            coarse_stage_output=model1(img)
            rgb_pred_rgb = torch.cat((coarse_stage_output[0], img), 1)
            refined_stage_output = model2(rgb_pred_rgb, coarse_stage_output[1], coarse_stage_output[2], coarse_stage_output[3])

            #Visual display
            outputs_1 = coarse_stage_output[0]
            outputs_1 = outputs_1.permute(0, 2, 3, 1)
            outputs_2 = refined_stage_output[0]
            outputs_2 = outputs_2.permute(0, 2, 3, 1)
            outputs_1 = outputs_1.cpu().detach().numpy()
            outputs_2 = outputs_2.cpu().detach().numpy()
            pred_mask_1 = outputs_1[0]
            pred_mask_1 = pred_mask_1.squeeze(2)
            pred_mask_1 = np.where(pred_mask_1 > 0.5, 255, 0)
            pred_mask_2 = outputs_2[0]
            pred_mask_2 = pred_mask_2.squeeze(2)
            pred_mask_2 = np.where(pred_mask_2 > 0.5, 255, 0)
            plt.subplot(1, 2, 1)
            plt.imshow(pred_mask_1)
            plt.title('The coarse stage prediction.')
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask_2)
            plt.title('The refined stage prediction.')
            plt.show()
            plt.clf()

            #Save
            pred_mask_1 = Image.fromarray(pred_mask_1).convert('L')
            pred_mask_2 = Image.fromarray(pred_mask_2).convert('L')
            pred_save_path_1 = os.path.join(args.save_path, 'stage1')
            pred_save_path_2 = os.path.join(args.save_path, 'stage2')
            if not os.path.exists(pred_save_path_1):
                os.makedirs(pred_save_path_1)
            pred_mask_1.save(os.path.join(pred_save_path_1, item))
            if not os.path.exists(pred_save_path_2):
                os.makedirs(pred_save_path_2)
            pred_mask_2.save(os.path.join(pred_save_path_2, item))





if __name__=='__main__':
    main()

