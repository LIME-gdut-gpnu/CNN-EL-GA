#coding=UTF-8
import os
import shutil
import sys
from glob import glob
from scipy import misc
import numpy as np
import nibabel as nib
import time
import math
import random

#进行灰度归一化
def gray_norm(img):
    maxValue = np.max(img)
    minValue = np.min(img)
    img = (img - minValue) / (maxValue - minValue)
    return img

def clip_to_target_size(img,target_size):
    new_img = np.zeros(target_size)
    #原图像的裁剪部分
    x_begin=max(math.floor(img.shape[0]/2)-math.floor(target_size[0])/2,0)
    x_end=  min(x_begin+target_size[0],img.shape[0])
    y_begin=max(math.floor(img.shape[1]/2)-math.floor(target_size[1])/2,0)
    y_end=  min(y_begin+target_size[1],img.shape[1])
    z_begin=max(math.floor(img.shape[2]/2)-math.floor(target_size[2])/2,0)
    z_end=  min(z_begin+target_size[2],img.shape[2])
    #目标图像的填充部分
    x_b=max(math.floor(target_size[0]/2)-math.floor(img.shape[0]/2),0)
    x_e=min(x_b+img.shape[0],target_size[0])
    y_b=max(math.floor(target_size[1]/2)-math.floor(img.shape[1]/2),0)
    y_e=min(y_b+img.shape[1],target_size[1])
    z_b=max(math.floor(target_size[2]/2)-math.floor(img.shape[2]/2),0)
    z_e=min(z_b+img.shape[2],target_size[2])

    #保存剪切后图像的三维矩阵
    new_img[x_b:x_e,y_b:y_e,z_b:z_e]=img[x_begin:x_end,y_begin:y_end,z_begin:z_end]
    return new_img

def save_slice(img,axis,position,savename):
    if axis=='X':
        slice = img[position,:,:]
    elif axis=='Y':
        slice = img[:,position,:]
    elif axis=='Z':
        slice = img[:,:,position]
    else:
        print('error:no coordinate axis is specified,no slices generated ')
        exit()
    misc.imsave(savename, slice)  # 保存图片

def create_and_clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)

def create_classify_data(slices_base_dir,slices_pos_all):
    #这四个路径是存放4类病人切片的目录
    file_base_dir1 = os.path.join(slices_base_dir, 'AD')
    file_base_dir2 = os.path.join(slices_base_dir, 'NC')
    file_base_dir3 = os.path.join(slices_base_dir, 'MCIc')
    file_base_dir4 = os.path.join(slices_base_dir, 'MCInc')
    # 这三个路径是存放3种分类实验所需数据的目录，每个目录里包含两种病人图片
    classify_base_dir1 = os.path.join(slices_base_dir, 'AD_NC')
    classify_base_dir2 = os.path.join(slices_base_dir, 'MCIc_NC')
    classify_base_dir3 = os.path.join(slices_base_dir, 'MCIc_MCInc')

    file_base_paths=[file_base_dir1,file_base_dir2,file_base_dir3,file_base_dir4]
    classify_base_paths=[classify_base_dir1,classify_base_dir2,classify_base_dir3]

    # 这里的切片坐标和create_and_save_slices里的相同，表示要从每个nii图像中取哪几个切片，从三个不同的方向
    slices_pos_all=slices_pos_all
    save_dir_prefixes = ['X', 'Y', 'Z']
    # 如果该标志为1，则清空classify文件夹
    clear_classify_fold = 1
    target_folders = []
    # 第一次时清空classify_fold,创建目录结构
    if clear_classify_fold == 1:
        for i in range(len(classify_base_paths)):
            for j in range(len(save_dir_prefixes)):
                current_path = os.path.join(file_base_paths[i], save_dir_prefixes[j])
                current_path_files_num = len([name for name in os.listdir(current_path)])
                for k in range(current_path_files_num):
                    target_folders.append(os.path.join(classify_base_paths[i], save_dir_prefixes[j], 'group'+str(k), 'slices_all'))

        for p in range(len(target_folders)):
            create_and_clean_dir(target_folders[p])

    for i in range(len(file_base_paths)):
        for j in range(len(save_dir_prefixes)):
            current_change_path = os.path.join(file_base_paths[i], save_dir_prefixes[j])
            current_change_path_files_num = len([name for name in os.listdir(current_change_path)])
            for k in range(current_change_path_files_num):
                # AD,HC,MCIc,MCInc四个目录下的子目录
                file_path = os.path.join(current_change_path, 'group'+str(k))
                # 3组分类实验所需的数据目录
                classify1_data_path = os.path.join(classify_base_paths[0], save_dir_prefixes[j], 'group'+str(k), 'slices_all')
                classify2_data_path = os.path.join(classify_base_paths[1], save_dir_prefixes[j], 'group'+str(k), 'slices_all')
                classify3_data_path = os.path.join(classify_base_paths[2], save_dir_prefixes[j], 'group'+str(k), 'slices_all')

                # 检查数据目录是否存在
                if not os.path.exists(file_path):
                    print('No slices in path ' + file_path)
                    exit()

                # 读取文件夹下的所有jpg
                files = glob(os.path.join(file_path, '*.jpg'))

                # 将选取的图片分别复制到trains, tests文件夹下
                for file in files:
                    # 另外将这些图片复制到相应的classify目录下##################
                    # 将AD里的图片复制到AD_HC下
                    # 将HC里的图片复制到AD_HC和MCIc_HC下
                    # 将MCIc里的图片复制到MCIc_HC和MCIc_MCInc下
                    # 将MCInc里的图片复制到MCIc_MCInc下
                    if i == 0:
                        shutil.copy(file, classify1_data_path)
                    elif i == 1:
                        shutil.copy(file, classify1_data_path)
                        shutil.copy(file, classify2_data_path)
                    elif i == 2:
                        shutil.copy(file, classify2_data_path)
                        shutil.copy(file, classify3_data_path)
                    elif i == 3:
                        shutil.copy(file, classify3_data_path)
                print('slices in path ' + file_path + ' have processed successfully')



def create_and_save_slices(data_base_path,slices_base_dir,slices_pos_all):
    #nii文件主目录
    data_base_path=data_base_path
    #切片保存主目录
    save_base_path = slices_base_dir
    #各个类别病人的nii文件分目录
    #
    # path1=os.path.join(data_base_path,'AD','mri')
    # path2=os.path.join(data_base_path,'CN','mri')
    # path3=os.path.join(data_base_path,'MCI_c','mri')
    # path4=os.path.join(data_base_path,'MCI_nc','mri')
    path1=os.path.join(data_base_path,'AD')
    path2=os.path.join(data_base_path,'NC')
    path3=os.path.join(data_base_path,'MCIc')
    path4=os.path.join(data_base_path,'MCInc')

    save_base_dir1=os.path.join(save_base_path,'AD')
    save_base_dir2=os.path.join(save_base_path,'NC')
    save_base_dir3=os.path.join(save_base_path,'MCIc')
    save_base_dir4=os.path.join(save_base_path,'MCInc')

    data_paths=[path1,path2,path3,path4]
    save_paths=[save_base_dir1,save_base_dir2,save_base_dir3,save_base_dir4]

    target_size_x = [121, 145, 145]
    target_size_y = [145, 145, 145]
    target_size_z = [145, 145, 121]

    #存放所有切片坐标，（三个方向，每个方向用一个数组储存切片位置）
    slices_pos_all=slices_pos_all
    #因为nii配准之后是121*145*121的，为了不修改模型，每个方向的切片都填充成145*145的统一大小
    target_size_all=[target_size_x,target_size_y,target_size_z]
    #这个是save目录的前缀，代表了垂直与哪个轴切
    save_dir_prefixes=['X','Y','Z']
    slices_prefixs = ['AD_S', 'NC_S', 'MCIc_S', 'MCInc_S']

    for i in range(len(save_paths)):
        for j in range(len(save_dir_prefixes)):
            target_size = target_size_all[j]
            save_dir_prefix = save_dir_prefixes[j]
            slices_pos_axis = slices_pos_all[j]
            group_slices_num = int(len(slices_pos_axis)/9)
            for k in range(group_slices_num):
                group_id = k * 9
                for p in range(0,15):
                    pos = slices_pos_axis[group_id+p]
                    axis_id = save_dir_prefix+str(pos)
                    print('######## Begin create ' + 'group' + str(k) + ' slices ########')
                    save_path = os.path.join(save_paths[i], save_dir_prefix, 'group' + str(k))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print('Start scan fold:', save_paths[i], save_dir_prefixes[j])
                    print('group'+str(p))
                    # 读取文件列表
                    files = glob(os.path.join(data_paths[i], "*.nii"))  # 这里以s开头的nii文件代表的是是经过smooth处理后的文件
                    filenums = len(files)
                    print('this fold contents ', filenums, ' nii files')

                    for q in range(len(files)):
                        nii=nib.load(files[q])
                        filename_split=os.path.basename(files[q]).split('_')
                        subject_id=filename_split[1]+'_'+filename_split[2]+'_'+filename_split[3]

                        img=nii.get_data()

                        #灰度归一化
                        img=gray_norm(img)
                        #裁剪到指定大小，这里是为了使三个方向的切片大小一致
                        new_img=clip_to_target_size(img,target_size)
                        #设置文件名并保存slice
                        slice_name=os.path.join(save_path,slices_prefixs[i]+'_'+axis_id+'_'+subject_id+'.jpg')
                        save_slice(new_img,save_dir_prefix,pos,slice_name)


def generate_fold_and_slices():
    #原始nii图像主目录
    #509数据集，该部分数据要用来做train集和test集
    data_base_path_509 = '/home/DATA/HY/DATA_MRI'
    #切片存放主目录
    slices_base_dir='/home/hy/WorkProject/project/Data/MRI'

    slices_pos_x = [x for x in range(20, 98, 1)]
    slices_pos_y = [x for x in range(21, 126, 1)]
    slices_pos_z = [x for x in range(28, 97, 1)]

    slices_pos_all = [slices_pos_x, slices_pos_y, slices_pos_z]

    # 控制台
    flag_clean_base_dir = 0  # 是否清空目录，重新开始生成切片（0不清空，1清空）
    flag_create_and_save_slices = 0  # 是否生成并保存原始切片到AD，NC,MCIc,MCInc等基础目录中（0不重新生成原始切片，1重新生成原始切片）
    flag_create_classify_data = 1  # 是否将切片复制到相应的二分类器的数据目录中（0不执行该步骤，1执行该步骤）
    flag_create_and_save_slices_only_test = 0  # （对于509之外的数据集）是否生成并保存原始切片到AD，HC,MCIc,MCInc等基础目录中（0不重新生成原始切片，1重新生成原始切片）
    flag_create_classify_data_only_test = 0  # （对于509之外的数据集）是否将切片复制到相应的二分类器的数据目录中（0不执行该步骤，1执行该步骤）

    # 根据控制台中的flag,决定执行哪些功能
    if flag_clean_base_dir == 1:
        shutil.rmtree(slices_base_dir)

    if flag_create_and_save_slices == 1:
        create_and_save_slices(data_base_path_509, slices_base_dir, slices_pos_all)

    if flag_create_classify_data==1:
        create_classify_data(slices_base_dir, slices_pos_all)


if __name__=='__main__':
    generate_fold_and_slices()