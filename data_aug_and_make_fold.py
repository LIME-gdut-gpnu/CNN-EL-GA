#-*- coding: utf-8 -*-
import os
import sys
import random
import shutil
from glob import glob

import scipy
import numpy as np
from image import ImageDataGenerator, array_to_img, img_to_array, load_img
from creat_slices import generate_fold_and_slices

# 辅助函数：重构文件名，替换文件名中的坐标信息，如将：AD_S_X61_002_S_0816变为AD_S_Y41_002_S_0816
def refactor_filename(files_names, axis_id):
    file_list = []
    for name in files_names:
        name_split = name.split('_')
        name_split[2] = axis_id
        new_name = '_'.join(name_split)
        file_list.append(new_name)
    return file_list

def Rotation(path, image_save_dir, aug_number):
    datagen = ImageDataGenerator(rotation_range=15)  # 15
    img = load_img(path, grayscale=True)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组
    image_dir = os.path.dirname(path)
    image_name = os.path.basename(path)
    prefix = image_name.split(".")[0]

    i = 1
    for batch in datagen.flow(x, batch_size=1, save_to_dir=image_save_dir,
                              save_prefix=prefix + "_aug_" + 'rot', save_format='jpg'):
        i += 1
        if i > aug_number:
            break  # 否则生成器会退出循环
            # print("Rotation end")


def Gamma_correction(path, image_save_dir, aug_number):
    img = load_img(path, grayscale=True)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组
    x = x.reshape(x.shape[0], -1)  # 转成二维数组

    for i in range(aug_number):
        gamma = np.random.uniform(low=0.7, high=1.3)
        adjusted_img = (x / 255) ** gamma * 255
        image_name = os.path.basename(path)
        prefix = image_name.split(".")[0]
        save_path = os.path.join(image_save_dir, prefix + '_aug_gamma_' + str(i) + '.jpg')
        scipy.misc.imsave(save_path, adjusted_img)
        # print("Gamma_correction end")


def Noise_Injection(path, image_save_dir, aug_number):
    img = load_img(path, grayscale=True)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组
    x = x.reshape(x.shape[0], -1)  # 转成二维数组
    x_shape = x.shape
    for i in range(aug_number):
        noise = np.random.normal(loc=0.0, scale=0.01, size=x_shape)
        # noise=np.random.random(size=x_shape)
        x_add_noise = np.add(x, noise)
        # 保存图片
        image_dir = os.path.dirname(path)
        image_name = os.path.basename(path)
        prefix = image_name.split(".")[0]
        save_path = os.path.join(image_save_dir, prefix + '_aug_noise_' + str(i) + '.jpg')
        scipy.misc.imsave(save_path, x_add_noise)
        # print("Noise_Injection end")
        # self.x_corrupted = tf.add(x, self.scale*tf.random_normal(shape = (self.n_input,),mean=0.0,stddev=1.0,))


def Random_translation(path, image_save_dir, aug_number):
    datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05)  # 0.2
    img = load_img(path, grayscale=True)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组
    image_dir = os.path.dirname(path)
    image_name = os.path.basename(path)
    prefix = image_name.split(".")[0]
    i = 1
    for batch in datagen.flow(x, batch_size=1, save_to_dir=image_save_dir,
                              save_prefix=prefix + "_aug_" + 'trans', save_format='jpg'):
        i += 1
        if i > aug_number:
            break  # 否则生成器会退出循环
            # print("Random_translation end")


def Scaling(path, image_save_dir, aug_number):
    datagen = ImageDataGenerator(zoom_range=0.3, fill_mode='constant')  # 0.3
    img = load_img(path, grayscale=True)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组
    image_dir = os.path.dirname(path)
    image_name = os.path.basename(path)
    prefix = image_name.split(".")[0]
    i = 1
    for batch in datagen.flow(x, batch_size=1, save_to_dir=image_save_dir,
                              save_prefix=prefix + "_aug_" + 'Scal', save_format='jpg'):
        i += 1
        if i > aug_number:
            break  # 否则生成器会退出循环
            # print("Scaling end")


def Random_affine_transform(path, image_save_dir, aug_number):
    datagen = ImageDataGenerator(shear_range=3)  # 水平或垂直投影变换,shear_range是角度范围 #5
    img = load_img(path, grayscale=True)  # 这是一个PIL图像
    x = img_to_array(img)  # 把PIL图像转换成一个numpy数组
    x = x.reshape((1,) + x.shape)  # 这是一个numpy数组
    image_dir = os.path.dirname(path)
    image_name = os.path.basename(path)
    prefix = image_name.split(".")[0]
    i = 1
    for batch in datagen.flow(x, batch_size=1, save_to_dir=image_save_dir,
                              save_prefix=prefix + "_aug_" + 'aff', save_format='jpg'):
        i += 1
        if i > aug_number:
            break  # 否则生成器会退出循环
            # print("Random_affine_transform end")


# 对于每类图片，每张图片都生成相同张数的augment图片
def aug_image_for_same_augnum(images_all, image_save_dir, aug_number):
    # 如果存在保存目录，则清空里面的文件然后创建该目录
    print("aug start!")
    if os.path.exists(image_save_dir):
        shutil.rmtree(image_save_dir)
    os.makedirs(image_save_dir)
    # 先为所有的图片生成augment图片
    for image_path in images_all:
        Gamma_correction(image_path, image_save_dir, aug_number)
        Rotation(image_path, image_save_dir, aug_number)
        Random_translation(image_path, image_save_dir, aug_number)
        Scaling(image_path, image_save_dir, aug_number)
        Random_affine_transform(image_path, image_save_dir, aug_number)
        Noise_Injection(image_path, image_save_dir, aug_number)
    print("aug end!")

# 对于每类图片，根据图像数量的不平衡性，每类每张图片都生成不同张数的augment图片
def aug_image_for_diff_augnum(images_all, image_save_dir, aug_class_less, aug_class_more, aug_number1, aug_number2):
    # 如果存在保存目录，则清空里面的文件然后创建该目录
    print("aug start!")
    if os.path.exists(image_save_dir):
        shutil.rmtree(image_save_dir)
    os.mkdir(image_save_dir)
    # 先为所有的图片生成augment图片
    for image_path in images_all:
        if aug_class_less in os.path.split(image_path)[1].split('.')[0]:
            Gamma_correction(image_path, image_save_dir, aug_number1)
            Rotation(image_path, image_save_dir, aug_number1)
            Random_translation(image_path, image_save_dir, aug_number1)
            Scaling(image_path, image_save_dir, aug_number1)
            Random_affine_transform(image_path, image_save_dir, aug_number1)
            Noise_Injection(image_path, image_save_dir, aug_number1)
        elif aug_class_more in os.path.split(image_path)[1].split('.')[0]:
            Gamma_correction(image_path, image_save_dir, aug_number2)
            Rotation(image_path, image_save_dir, aug_number2)
            Random_translation(image_path, image_save_dir, aug_number2)
            Scaling(image_path, image_save_dir, aug_number2)
            Random_affine_transform(image_path, image_save_dir, aug_number2)
            Noise_Injection(image_path, image_save_dir, aug_number2)
    print("aug end!")

# 将数据集划分成k折，由于要保证每个坐标的每折中，对应的原始病人都是一样的，因此第一个切片在划分fold时，要将fold的划分情况存储起来，后面其他切片在划分时，就读取这个保存好的划分。
# 所以第一个切片的处理跟后面其他切片有区别，单独用一个函数进行处理
# 该函数在划分时能保证每个fold中，每个类别（如AD/HC）的切片数比例是相同的
def make_fold_list1(image_base_dir, images_all, images_aug, fold):
    # 该函数只在第一次划分时调用
    print("Start make first slice fold datalist...")

    # 这些变量用于保存k fold的划分情况
    train_data_filename_mat = []
    test_data_filename_mat = []

    # 这些变量用于自动从传入的图片列表中找出两类图片各有多少，并将图片按类别划分开
    class1 = ''
    class2 = ''
    class1_imgs = []
    class2_imgs = []
    class1_num = 0
    class2_num = 0
    class1_sample_num = 0
    class2_sample_num = 0

    # 只提取文件名
    all_data_filename = [os.path.split(image)[1].split('.')[0] for image in images_all]
    if all_data_filename[0].split('_')[0] == 'AD':
        class1_sample_num = 93
    elif all_data_filename[0].split('_')[0] == 'MCIc':
        class1_sample_num = 76

    if all_data_filename[-1].split('_')[0] == 'NC':
        class2_sample_num = 100
    elif all_data_filename[-1].split('_')[0] == 'MCInc':
        class2_sample_num = 134

    some_data = all_data_filename[0:class1_sample_num] + all_data_filename[len(all_data_filename)-class2_sample_num:len(all_data_filename)]
    for id,value in enumerate(some_data):
        temp = value.split('_')
        some_data[id] = temp[0]+'_'+temp[3]+'_'+temp[4]+'_'+temp[5]

    some_data = sorted(some_data)

    # 这两个循环用于将AD_HC中的图片分开成AD、HC两个类别，并将两类图片分开存储，用于后面实现多折抽样中每折类别比例平衡
    for i in range(len(some_data)):
        if i == 0:
            class1 = some_data[i].split('_')[0]
            class1_num += 1
            class1_imgs.append(some_data[i])
        else:
            if some_data[i].split('_')[0] == class1:
                class1_num += 1
                class1_imgs.append(some_data[i])
            else:
                class2 = some_data[i].split('_')[0]
                class2_num += 1
                class2_imgs.append(some_data[i])

    # 将两类图片区分开后，进行顺序打乱，计算每类数据分成k折每折应该有多少张
    random.shuffle(class1_imgs)
    random.shuffle(class2_imgs)
    fold_size1 = class1_num // fold
    fold_size2 = class2_num // fold
    ###############################################
    for i in range(fold):

        train_data = []
        test_data = []

        test_data1 = class1_imgs[i * fold_size1:(i + 1) * fold_size1]
        train_data1 = class1_imgs[0:i * fold_size1] + class1_imgs[(i + 1) * fold_size1:len(class1_imgs)]

        test_data2 = class2_imgs[i * fold_size2:(i + 1) * fold_size2]
        train_data2 = class2_imgs[0:i * fold_size2] + class2_imgs[(i + 1) * fold_size2:len(class2_imgs)]

        train_temp = train_data1 + train_data2
        test_temp = test_data1 + test_data2

        for id, value in enumerate(all_data_filename):
            temp = value.split('_')
            if temp[0]+'_'+temp[3]+'_'+temp[4]+'_'+temp[5] in train_temp:
                train_data.append(value)
            else:
                test_data.append(value)

        # 由于之train_data和test_data中都只保留了文件名，而txt中应该保存绝对路径，所以这里把文件名拼接成图片绝对路径
        images_path = os.path.join(image_base_dir, 'slices_all')
        train_data_full_names = [os.path.join(images_path, name + '.jpg') for name in train_data]
        test_data_full_names = [os.path.join(images_path, name + '.jpg') for name in test_data]

        # 初始的时候把原始图片也加到训练数据中,train_folds中包含了原始训练图片和aug中的
        train_folds = train_data_full_names

        ##排序是为了方便比较，不影响结果，调试完后可以注释掉
        # train_data = sorted(train_data)
        # test_data = sorted(test_data)
        # 只提取文件名
        # train_data_filename = [os.path.split(image)[1].split('.')[0] for image in train_data]
        # test_data_filename = [os.path.split(image)[1].split('.')[0] for image in test_data]
        train_data_filename = train_temp
        test_data_filename = test_temp

        # 训练集还要加上aug得到的图片，测试集不用加aug的图片
        train_data_filename_set = set(train_temp)  # 这里转换成set是因为在set上做in运算比在list上做in运算要快
        for j in range(len(images_aug)):
            value = os.path.split(images_aug[j])[1].split('_aug_')[0]
            temp = value.split('_')
            if temp[0]+'_'+temp[3]+'_'+temp[4]+'_'+temp[5] in train_data_filename_set:
                train_folds = train_folds + [images_aug[j]]

        # 排序是为了方便比较，不影响结果，调试完后可以注释掉
        # train_folds = sorted(train_folds)

        # 设置文件名并保存文件，文件里都是记录的图片绝对路径
        train_txt_name = os.path.join(image_base_dir, 'fold' + str(i) + '_train.txt')
        test_txt_name = os.path.join(image_base_dir, 'fold' + str(i) + '_test.txt')
        train_aug_txt_name = os.path.join(image_base_dir, 'fold' + str(i) + '_train_aug.txt')
        np.savetxt(train_txt_name, train_data_full_names, fmt='%s')
        np.savetxt(test_txt_name, test_data_full_names, fmt='%s')
        np.savetxt(train_aug_txt_name, train_folds, fmt='%s')

        # 在第一个切片上划分多折时，保存每个fold的测试集和训练集的图像名称并返回。这样才能保证其他切片在划分时和这个一样
        if i == 0:
            train_data_filename_mat = train_data_filename
            test_data_filename_mat = test_data_filename
        else:
            train_data_filename_mat = np.c_[train_data_filename_mat, train_data_filename]
            test_data_filename_mat = np.c_[test_data_filename_mat, test_data_filename]

            # 如何读取保存的文件列表
            # aa = np.loadtxt(train_txt_name, dtype='str')
            # bb = np.loadtxt(test_txt_name, dtype='str')
    print("first slice all fold datalist generated")
    return train_data_filename_mat, test_data_filename_mat

# 根据传入的划分规则，将数据集划分成k折
def make_fold_list2(image_base_dir, images_all, images_aug, train_filename_mat, test_filename_mat, fold):  #
    print("Start make fold datalist...")

    # 根据第一个切片上的划分情况来划分其他切片
    for i in range(fold):
        print("generate " + image_base_dir + " " + "fold" + str(i) + " list...")
        # 这里要根据传进来的两个矩阵确定训练样本和测试样本
        test_data = []
        train_data = []

        # 拼接文件名，形成文件名列表
        train_data_filename = train_filename_mat[:, i]
        # train_data_filename = refactor_filename(train_data_filename, axis_id)
        test_data_filename = test_filename_mat[:, i]
        # test_data_filename = refactor_filename(test_data_filename, axis_id)

        for image in images_all:
            image_name = os.path.split(image)[1].split('.')[0]
            temp = image_name.split('_')
            if temp[0]+'_'+temp[3]+'_'+temp[4]+'_'+temp[5] in test_data_filename:
                test_data.append(image)
            else:
                train_data.append(image)

        train_folds = train_data  # 初始的时候把原始图片也加到训练数据中
        # 训练集还要加上aug得到的图片，测试集不用加aug的图片

        train_data_filename_set = set(train_data_filename)  # 这里转换成set是因为在set上做in运算比在list上做in运算要快
        for j in range(len(images_aug)):
            value = os.path.split(images_aug[j])[1].split('_aug_')[0]
            temp = value.split('_')
            if temp[0]+'_'+temp[3]+'_'+temp[4]+'_'+temp[5] in train_data_filename_set:
                train_folds = train_folds + [images_aug[j]]

        train_txt_name = os.path.join(image_base_dir, 'fold' + str(i) + '_train.txt')
        test_txt_name = os.path.join(image_base_dir, 'fold' + str(i) + '_test.txt')
        train_aug_txt_name = os.path.join(image_base_dir, 'fold' + str(i) + '_train_aug.txt')

        np.savetxt(train_txt_name, train_data, fmt='%s')
        np.savetxt(test_txt_name, test_data, fmt='%s')
        np.savetxt(train_aug_txt_name, train_folds, fmt='%s')


        # 如何读取保存的文件列表
        # aa = np.loadtxt(train_txt_name, dtype='str')
        # bb = np.loadtxt(test_txt_name, dtype='str')
    print("all fold datalist generated")

def main():
    print("generate_fold_and_slices begin...")
    # 是否调用函数生成切片，如果切片已生成，就可以不运行此函数
    # generate_fold_and_slices()
    print("generate_fold_and_slices end")
    print("Augmentation begin...")

    # 切片存放主目录
    # workdir = os.path.dirname(sys.argv[0])
    workdir = "/home/hy/WorkProject/project"
    slices_base_dir = os.path.join(workdir, 'Data', 'MRI')

    # 生成3个类别文件夹
    path1 = os.path.join(slices_base_dir, 'AD_NC')
    path2 = os.path.join(slices_base_dir, 'MCIc_NC')
    path3 = os.path.join(slices_base_dir, 'MCIc_MCInc')
    data_paths = [path1, path2, path3]


    # 文件夹前缀
    dir_prefixes = ['X', 'Y', 'Z']

    # 四种图像数据的数量，用来算不平衡augment时，每类图像augment的比例
    AD_num = 93
    NC_num = 100
    MCIc_num = 76
    MCInc_num = 134

    # 进行5折交叉验证
    fold = 10
    # 每张图片生成多少张扰动后的图片，当两类图像分别生成不同数量的aug图片时，该数值为较多的那类图片的每张扩充量
    base_aug_number = 5

    aug_image = 0  # 1 每类扩充同样的样本;2 两类根据其不平衡程度扩充不同的量;0 不执行样本扩充
    make_fold = 1  # 1 生成多折交叉验证数据文件 ; 0 不执行该操作

    # 循环不同类别
    for i in range(len(data_paths)):
        classify_base_dir = data_paths[i]
        # 设置计数器
        count = 0
        # 这两个矩阵用来保存第一种切片上划分的训练集和测试集图像名称
        train_data_filename_mat = []
        test_data_filename_mat = []
        # 循环不同坐标轴
        for j in range(len(dir_prefixes)):
            dir_prefix = dir_prefixes[j]
            current_path = os.path.join(classify_base_dir, dir_prefix)
            current_path_files_num = len([name for name in os.listdir(current_path)])
            for k in range(current_path_files_num):
                image_base_dir = os.path.join(current_path, 'group'+str(k))
                image_base_dir_slices_all = os.path.join(image_base_dir, 'slices_all')
                aug_image_save_dir = os.path.join(image_base_dir, 'aug_all')

                print('Slices in ' + image_base_dir + ' Process Start...')

                # 自动检测文件夹，进行哪类实验数据的生成
                if 'AD_NC' in image_base_dir:
                    aug_flag = 1  # 1:AD_NC ;2:MCIc_NC;3:MCIc_MCInc
                elif 'MCIc_NC' in image_base_dir:
                    aug_flag = 2
                elif 'MCIc_MCInc' in image_base_dir:
                    aug_flag = 3
                else:
                    exit()

                # 获取所有image的列表
                images_all = glob(os.path.join(image_base_dir_slices_all, "*.jpg"))

                # 计算每种类别的样本数，然后按不同比例生成扩充数据集，使扩充后的数据集正负例样本趋于平衡
                if aug_flag == 1:
                    aug_class_less = 'AD'
                    aug_class_more = 'NC'
                    aug_number1 = round(base_aug_number * (NC_num / AD_num))
                    aug_number2 = base_aug_number
                elif aug_flag == 2:
                    aug_class_less = 'MCIc'
                    aug_class_more = 'NC'
                    aug_number1 = round(base_aug_number * (NC_num / MCIc_num))
                    aug_number2 = base_aug_number
                elif aug_flag == 3:
                    aug_class_less = 'MCIc'
                    aug_class_more = 'MCInc'
                    aug_number1 = round(base_aug_number * (MCInc_num / MCIc_num))
                    aug_number2 = base_aug_number

                # 调用函数，采用6种方法生成扩充数据
                if aug_image == 1:
                    aug_image_for_same_augnum(images_all, aug_image_save_dir, base_aug_number)  # 每类扩充同样的样本
                    print("Augmentation end...")
                elif aug_image == 2:
                    aug_image_for_diff_augnum(images_all, aug_image_save_dir, aug_class_less, aug_class_more,
                                              aug_number1, aug_number2)  # 两类根据其不平衡程度扩充不同的量
                    print("Augmentation end...")

                images_aug = glob(os.path.join(aug_image_save_dir, "*.jpg"))

                # 分成n个数据集,n为n折交叉验证
                if make_fold == 1:
                    # 在第一个切片处要保存分割后各个fold的划分情况,所以第一次z循环与后面的循环调用不同的函数处理
                    if count == 0:
                        train_data_filename_mat, test_data_filename_mat = make_fold_list1(image_base_dir, images_all,
                                                                                          images_aug, fold)
                    else:
                        # 由于每个文件夹下的图片名中带了Z40，X40之类的标记，所以要将文件名的这部分改变一下
                        make_fold_list2(image_base_dir, images_all, images_aug, train_data_filename_mat,
                                        test_data_filename_mat, fold)
                count += 1




if __name__ == '__main__':
    main()