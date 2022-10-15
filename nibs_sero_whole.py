import numpy as np
import os
from sklearn import preprocessing
import tifffile
from skimage.io import imread
import SimpleITK as sitk
from myutils import *
from einops import rearrange
import nibabel as nib
from skimage import transform
import matplotlib.pyplot as plt


'''
zsl9_4x z y x 1076 3599 (3616)[120,3320] -> 1076 600 904
x 0 904 1808 2712 y [0,5]i*600
zsl9_4x_half: after resize 320 456(234) 528 before 1076 3599(1847) 3616
x 0 904 1808 2712 y [0,2]i*616(last patch 615)
zsl9_4x z y x 1043 3601 3622 -> 1043 600 1207
x 0 1207 2414 y [0,5]*600
gaba_2 z y x 1033 3743 3759(3360)[398,.] -> 1033 623 840
x 398 1238 2078 2918 y [0,5]*623
'''
# ============ crop large volume into small cubes in order ============= #
# base1 = "/media/root/ssd1/NIBS_1/"
# base2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/gaba_2/"
# vpath = join(base1, "ZSL_20220508_Vgat_Left-side_#2_647_4x")
# x_patch = 840
# y_patch = 623
# z_gap = 1033
# for i in range(6):
#     x, y, zslice = [398, i*y_patch, 0]
#     print(x, y, zslice)
#     print("\ngaba2_" + str(i*4).zfill(2) + "\n")
#     whole_path = join(base2, "gaba2_" + str(i*4).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)
#
#     x, y, zslice = [1238, i*y_patch, 0]
#     print(x, y, zslice)
#     print("\ngaba2_" + str(i*4+1).zfill(2) + "\n")
#     whole_path = join(base2, "gaba2_" + str(i*4+1).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)
#
#     x, y, zslice = [2078, i*y_patch, 0]
#     print(x, y, zslice)
#     print("\ngaba2_" + str(i*4+2).zfill(2) + "\n")
#     whole_path = join(base2, "gaba2_" + str(i*4+2).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)
#
#     x, y, zslice = [2918, i * y_patch, 0]
#     print(x, y, zslice)
#     print("\ngaba2_" + str(i * 4 + 3).zfill(2) + "\n")
#     whole_path = join(base2, "gaba2_" + str(i * 4 + 3).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y + y_patch, x: x + x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)


# ========== tiff cubes to nii ========== #
base = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/gaba_2/"
out = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/gaba2_7164/"
paths = os.listdir(base)
for path in paths:
    if ".tif" in path:
        print(path)
        img = load_tiff_to_nifti(join(base, path))
        outp = join(out, str(path).split(".")[0] + "_0000.nii.gz")
        sitk.WriteImage(img, outp)

'''
CUDA_VISIBLE_DEVICES=2 nnUNet_predict -i "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/gaba2_7164/whole/" -o "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/gaba2_7163/pred13/" -tr MyTrainerAxial -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1_16GB -t 7163 -f 0
'''


# 这个运行6次 012 345 678 91011 1213114 151617
# ============ merge 4 brain seg results into 1/4 whole ============= #
# path1 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred00/zsl9_4x_00.nii.gz"
# path2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred01/zsl9_4x_01.nii.gz"
# path3 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred01/zsl9_4x_02.nii.gz"
# itk_img = sitk.ReadImage(path1)
# img = sitk.GetArrayFromImage(itk_img)
# arr1 = np.array(img).astype(np.uint8)
# print(arr1.shape)
# itk_img = sitk.ReadImage(path2)
# img = sitk.GetArrayFromImage(itk_img)
# arr2 = np.array(img).astype(np.uint8)
# print(arr2.shape)
# itk_img = sitk.ReadImage(path3)
# img = sitk.GetArrayFromImage(itk_img)
# arr3 = np.array(img).astype(np.uint8)
# print(arr3.shape)
# arr = np.concatenate((arr1, arr2, arr3), axis=2)
# # arr12 = np.concatenate((arr1, arr2), axis=2)  # x
# # arr34 = np.concatenate((arr3, arr4), axis=2)  # x
# # arr = np.concatenate((arr12, arr34), axis=1)  # y
# print(arr.shape)
# itk_img = sitk.GetImageFromArray(arr)
# path = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl9_4x_012.nii.gz"  # 3612 600 1043
# sitk.WriteImage(itk_img, path)


# ============ resize segments of 1/4 whole ============= #
# path1 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl_4x_whole_0246.nii.gz"
# path2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl_4x_whole_8101214.nii.gz"
# path3 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl_4x_whole_16182022.nii.gz"
# path1 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl9_4x_012.nii.gz"
# path2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl9_4x_345.nii.gz"
# path3 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/pred_zsl9_4x_678.nii.gz"
# target = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/re_pred_zsl_4x_whole_up.nii.gz"
# img = nib.load(path1).get_fdata()
# print(img.shape)
# out1 = transform.resize(img, (903, 150, 521))  # x y z
# print(out1.shape)
# img = nib.load(path2).get_fdata()
# print(img.shape)
# out2 = transform.resize(img, (903, 150, 521))  # x y z
# print(out2.shape)
# img = nib.load(path3).get_fdata()
# print(img.shape)
# out3 = transform.resize(img, (903, 150, 521))  # x y z
# print(out3.shape)
# arr = np.concatenate((out1, out2, out3), axis=1)  # 903 450 521
# print(arr.shape)
# # itk_img = sitk.GetImageFromArray(arr)
# # sitk.WriteImage(itk_img, target)
# img = nib.Nifti1Image(arr, np.eye(4))
# img.to_filename(target)

# ============ merge resized seg results to whole ============= #
# path1 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/re_pred_zsl_4x_whole_left.nii.gz"
# path2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/re_pred_zsl_4x_whole_right.nii.gz"
# img1 = nib.load(path1).get_fdata()
# print(img1.shape)
# img2 = nib.load(path2).get_fdata()
# print(img2.shape)
# arr = np.concatenate((img1, img2), axis=0)  # x
# print(arr.shape)
# path = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/zsl9_7143/re_pred_zsl_4x_whole.nii.gz"
# # # itk_img = sitk.GetImageFromArray(arr)
# # # sitk.WriteImage(itk_img, path)
# img = nib.Nifti1Image(arr, np.eye(4))
# img.to_filename(path)
