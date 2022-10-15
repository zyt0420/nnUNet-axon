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
141410 z y x (2078)2080 2560 2160 -> 1039 640 1080
141827 z y x 2007 2560 2160 -> 1004(1003) 640 1080
154333 z y x 2005 2560 2160 -> 1003(1002) 640 1080
DN 155829 z y x 1892 2560 2160 -> 946 640 1080
 00 [0, 0, 0]      01 [0, 0, 1039]      02 [1080, 0, 0]      03 [1080, 0, 1039]
 04 [0, 640, 0]    05 [0, 640, 1039]    06 [1080, 640, 0]    07 [1080, 640, 1039]
 08 [0, 1280, 0]   09 [0, 1280, 1039]   10 [1080, 1280, 0]   11 [1080, 1280, 1039]
 12 [0, 1920, 0]   13 [0, 1920, 1039]   14 [1080, 1920, 0]   15 [1080, 1920, 1039]
'''
# ============ crop large volume into small cubes in order ============= #
# base1 = "/media/root/lzy_data/647_Stanford/DN/"
# base2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN_155829/"
# vpath = join(base1, "190702_647_0-8x_28xdynamicfocussing_newcap_15-58-29")
# x_patch = 1080
# y_patch = 640
# z_gap = 946
# for i in range(4):
#     x, y, zslice = [0, i*y_patch, 0]
#     print(x, y, zslice)
#     print("\nDN_155829_" + str(i*4).zfill(2) + "\n")
#     whole_path = join(base2, "DN_155829_" + str(i*4).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)
#
#     x, y, zslice = [0, i*y_patch, 946]
#     print(x, y, zslice)
#     print("\nDN_155829_" + str(i*4+1).zfill(2) + "\n")
#     whole_path = join(base2, "DN_155829_" + str(i*4+1).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)
#
#     x, y, zslice = [1080, i*y_patch, 0]
#     print(x, y, zslice)
#     print("\nDN_155829_" + str(i*4+2).zfill(2) + "\n")
#     whole_path = join(base2, "DN_155829_" + str(i*4+2).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)
#
#     x, y, zslice = [1080, i*y_patch, 946]
#     print(x, y, zslice)
#     print("\nDN_155829_" + str(i*4+3).zfill(2) + "\n")
#     whole_path = join(base2, "DN_155829_" + str(i*4+3).zfill(2))
#     rmtree(whole_path)
#     for idx, f in enumerate(sorted(os.listdir(vpath))[zslice: zslice + z_gap]):
#         img = np.array(Image.open(join(vpath, f))).astype(np.uint16)
#         cropped = img[y: y+y_patch, x: x+x_patch]
#         tifffile.imsave(join(whole_path, str(idx).zfill(4)) + '.tiff', cropped)


# ========== tiff cubes to nii ========== #
# base = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN_155829/"
# out = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/"
# paths = os.listdir(base)
# for path in paths:
#     if ".tif" in path:
#         print(path)
#         img = load_tiff_to_nifti(join(base, path))
#         outp = join(out, str(path).split(".")[0] + "_0000.nii.gz")
#         sitk.WriteImage(img, outp)


# 这个要运行4次 分别是0123 4567 891011 12131415
# xy轴大小一样 根据z轴张数计算resize之后z轴的大小（缩小4倍）
# ============ resize 4 brain seg results and merge into 1/4 whole ============= #
# path1 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/pred12/DN_155829_12.nii.gz"
# path2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/pred13/DN_155829_13.nii.gz"
# path3 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/pred14/DN_155829_14.nii.gz"
# path4 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/pred15/DN_155829_15.nii.gz"
# path = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/re_pred_DN_155829_12131415.nii.gz"
# arr = nib.load(path1).get_fdata()
# print(arr.shape)  # z y x 946 640 1080
# out1 = transform.resize(arr, (270, 160, 237))  # x y z
# print(out1.shape)
# arr = nib.load(path2).get_fdata()
# print(arr.shape)
# out2 = transform.resize(arr, (270, 160, 237))  # x y z
# print(out2.shape)
# arr = nib.load(path3).get_fdata()
# print(arr.shape)
# out3 = transform.resize(arr, (270, 160, 237))  # x y z
# print(out3.shape)
# arr = nib.load(path4).get_fdata()
# print(arr.shape)
# out4 = transform.resize(arr, (270, 160, 237))  # x y z
# print(out4.shape)
# arr12 = np.concatenate((out1, out2), axis=2)  # z
# print(arr12.shape)
# arr34 = np.concatenate((out3, out4), axis=2)  # z
# print(arr34.shape)
# arr = np.concatenate((arr12, arr34), axis=0)  # x
# print(arr.shape)
# # arr = np.transpose(arr, (2, 1, 0))
# # itk_img = sitk.GetImageFromArray(arr)
# # sitk.WriteImage(itk_img, path)
# img = nib.Nifti1Image(arr, np.eye(4))
# img.to_filename(path)

# 这个不需要
# ============ resize segments of 1/4 whole ============= #
# allen x y z 456 528 320
# stanford 540 640 z/4
# base = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/154333_nnunet"
# paths = os.listdir(base)
# for path in paths:
#     if "pred_" in path:
#         print(path)
#         target = "re_" + path
#         itk_img = sitk.ReadImage(join(base, path))
#         img = np.array(itk_img).astype(np.uint8)
#         print(img.shape)
#         out = transform.resize(img, (501, 640, 540))  # z y x
#         print(out.shape)
#         itk_img = sitk.GetImageFromArray(out)
#         sitk.WriteImage(itk_img, target)
# 到这为止不需要

# ============ merge resized seg results to whole ============= #
# path1 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/re_pred_DN_155829_0123.nii.gz"
# path2 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/re_pred_DN_155829_4567.nii.gz"
# path3 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/re_pred_DN_155829_891011.nii.gz"
# path4 = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/re_pred_DN_155829_12131415.nii.gz"
# target = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zyt/DN155829_nnunet/re_pred_DN_155829_whole.nii.gz"
# arr1 = nib.load(path1).get_fdata()
# print(arr1.shape)
# arr2 = nib.load(path2).get_fdata()
# print(arr2.shape)
# arr3 = nib.load(path3).get_fdata()
# print(arr3.shape)
# arr4 = nib.load(path4).get_fdata()
# print(arr4.shape)
# arr = np.concatenate((arr1, arr2, arr3, arr4), axis=1)  # y
# print(arr.shape)
# # itk_img = sitk.GetImageFromArray(arr)
# # sitk.WriteImage(itk_img, target)
# img = nib.Nifti1Image(arr, np.eye(4))
# img.to_filename(target)
