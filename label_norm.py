import numpy as np
import os
from sklearn import preprocessing
import tifffile
from skimage.io import imread
import SimpleITK as sitk
from myutils import *

# file = "/media/root/data4/zyt/train/155829/train/labels/label-2010.tif"
# img = imread(file)
# print("原始：\n")
# print(img)
# img[img > 0] = 1
# print("大于0变为1：\n")
# print(img)
# img = img.astype(np.float32)
# print("转类型：\n")
# print(img)

# ==================== tiff to nifti ==================== #
# base = "/media/root/data4/zyt/validate/141410/"
# # v_path = os.path.join(base, "162245_whole.tiff")
# l_path = os.path.join(base, "label_rect.tiff")
# # v_out_path = "/media/root/data4/zyt/validate/162245_nii/whole/162245_whole_0000.nii.gz"
# l_out_path = "/media/root/data4/zyt/validate/141410_nii/141410_label_rect.nii.gz"
# # itk_v = load_tiff_to_nifti(v_path)
# # sitk.WriteImage(itk_v, v_out_path)
# itk_l = load_tiff_to_nifti(l_path)
# sitk.WriteImage(itk_l, l_out_path)

# ==================== nifti to tiff ==================== #
base = '/media/root/data4/zyt/validate/141410_nii/test/whole/nnunet7090/141410_whole.nii.gz'
out = '/media/root/data4/zyt/validate/141410_nii/test/whole/nnunet7090/141410_nnunet_pred.tiff'
load_nifti_to_tiff(base, out)

'''
base = "/media/root/data4/zyt/validate/141410/"
v_path = os.path.join(base, "141410_volumes")
l_path = os.path.join(base, "junk_labels_rectified")
vpaths = os.listdir(v_path)
lpaths = os.listdir(l_path)
v_out_path = "/media/root/data4/zyt/validate/141410_nii/volumes/"
l_out_path = "/media/root/data4/zyt/validate/141410_nii/labels/"
task_name= "GT141410_"
# os.makedirs(out_path, exist_ok=True)
for i, vpath in enumerate(vpaths):
    print(vpath)
    # case = str(vpath).split(".")[0].split("-")[1]  # for layout: volume-xxx.tiff
    case = str(vpath).split(".")[0]  # for layout: xxx.tiff (in validate file)
    # label = join(train_label_path, "label-" + case + ".tif")
    vol_file = os.path.join(v_path, vpath)
    # lab_file = os.path.join(l_path, vpath)
    v = imread(vol_file)
    v_itk = sitk.GetImageFromArray(v.astype(np.float32))
    # l = imread(lab_file)
    # l[l > 0] = 1
    # l_itk = sitk.GetImageFromArray(l.astype(np.float32))
    casename = task_name + case
    v_out_base = os.path.join(v_out_path, casename)
    sitk.WriteImage(v_itk, os.path.join(v_out_base + "_0000.nii.gz"))
    # anno_out = os.path.join(l_out_path, casename + ".nii.gz")
    # sitk.WriteImage(l_itk, anno_out)
'''
