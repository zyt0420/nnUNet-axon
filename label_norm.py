import numpy as np
import os
from sklearn import preprocessing
import tifffile
from skimage.io import imread
import SimpleITK as sitk
from myutils import *
from einops import rearrange

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
# # base = "/media/root/data4/zyt/validate/gaba/"
# # v_path = os.path.join(base, "gaba_whole.tif")
# v_path = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/NIBS_Data/Sunmap-master-master/validate/gaba/gaba_label_new/label_0917.tiff"
# # l_path = os.path.join(base, "whole_label", "label.tiff")
# # v_out_path = "/media/root/data4/zyt/validate/gaba_nii/whole/gaba_whole_0000.nii.gz"
# v_out_path = "/media/root/data4/zyt/validate/gaba_nii/gaba_label_0917.nii.gz"
# # l_out_path = "/media/root/data4/zyt/validate/gaba_nii/gaba_label.nii.gz"
# itk_v = load_tiff_to_nifti(v_path)
# sitk.WriteImage(itk_v, v_out_path)
# # itk_l = load_tiff_to_nifti(l_path)
# # sitk.WriteImage(itk_l, l_out_path)

# ==================== nifti to tiff ==================== #
# base = "/media/root/data4/zyt/validate/162245_nii/test_whole/nnunet705/162245_whole.nii.gz"
# out = "/media/root/data4/zyt/validate/162245_nii/test_whole/nnunet705/162245_705pred.tiff"
# load_nifti_to_tiff(base, out)

# base = "/media/root/data4/zyt/validate/zsl4x_nii/test/nnunet705_a"
# out = "/media/root/data4/zyt/validate/zsl4x_nii/test/nnunet705_a/tiff"
# for i in sorted(os.listdir(base)):
#     print(i)
#     if ".gz" in str(i):
#         vbase = os.path.join(base, i)
#         casename = str(i).split(".")[0]
#         vout = os.path.join(out, casename + ".tiff")
#         load_nifti_to_tiff(vbase, vout)
#     else:
#         continue

# base = "/media/root/data4/zyt/validate/gaba_nii/test_whole/nnunet715_aa_f0/gaba_whole.nii.gz"
# outp = "/media/root/data4/zyt/validate/gaba_nii/test_whole/nnunet715_aa_f0/nnunet715_pred_gaba.tiff"
# img = load_nifti_to_tiff(base, outp)

# path = "/media/root/data4/zyt/validate/"
# arr = np.zeros((150, 150, 150), dtype=np.uint16)
# tifffile.imwrite(join(path, "junk_label.tiff"), arr)

# v_out_path = join(base, "141410_volumes")
# rect = 4
# rmtree(v_out_path)
# whole_volume = read_tiff_files(vpath)
# print(whole_volume.shape)
# # whole_volume = rearrange(whole_volume, '(z p_z)(h p_h)(w p_w) -> (z h w) p_z p_h p_w',
# #                          p_h=rect, p_w=rect, p_z=rect)  # n*(rect,rect,rect)
# whole_volume = rearrange(whole_volume, '(p_z z)(p_h h)(p_w w) -> (p_z p_h p_w) z h w',
#                          p_h=rect, p_w=rect, p_z=rect)  # 8*(z/rect,h/rect,w/rect)
# print(whole_volume.shape)
# for idx, v in enumerate(whole_volume):
#     # tifffile.imsave(join(v_out_path, str(idx).zfill(3) + '.tiff'), v)
#     image = sitk.GetImageFromArray(v)
#     sitk.WriteImage(image, join(v_out_path, str(idx).zfill(3) + '_0000.nii.gz'))
#     print(idx)

# ============ merge 48 cubes to one whole ============= #
# base = "/media/root/data4/zyt/validate/zsl8_max/"
# path = os.path.join(base, "zsl8_max_volumes")
# all_vols = []
# for i in sorted(os.listdir(path)):
#     all_vols.append(read_tiff_stack(os.path.join(path, i)))
# # for i in range(48):
# #     fname = str(i).zfill(3) + '.tif'
# #     fpath = os.path.join(path, fname)
# #     all_vols.append(read_tiff_stack(fpath).astype(np.uint8))
# all_vols = np.array(all_vols)
# whole_vol = rearrange(all_vols, '(z h w) p_z p_h p_w-> (z p_z)(h p_h)(w p_w)', z=3, h=4, w=4)
# # tifffile.imsave(os.path.join(base, "zsl8_max_whole.tiff"), whole_vol.astype(np.uint8))
# tifffile.imsave(os.path.join(base, "zsl8_max_whole.tiff"), whole_vol)

# base = "/media/root/data4/zyt/train/crop_ZSL4x/data/val/"
# v_path = os.path.join(base, "volumes")
# l_path = os.path.join(base, "labels")
# vpaths = os.listdir(v_path)
# lpaths = os.listdir(l_path)
# v_out_path = "/media/root/data4/zyt/validate/zsl4x_nii/volumes/"
# l_out_path = "/media/root/data4/zyt/validate/zsl4x_nii/labels/"
# task_name = "zsl4x_"
# # os.makedirs(out_path, exist_ok=True)
# for i, vpath in enumerate(vpaths):
#     print(vpath)
#     case = str(vpath).split(".")[0].split("-")[1]  # for layout: volume-xxx.tiff
#     # case = str(vpath).split(".")[0]  # for layout: xxx.tiff (in validate file)
#     label = join(l_path, "label-" + case + ".tif")
#     vol_file = os.path.join(v_path, vpath)
#     # lab_file = os.path.join(l_path, vpath)
#     v = imread(vol_file)
#     v_itk = sitk.GetImageFromArray(v.astype(np.float32))
#     l = imread(label)
#     l[l > 0] = 1
#     l_itk = sitk.GetImageFromArray(l.astype(np.float32))
#     casename = task_name + case
#     v_out_base = os.path.join(v_out_path, casename)
#     sitk.WriteImage(v_itk, os.path.join(v_out_base + "_0000.nii.gz"))
#     anno_out = os.path.join(l_out_path, casename + ".nii.gz")
#     sitk.WriteImage(l_itk, anno_out)
