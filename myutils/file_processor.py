import os
from os import listdir
from os.path import join
import SimpleITK as sitk
from skimage.io import imsave
import numpy as np
from PIL import Image


def read_tiff_stack(path):
    if os.path.isdir(path):
        images = [np.array(Image.open(os.path.join(path,p))) for p in sorted(os.listdir(path))]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)

def get_dir(path):
    tiffs = [join(path, f) for f in listdir(path) if f[0] != '.']
    return sorted(tiffs)

def read_nifti(path):
    if os.path.isdir(path):
        images = []
        for p in sorted(os.listdir(path)):
            if ".nii.gz" in p:
                itk_img = sitk.ReadImage(join(path, p))
                img = sitk.GetArrayFromImage(itk_img)
                images.append(img)
            else:
                continue
        return np.array(images)
    else:
        itk_img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(itk_img)
        return np.array(img).astype(np.uint8)

def load_nifti_to_tiff(path, out):
    itk_img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itk_img)
    imsave(out, img.astype(np.uint16), compress=6)

def load_tiff_to_nifti(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)
    itk_img = sitk.GetImageFromArray(np.array(images))
    return itk_img
