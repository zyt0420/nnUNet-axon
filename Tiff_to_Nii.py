#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
from nnunet.paths import preprocessing_output_dir
from skimage.io import imread
import pdb

# join = os.path.join


def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out, spacing):
    img = imread(img_file)
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    # img_itk.SetSpacing(np.array(spacing)[::-1])
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        l[l > 0] = 1
        l_itk = sitk.GetImageFromArray(l.astype(np.float32))
        # l_itk.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(l_itk, anno_out)


def prepare_task(base, task_id, task_name, spacing):
    p = Pool(16)

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    res = []

    train_volume_path = join(base, "train", "volumes")
    train_label_path = join(base, "train", "labels")
    val_volume_path = join(base, "val", "volumes")

    tpaths = os.listdir(train_volume_path)
    for i, tpath in enumerate(tpaths):
        case = str(tpath).split(".")[0].split("-")[1]
        volume = join(train_volume_path, "volume-"+case+".tiff")
        label = join(train_label_path, "label-"+case+".tif")
        if not isfile(label):
            continue
        casename = task_name + case
        img_out_base = join(imagestr, casename)
        anno_out = join(labelstr, casename + ".nii.gz")
        res.append(
            p.starmap_async(load_tiff_convert_to_nifti, ((volume, label, img_out_base, anno_out, spacing),)))
        train_patient_names.append(casename)
        # if len(train_patient_names) >= 10:
        #     break

    vpaths = os.listdir(val_volume_path)
    for i, vpath in enumerate(vpaths):
        case = str(vpath).split(".")[0].split("-")[1]
        volume = join(val_volume_path, "volume-" + case + ".tiff")
        label = None
        casename = task_name + case
        img_out_base = join(imagests, casename)
        anno_out = None
        res.append(
            p.starmap_async(load_tiff_convert_to_nifti, ((volume, label, img_out_base, anno_out, spacing),)))
        test_patient_names.append(casename)
        # if len(test_patient_names) >= 6:
        #     break

    _ = [i.get() for i in res]

    json_dict = {}
    json_dict['name'] = task_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "AX",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "axon",
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    p.close()
    p.join()

    # pdb.set_trace()


if __name__ == "__main__":
    base = "/media/root/data4/zyt/train/155829/"
    task_id = 703
    task_name = 'DN155829'
    spacing = (1, 0.126, 0.126)
    prepare_task(base, task_id, task_name, spacing)
'''
    task_name = "Task0701_DNBrains"
    labelsTr = join(nnUNet_raw_data, task_name, "labelsTr")
    cases = subfiles(labelsTr, suffix='.nii.gz', join=False)
    splits = []
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('01_') or i.startswith('02_SIM')],
         'val': [i[:-7] for i in cases if i.startswith('02_') and not i.startswith('02_SIM')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('02_') or i.startswith('01_SIM')],
         'val': [i[:-7] for i in cases if i.startswith('01_') and not i.startswith('01_SIM')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('01_') or i.startswith('02_') and not i.startswith('02_SIM')],
         'val': [i[:-7] for i in cases if i.startswith('02_SIM')]}
    )
    splits.append(
        {'train': [i[:-7] for i in cases if i.startswith('02_') or i.startswith('01_') and not i.startswith('01_SIM')],
         'val': [i[:-7] for i in cases if i.startswith('01_SIM')]}
    )
    save_pickle(splits, join(preprocessing_output_dir, task_name, "splits_final.pkl"))
'''
