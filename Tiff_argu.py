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
from PIL import Image
import random
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
from nnunet.paths import preprocessing_output_dir
from skimage.io import imread
import pdb
from tqdm import tqdm
import myutils
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import tifffile

join = os.path.join
artifact = True
dim = 64
sample = 10
datapath = "/media/root/data4/zyt/train/155829"
isTrain = True


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

def np_convert_to_nifti(data, label, img_out_base, anno_out, spacing):
    img_itk = sitk.GetImageFromArray(data.astype(np.float32))
    # img_itk.SetSpacing(np.array(spacing)[::-1])
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if label is not None:
        l_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # l_itk.SetSpacing(np.array(spacing)[::-1])
        sitk.WriteImage(l_itk, anno_out)


def make_data(base, task_id, task_name, spacing, n_samples, input_dim, data_mix, flag):
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

    datas = []
    labels = []
    data_ori = []
    data_path = join(base, 'train') if flag else join(base, 'val')

    volumes_folder_path = join(data_path, "volumes")
    labels_folder_path = join(data_path, "labels_sk") if flag else join(data_path, 'labels')

    volumes_path = myutils.get_dir(volumes_folder_path)
    labels_path = myutils.get_dir(labels_folder_path)

    assert len(labels_path) == len(volumes_path)
    if n_samples == None:
        n_samples = len(labels_path)

    callfunc = {
        0: lambda: [np.fliplr(volume_rot), np.fliplr(label_rot)],
        1: lambda: [np.flipud(volume_rot), np.flipud(label_rot)],
    }

    total_volumes = 0
    with tqdm(total=len(volumes_path) * n_samples, desc=f'volumes numbers') as pbar:
        for vpath, lpath in zip(volumes_path, labels_path):
            case = str(vpath).split(".")[0].split("-")[1]
            casename = task_name + case
            img_out_base = join(imagestr, casename)
            anno_out_base = join(labelstr, casename)

            volume = myutils.read_tiff_stack(vpath)
            label = myutils.read_tiff_stack(lpath)
            if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                    or volume.shape[2] < input_dim:
                continue
            for j in range(n_samples):
                img_out = join(img_out_base + str(j))
                anno_out = join(anno_out_base + str(j) + ".nii.gz")
                casename_s = casename + str(j)
                z = random.randint(0, label.shape[0] - input_dim)
                x = random.randint(0, label.shape[1] - input_dim)
                y = random.randint(0, label.shape[2] - input_dim)
                volume_chunk = volume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                label_chunk = label[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                if random.randint(0, 1) == 0:
                    volume_chunk = myutils.contrast_augmentation(volume_chunk, label_chunk, rad=15, N=3)

                k_seed = random.randint(0, 3)
                flip_seed = random.randint(0, 1)

                volume_rot = np.rot90(np.swapaxes(volume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                label_rot = np.rot90(np.swapaxes(label_chunk, 0, 2), k=k_seed).swapaxes(2, 0)

                data, annotation = callfunc[flip_seed]()
                # data_ori = ((data - data.min()) / (data.max() - data.min()))[np.newaxis, :, :, :]
                data_ori = ((data - data.min()) / (data.max() - data.min()))
                annotation[annotation > 0] = 1

                # print(data_ori.shape)
                # pdb.set_trace()
                datas.append(data_ori[np.newaxis, :, :, :].astype(np.float32))
                labels.append(annotation[np.newaxis, ...].astype(np.float32))
                res.append(
                    p.starmap_async(np_convert_to_nifti, ((data_ori, annotation, img_out, anno_out, spacing),)))
                train_patient_names.append(casename_s)

                total_volumes += 1
                pbar.update()
        total_volumes_axon = total_volumes
        if data_mix:
            artifacts_folder_path = data_path + '/artifacts/'
            artifacts_path = myutils.get_dir(artifacts_folder_path)
            with tqdm(total=len(volumes_path) * n_samples, desc=f'artifacts numbers') as pbar:
                for apath in artifacts_path:
                    case_a = str(apath).split(".")[0].split("-")[1]
                    casename_a = task_name + "a" + case_a
                    a_out_base = join(imagestr, casename_a)
                    l_out_base = join(labelstr, casename_a)
                    ak_seed = random.randint(0, 3)
                    artifact = myutils.read_tiff_stack(apath)
                    if artifact.shape[0] < input_dim or artifact.shape[1] < input_dim \
                            or artifact.shape[2] < input_dim:
                        print(artifact.shape)
                        continue
                    # for k in range(max(1, int(len(volumes_path) / len(artifacts_path) * n_samples))):
                    for k in range(max(1, int(len(volumes_path) / len(artifacts_path) * n_samples / 2))):
                        a_out = join(a_out_base + str(k))
                        l_out = join(l_out_base + str(k) + ".nii.gz")
                        casename_as = casename_a + str(k)
                        z = random.randint(0, artifact.shape[0] - input_dim)
                        x = random.randint(0, artifact.shape[1] - input_dim)
                        y = random.randint(0, artifact.shape[2] - input_dim)
                        artifact = artifact[z:z + input_dim, x:x + input_dim, y:y + input_dim]
                        artifact = np.rot90(np.swapaxes(artifact, 0, 2), k=ak_seed).swapaxes(2, 0)
                        # artifact = equal(artifact, 0.9)
                        artifact = (artifact - artifact.min()) / (artifact.max() - artifact.min())
                        data_ori = artifact.copy()[np.newaxis, :, :, :]
                        if random.randint(0, 1) == 0:
                            mix_seed = random.random() * 0.2 + 0.4  # from 0.4 to 0.6
                            idx = random.randint(0, total_volumes_axon - 1)
                            data_axon = datas[idx]
                            data_ori = (data_ori * mix_seed + data_axon * (1 - mix_seed))
                            annotation = labels[idx][0]
                        else:
                            annotation[annotation > 0] = 0
                        datas.append(data_ori.astype(np.float32))
                        labels.append(annotation[np.newaxis, ...].astype(np.float32))
                        res.append(
                            p.starmap_async(np_convert_to_nifti, ((data_ori[0], annotation, a_out, l_out, spacing),)))
                        train_patient_names.append(casename_as)

                        total_volumes += 1
                        pbar.update()

        val_volume_path = join(base, "val", "volumes")
        vpaths = os.listdir(val_volume_path)
        for i, vpath in enumerate(vpaths):
            case = str(vpath).split(".")[0].split("-")[1]
            volume = join(val_volume_path, "volume-" + case + ".tif")
            label = None
            casename = task_name + case
            img_out_base = join(imagests, casename)
            anno_out = None
            res.append(
                p.starmap_async(load_tiff_convert_to_nifti, ((volume, label, img_out_base, anno_out, spacing),)))
            test_patient_names.append(casename)

        _ = [i.get() for i in res]

        json_dict = {}
        json_dict['name'] = task_name
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "MI",  # microscope image
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


def histogram_match_data(base, source, task_id, task_name, spacing, n_samples, input_dim,
                         data_mix, flag, match_flag, cutmix, join):
    p = Pool(16)
    augment = myutils.dataAugmentation()

    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    origin = join(out_base, "origin")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(origin)

    train_patient_names = []
    test_patient_names = []
    res = []

    casenames = []

    datas = []
    labels = []
    datas_ori = []
    labels_ori = []
    data_path = join(base, 'train') if flag else join(base, 'val')

    volumes_folder_path = join(data_path, "volumes")
    labels_folder_path = join(data_path, "labels_sk") if flag else join(data_path, 'labels')
    volumes_path = myutils.get_dir(volumes_folder_path)
    labels_path = myutils.get_dir(labels_folder_path)
    assert len(labels_path) == len(volumes_path)
    if n_samples == None:
        n_samples = len(labels_path)
    artifacts_folder_path = data_path + '/artifacts/'
    artifacts_path = myutils.get_dir(artifacts_folder_path)
    if match_flag:
        source_path = join(source, 'train')  # # match to val cubes
        volumes_folder_path_s = join(source_path, "volumes")
        volumes_path_s = myutils.get_dir(volumes_folder_path_s)
        labels_folder_path_s = join(source_path, "labels_sk")
        labels_path_s = myutils.get_dir(labels_folder_path_s)
        artifacts_folder_path_s = source_path + '/artifacts/'
        artifacts_path_s = myutils.get_dir(artifacts_folder_path_s)

    total_ori_volumes = 0
    total_volumes = 0
    with tqdm(total=(len(volumes_path) + len(artifacts_path))*2, desc=f'original volume numbers') as pbar:
        for vpath, lpath in zip(volumes_path, labels_path):
            case = str(vpath).split(".")[0].split("-")[1]
            casename = task_name + case
            casenames.append(casename)

            volume = myutils.read_tiff_stack(vpath)
            label = myutils.read_tiff_stack(lpath)
            if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                    or volume.shape[2] < input_dim:
                continue
            datas_ori.append(volume)
            labels_ori.append(label)
            total_ori_volumes += 1
            pbar.update()
        axon_num = total_ori_volumes
        print("axon_num: ", axon_num)
        if match_flag:
            for i in range(axon_num):
                casename = casenames[i] + "m"
                casenames.append(casename)
                volume = datas_ori[i].copy()
                label = labels_ori[i].copy()
                match_ref_path = random.choice(volumes_path_s)
                match_ref = myutils.read_tiff_stack(match_ref_path)
                volume_match = match_histograms(volume, match_ref)
                datas_ori.append(volume_match)
                labels_ori.append(label)
                total_ori_volumes += 1
                pbar.update()
        aug_axon_num = total_ori_volumes
        print("aug_axon_num: ", aug_axon_num)
        if data_mix:
            for apath in artifacts_path:
                case_a = str(apath).split(".")[0].split("-")[1]
                casename_a = task_name + "a" + case_a
                casenames.append(casename_a)

                ak_seed = random.randint(0, 3)
                artifact = myutils.read_tiff_stack(apath)
                if artifact.shape[0] < input_dim or artifact.shape[1] < input_dim \
                        or artifact.shape[2] < input_dim:
                    print(artifact.shape)
                    continue
                datas_ori.append(artifact)
                labels_ori.append(np.zeros_like(label))
                total_ori_volumes += 1
                pbar.update()
            junk_num = total_ori_volumes - aug_axon_num
            print("junk_num", junk_num)
            if match_flag:
                for i in range(junk_num):
                    casename = casenames[i+aug_axon_num] + "m"
                    casenames.append(casename)
                    volume = datas_ori[i+aug_axon_num].copy()
                    label = labels_ori[i+aug_axon_num].copy()
                    match_ref_path = random.choice(artifacts_path_s)
                    match_ref = myutils.read_tiff_stack(match_ref_path)
                    volume_match = match_histograms(volume, match_ref)
                    datas_ori.append(volume_match)
                    labels_ori.append(label)
                    total_ori_volumes += 1
                    pbar.update()
                aug_junk_num = total_ori_volumes - aug_axon_num
            if cutmix:
                for j in range(axon_num):
                    casename = casenames[j] + "c"
                    casenames.append(casename)
                    m_seed = random.randint(0, 1)
                    volume = datas_ori[j+axon_num].copy() if match_flag and m_seed else datas_ori[j].copy()
                    label = labels_ori[j+axon_num].copy() if match_flag and m_seed else labels_ori[j].copy()
                    a_index = random.randint(0, junk_num) + aug_axon_num
                    artifact_cut = datas_ori[a_index+junk_num] if match_flag and m_seed else datas_ori[a_index]
                    z = random.randint(0, label.shape[0])
                    x = random.randint(0, label.shape[1])
                    y = random.randint(0, label.shape[2])
                    artifact_chunk = artifact_cut[:z, :x, :y].copy()
                    volume[:z, :x, :y] = artifact_chunk
                    label[:z, :x, :y] = np.zeros_like(artifact_chunk)
                    tifffile.imwrite(join(origin, casename)+"_vol.tiff", np.array(volume).astype(np.uint16))
                    tifffile.imwrite(join(origin, casename) + "_lab.tiff", np.array(label).astype(np.uint16))
                    datas_ori.append(volume)
                    labels_ori.append(label)
                    total_ori_volumes += 1
            if join:
                for vpath_s, lpath_s in zip(volumes_path_s, labels_path_s):
                    case = str(vpath_s).split(".")[0].split("-")[1]
                    casename = task_name + case
                    casenames.append(casename)

                    volume = myutils.read_tiff_stack(vpath_s)
                    label = myutils.read_tiff_stack(lpath_s)
                    if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                            or volume.shape[2] < input_dim:
                        continue
                    datas_ori.append(volume)
                    labels_ori.append(label)
                    total_ori_volumes += 1
                    pbar.update()
                if data_mix:
                    for apath_s in artifacts_path_s:
                        case_a = str(apath_s).split(".")[0].split("-")[1]
                        casename_a = task_name + "a" + case_a
                        casenames.append(casename_a)

                        ak_seed = random.randint(0, 3)
                        artifact = myutils.read_tiff_stack(apath_s)
                        if artifact.shape[0] < input_dim or artifact.shape[1] < input_dim \
                                or artifact.shape[2] < input_dim:
                            print(artifact.shape)
                            continue
                        datas_ori.append(artifact)
                        labels_ori.append(np.zeros_like(label))
                        total_ori_volumes += 1
                        pbar.update()
    print("{} data of original size finish.".format(total_ori_volumes))

    with tqdm(total=(total_ori_volumes * n_samples), desc='volumes numbers') as pbar:
        for index in range(total_ori_volumes):
            img_out_base = join(imagestr, casenames[index])
            anno_out_base = join(labelstr, casenames[index])
            if index < aug_axon_num or index >= aug_axon_num + aug_junk_num:
                volume = datas_ori[index].copy()
                label = labels_ori[index].copy()
                for j in range(n_samples):
                    img_out = join(img_out_base + str(j))
                    anno_out = join(anno_out_base + str(j) + ".nii.gz")
                    casename_s = casenames[index] + str(j)
                    z = random.randint(0, label.shape[0] - input_dim)
                    x = random.randint(0, label.shape[1] - input_dim)
                    y = random.randint(0, label.shape[2] - input_dim)
                    volume_chunk = volume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    label_chunk = label[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    # ------------------------------------------------------------------ #
                    # if random.randint(0, 1) == 0:
                    #     volume_chunk = myutils.contrast_augmentation(volume_chunk, label_chunk, rad=15, N=3)
                    #
                    # ---------------- rotation ---------------- #
                    callfunc = {
                        0: lambda: [np.fliplr(volume_rot), np.fliplr(label_rot)],
                        1: lambda: [np.flipud(volume_rot), np.flipud(label_rot)],
                    }
                    k_seed = random.randint(0, 3)
                    flip_seed = random.randint(0, 1)
                    volume_rot = np.rot90(np.swapaxes(volume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    label_rot = np.rot90(np.swapaxes(label_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    volume_chunk, label_chunk = callfunc[flip_seed]()
                    # data augmentation
                    data, annotation = augment.data_augmentation(volume_chunk, label_chunk)
                    # ------------------------------------------------------------------ #

                    # data_ori = ((data - data.min()) / (data.max() - data.min()))[np.newaxis, :, :, :]
                    data = data / 6553
                    data_ori = ((data - data.min()) / (data.max() - data.min()))
                    annotation[annotation > 0] = 1

                    datas.append(data_ori[np.newaxis, :, :, :].astype(np.float32))
                    labels.append(annotation[np.newaxis, ...].astype(np.float32))
                    res.append(
                        p.starmap_async(np_convert_to_nifti, ((data_ori, annotation, img_out, anno_out, spacing),)))
                    train_patient_names.append(casename_s)

                    total_volumes += 1
                    pbar.update()
                total_volumes_axon = total_volumes
                continue
            else:
                artifact = datas_ori[index].copy()
                for k in range(max(1, int(len(volumes_path) / len(artifacts_path) * n_samples / 2))):
                    a_out = join(img_out_base + str(k))
                    l_out = join(anno_out_base + str(k) + ".nii.gz")
                    casename_as = casenames[index] + str(k)
                    z = random.randint(0, artifact.shape[0] - input_dim)
                    x = random.randint(0, artifact.shape[1] - input_dim)
                    y = random.randint(0, artifact.shape[2] - input_dim)
                    artifact = artifact[z:z + input_dim, x:x + input_dim, y:y + input_dim]
                    artifact = np.rot90(np.swapaxes(artifact, 0, 2), k=ak_seed).swapaxes(2, 0)
                    # artifact = equal(artifact, 0.9)
                    artifact = (artifact - artifact.min()) / (artifact.max() - artifact.min())
                    # data_ori = artifact.copy()[np.newaxis, :, :, :]
                    data_ori = artifact.copy()
                    # if random.randint(0, 1) == 0:
                    #     mix_seed = random.random() * 0.2 + 0.4  # from 0.4 to 0.6
                    #     idx = random.randint(0, total_volumes_axon - 1)
                    #     data_axon = datas[idx]
                    #     data_ori = (data_ori * mix_seed + data_axon * (1 - mix_seed))
                    #     annotation = labels[idx][0]
                    # else:
                    annotation[annotation > 0] = 0
                    datas.append(data_ori[np.newaxis, :, :, :].astype(np.float32))
                    labels.append(annotation[np.newaxis, ...].astype(np.float32))
                    res.append(
                        p.starmap_async(np_convert_to_nifti, ((data_ori, annotation, a_out, l_out, spacing),)))
                    train_patient_names.append(casename_as)

                    total_volumes += 1
                    pbar.update()
                continue
        print("train {} cases finish".format(total_volumes))
        val_volume_path = join(base, "val", "volumes")
        # val_volume_path = join(source, "val", "volumes") if match_flag else join(base, "val", "volumes")
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

        _ = [i.get() for i in res]

        json_dict = {}
        json_dict['name'] = task_name
        json_dict['description'] = ""
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = ""
        json_dict['licence'] = ""
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "MI",  # microscope image
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
    val_volume_path = join(base, "train", "volumes")

    '''
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
    '''

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
    print("Test cases finish.")

    tpaths = os.listdir(labelstr)
    for i, tpath in enumerate(tpaths):
        case = str(tpath).split(".")[0]
        train_patient_names.append(case)
    print("Train cases finish.")

    _ = [i.get() for i in res]

    json_dict = {}
    json_dict['name'] = task_name
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MI",
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
    base = "/media/root/data4/zyt/train/Renlab/"
    source = "/media/root/data4/zyt/train/Renlab/"
    task_id = 7096
    task_name = 'Renlab_match_cutmix_nojunk'
    spacing = (1, 0.126, 0.126)
    histogram_match_data(base, source, task_id, task_name, spacing, 6, 128,
                         False, True, True, True, True)  # artifact mix, train, histogram match, cutmix, join(s&t)
    # make_data(source, task_id, task_name, spacing, 10, 128, True, True)  # for train
    # prepare_task(base, task_id, task_name, spacing)

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
