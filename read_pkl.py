import pickle
import pprint

# file = open("../DATASET/nnUNet_preprocessed/Task708_DN_match_Renlab/nnUNetPlansv2.1_16GB_plans_3D.pkl", "rb")
file = open("../DATASET/nnUNet_preprocessed/Task7095_Renlab_match_cutmix_aug_plus/nnUNetPlansv2.1_16GB_plans_3D.pkl", "rb")
plan = pickle.load(file)
pprint.pprint(plan)
file.close()
