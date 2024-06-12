from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = ["img.nii.gz"]
seg_filename = "seg.nii.gz"

base_dir = "data/"
image_dir = "new_dataset/"

def process_train():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "data/proc_dataset"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2]
    )

def plan():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    preprocessor.run_plan()

if __name__ == "__main__":

    plan()
    process_train()

