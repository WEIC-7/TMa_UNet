
from light_training.preprocessing.preprocessors.default_preprocessor_AbdomenAtlas1_0Mini import DefaultPreprocessor 
import numpy as np 
import pickle 
import json 


def process_train():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    base_dir = "/home/xingzhaohu/data/AbdomenAtlas1.0Mini"

    preprocessor = DefaultPreprocessor(base_dir=base_dir)

    out_spacing = [2.0, 0.8134765, 0.83007812]
    output_dir = "./data/fullres/train/"

    with open("./data_analysis_result.txt", "r") as f:
        content = f.read().strip("\n")
        print(content)
    content = eval(content)
    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                     num_processes=16,
                     foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel)

def process_val():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    base_dir = "./data/raw_data/Val"
    image_dir = "img"
    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    label_dir=None,
                                   )

    out_spacing = [0.5, 0.70410156, 0.70410156]

    with open("./data_analysis_result.txt", "r") as f:
        content = f.read().strip("\n")
        print(content)
    content = eval(content)
    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]

    output_dir = "./data/fullres/val_test/"
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir,
                     all_labels=[1, ],
                     foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel,
                     num_processes=16)

def process_test():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    base_dir = "/home/xingzhaohu/data/AbdomenAtlas1.0Mini_test"
    preprocessor = DefaultPreprocessor(base_dir=base_dir)

    out_spacing = [2.0, 0.8134765, 0.83007812]
    output_dir = "./data/fullres/test/"

    with open("./data_analysis_result.txt", "r") as f:
        content = f.read().strip("\n")
        print(content)
    content = eval(content)
    foreground_intensity_properties_per_channel = content["intensity_statistics_per_channel"]
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                     num_processes=16,
                     foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel)


def plan():
    base_dir = "/home/xingzhaohu/data/AbdomenAtlas1.0Mini"

    preprocessor = DefaultPreprocessor(base_dir=base_dir, 
                                    
                                   )

    preprocessor.run_plan()

if __name__ == "__main__":

    # plan()

    # process_train()

    process_test()
    # import time 
    # s = time.time()
    # process_val()
    # e = time.time()

    # print(f"preprocessing time is {e - s}")

    # process_val_semi()
    
    
# 
    # preprocessor.run(output_spacing=[3, 0.9765625, 0.9765625], output_dir=output_dir)

    # data = np.load("/home/xingzhaohu/sharefs/datasets/AIIB23_nnunet/train/AIIB23_96.npz")

    # image = data["data"]
    # label = data["seg"]
    # print(image.shape)
    # print(label.shape)

    # import matplotlib.pyplot as plt 

    # for i in range(20):
    #     plt.imshow(image[0, i], cmap="gray")
    #     plt.show()

    # df = open("/home/xingzhaohu/sharefs/datasets/AIIB23_nnunet/train/AIIB23_96.pkl", "rb")

    # info = pickle.load(df)
    # print(info)