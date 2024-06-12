import os 
import shutil

base = "data/"
data_dir = base + "dataset"
new_dir = base + "new_dataset/"

list = os.listdir(data_dir)

Seg = os.path.join(data_dir,list[0])
Img = os.path.join(data_dir,list[1])

Seg_name = sorted(os.listdir(Seg))
Img_name = sorted(os.listdir(Img))

Img_len = len(Img_name)

for i in range(Img_len):
    path = new_dir + Img_name[i].split('.')[0]
    os.mkdir(path)

    Seg_path = os.path.join(Seg,Seg_name[i])
    Img_path = os.path.join(Img,Img_name[i])

    Seg_new_path = os.path.join(path,"seg.nii.gz")
    Img_new_path = os.path.join(path,"img.nii.gz")

    shutil.copy(Seg_path,Seg_new_path)
    shutil.copy(Img_path,Img_new_path)
    