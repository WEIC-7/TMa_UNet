import os
import numpy as np
import nibabel as nib
import binary

class Cal:
    def __init__(self, path):
        self.path = path
        self.pre_list = os.listdir(path)
        self.pre_paths = sorted([os.path.join(self.path, i) for i in self.pre_list if i != '.ipynb_checkpoints'])
        
        self.BASE_Gt = 'GT/'
        self.GT_list = os.listdir(self.BASE_Gt)
        self.ori_paths = sorted([os.path.join(self.BASE_Gt, i) for i in self.GT_list if i != '.ipynb_checkpoints'])
        self.length = len(self.ori_paths)
        
    def compute_tp_fp_fn_tn(self, mask_ref, mask_pred):
        use_mask = mask_ref  
        tp = np.sum((mask_ref & mask_pred) & use_mask)
        fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
        fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
        tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
        return tp, fp, fn, tn
    
    def Dice(self): 
        Dice = 0 
        for i in range(self.length):
            pre_mask = nib.load(self.pre_paths[i]).get_fdata().astype(bool)
            ori_mask = nib.load(self.ori_paths[i]).get_fdata().astype(bool)
            tp, fp, fn, tn = self.compute_tp_fp_fn_tn(ori_mask, pre_mask)
            dice = 2 * tp / (2 * tp + fp + fn)
            Dice += dice
        return f'Dice: {Dice/self.length}'

    def Jaccard(self):
        Jaccard = 0 
        for i in range(self.length):
            pre_mask = nib.load(self.pre_paths[i]).get_fdata().astype(bool)
            ori_mask = nib.load(self.ori_paths[i]).get_fdata().astype(bool)
            tp, fp, fn, tn = self.compute_tp_fp_fn_tn(ori_mask, pre_mask)
            jaccard =  tp / (tp + fp +fn)
            Jaccard += jaccard
        return f'Jaccard: {Jaccard/self.length}'

    def PPV(self):
        PPV = 0 
        for i in range(self.length):
            pre_mask = nib.load(self.pre_paths[i]).get_fdata().astype(bool)
            ori_mask = nib.load(self.ori_paths[i]).get_fdata().astype(bool)
            tp, fp, fn, tn = self.compute_tp_fp_fn_tn(ori_mask, pre_mask)
            ppv =  tp / (tp + fp)
            PPV += ppv
   
        return f'PPV: {PPV/self.length}'

    
    def HD95(self):
        HD95_l = 0 
        HD95_r = 0 
        for i in range(self.length):
            pre_mask = nib.load(self.pre_paths[i]).get_fdata().astype(bool)
            ori_mask = nib.load(self.ori_paths[i]).get_fdata().astype(bool)

            pre_mask_l = pre_mask[:,:,:pre_mask.shape[2]//2]
            pre_mask_r = pre_mask[:,:,pre_mask.shape[2]//2:]

            ori_mask_l = ori_mask[:,:,:ori_mask.shape[2]//2]
            ori_mask_r = ori_mask[:,:,ori_mask.shape[2]//2:]

            hd_l = binary.hd95(pre_mask_l,ori_mask_l)
            hd_r = binary.hd95(pre_mask_r,ori_mask_r)
            
            
            HD95_l += hd_l
            HD95_r += hd_r
               
        return f'HD95_l: {HD95_l/self.length}\nHD95_r: {HD95_r/self.length} '

