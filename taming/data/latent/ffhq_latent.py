import numpy as np
from torch.utils.data import Dataset 



class LatentBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.arr_max = None
        self.arr_min = None
        self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        ori_item = self.data[i]
        #normailize to [-1, 1]
        norm_item = self.normalize(ori_item)
        return norm_item
    
    # 归一化到[-1,1]之间
    def normalize(self,x):
        return 2 * (x - self.arr_min) / (self.arr_max - self.arr_min) - 1

    # 反归一化
    def denormalize(self, normalized_x):
        return (normalized_x + 1) * (self.arr_max - self.arr_min) / 2 + self.arr_min

class FFHQTrain(LatentBase):
    def __init__(self, file_dir=None):
        super().__init__()
        # numpy memmap bug https://medium.com/dive-into-ml-ai/precautions-while-using-np-memmap-6e2b6c95e8ff
        # original_latent = np.memmap(file_dir, dtype='float32', mode='r')
        # original_latent = original_latent[32:].reshape((60000, 3, 64, 64 ))
        # original_latent = np.load(file_dir)

        self.data = np.lib.format.open_memmap(file_dir, dtype='float32', mode='r', shape=(60000, 3, 64, 64 ))
        self.arr_max =  self.data.max()
        self.arr_min =  self.data.min()
        self.length = 60000
       
        
class FFHQValidation(LatentBase):
    def __init__(self, file_dir=None):
        super().__init__()
        self.data = np.lib.format.open_memmap(file_dir, dtype='float32', mode='r', shape=(10000, 3, 64, 64 ))
        self.arr_max =  self.data.max()
        self.arr_min =  self.data.min()

        self.length = 10000


