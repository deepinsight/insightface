from PIL import Image
import os
import torch.utils.data as data
import torchvision.transforms as transforms


class Real_Dataset_112(data.Dataset):
    def __init__(self, args):
        super(Real_Dataset_112, self).__init__()
        
        self.img_root = args.img_root_R
        self.img_list, self.num_classes = self.list_reader(args.train_list_R)
        self.input_mode = args.input_mode

        self.transform = transforms.Compose([
            # transforms.RandomCrop(112),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_name, label = self.img_list[index]

        img = self.get_img_from_path(img_name)
        return {'img': img, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)

    def get_img_from_path(self, img_name):
        img_path = os.path.join(self.img_root, img_name)

        if self.input_mode == 'grey':
            img = Image.open(img_path).convert('L')
        elif self.input_mode == 'red':
            img = Image.open(img_path)
            img = img.split()[0]

        img = self.transform(img)
        return img

    def list_reader(self, list_file):
        img_list = []
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        pid_container = set()
        for line in lines:
            pid = int(line.strip().split(' ')[1])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        for line in lines:
            img_name, pid = line.strip().split(' ')
            if not os.path.exists(os.path.join(self.img_root, img_name)):
                continue
            label = pid2label[int(pid)]
            img_list.append((img_name, label))
        
        return img_list, len(pid_container)


class Real_Dataset_112_paired(data.Dataset):
    def __init__(self, args):
        super(Real_Dataset_112_paired, self).__init__()

        self.img_root = args.img_root_R
        self.img_list, self.num_classes = self.list_reader(args.train_list_R)
        self.input_mode = args.input_mode

        self.transform = transforms.Compose([
            # transforms.RandomCrop(112),
            transforms.ToTensor()
        ])

        self.vir_list = [(a,b,c) for (a,b,c) in self.img_list if c==0]
        self.nir_list = [(a,b,c) for (a,b,c) in self.img_list if c==1]

        self.vis_labels = np.array([p[1] for p in self.vir_list])
        self.nir_labels = np.array([p[1] for p in self.nir_list])

        self.visIndex = None
        self.nirIndex = None

    def __getitem__(self, index):
        vis_img_name, vis_label, vis_domain = self.vir_list[self.visIndex[index]]
        nir_img_name, nir_label, nir_domain = self.nir_list[self.nirIndex[index]]
        
        assert vis_domain == 0 and nir_domain == 1

        vis_img = self.get_img_from_path(vis_img_name)
        nir_img = self.get_img_from_path(nir_img_name)

        return vis_img, nir_img, vis_label, nir_label

    def __len__(self):
        return len(self.img_list)

    def get_img_from_path(self, img_name):
        img_path = os.path.join(self.img_root, img_name)

        if self.input_mode == 'grey':
            img = Image.open(img_path).convert('L')
        elif self.input_mode == 'red':
            img = Image.open(img_path)
            img = img.split()[0]

        img = self.transform(img)
        return img


    def list_reader(self, list_file):
        img_list = []
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        pid_container = set()
        for line in lines:
            pid = int(line.strip().split(' ')[1])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        for line in lines:
            img_name, pid = line.strip().split(' ')
            label = pid2label[int(pid)]
    
            domain = 0 if 'VIS' in img_name else 1 
            img_list.append((img_name, label, domain))
        
        return img_list, len(pid_container)

class Mix_Dataset_112(data.Dataset):
    def __init__(self, args):
        super(Mix_Dataset_112, self).__init__()
        
        self.img_root_R = args.img_root_R
        self.img_root_F = args.img_root_F
        self.img_list, self.num_classes = self.list_reader(args.train_list_R, args.train_list_F)
        self.input_mode = args.input_mode

        self.transform = transforms.Compose([
            # transforms.RandomCrop(112),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path, label = self.img_list[index]

        img = self.get_img_from_path(img_path)
        return {'img': img, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)

    def get_img_from_path(self, img_path):

        if self.input_mode == 'grey':
            img = Image.open(img_path).convert('L')
        elif self.input_mode == 'red':
            img = Image.open(img_path)
            img = img.split()[0]

        img = self.transform(img)
        return img

    def list_reader(self, list_file_real, list_file_fake):
        with open(list_file_real, 'r') as f:
            lines_real = f.readlines()
        with open(list_file_fake, 'r') as f:
            lines_fake = f.readlines()

        fake_label_start = max([int(l.strip().split(' ')[-1]) for l in lines_real]) + 1
        lines_fake = ["{} {}".format(l.strip().split(' ')[0], int(l.strip().split(' ')[1]) + fake_label_start) for l in lines_fake]

        lines = lines_real + lines_fake

        pid_container = set()
        for line in lines:
            pid = int(line.strip().split(' ')[1])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        img_list_R = []
        for line in lines_real:
            img_name, pid = line.strip().split(' ')
            label = pid2label[int(pid)]
            img_list_R.append((os.path.join(self.img_root_R + img_name), label))
        
        img_list_F = []
        for line in lines_fake:
            img_name, pid = line.strip().split(' ')
            label = pid2label[int(pid)]
            img_list_F.append((os.path.join(self.img_root_F + img_name), label))
        
        img_list = img_list_R + img_list_F

        return img_list, len(pid_container)


class Mix_Dataset_112_paired(data.Dataset):
    def __init__(self, args):
        super(Mix_Dataset_112_paired, self).__init__()

        self.img_root_R = args.img_root_R
        self.img_root_F = args.img_root_F
        self.img_list, self.num_classes = self.list_reader(args.train_list_R, args.train_list_F)
        self.input_mode = args.input_mode

        self.transform = transforms.Compose([
            # transforms.RandomCrop(112),
            transforms.ToTensor()
        ])

        self.vir_list = [(a,b,c) for (a,b,c) in self.img_list if c==0]
        self.nir_list = [(a,b,c) for (a,b,c) in self.img_list if c==1]

        self.vis_labels = np.array([p[1] for p in self.vir_list])
        self.nir_labels = np.array([p[1] for p in self.nir_list])

        self.visIndex = None
        self.nirIndex = None

    def __getitem__(self, index):
        vis_img_name, vis_label, vis_domain = self.vir_list[self.visIndex[index]]
        nir_img_name, nir_label, nir_domain = self.nir_list[self.nirIndex[index]]
        
        assert vis_domain == 0 and nir_domain == 1

        vis_img = self.get_img_from_path(vis_img_name)
        nir_img = self.get_img_from_path(nir_img_name)

        return vis_img, nir_img, vis_label, nir_label

    def __len__(self):
        return len(self.img_list)

    def get_img_from_path(self, img_path):
        if self.input_mode == 'grey':
            img = Image.open(img_path).convert('L')
        elif self.input_mode == 'red':
            img = Image.open(img_path)
            img = img.split()[0]

        img = self.transform(img)
        return img


    def list_reader(self, list_file_real, list_file_fake):
        with open(list_file_real, 'r') as f:
            lines_real = f.readlines()
        with open(list_file_fake, 'r') as f:
            lines_fake = f.readlines()

        fake_label_start = max([int(l.strip().split(' ')[-1]) for l in lines_real]) + 1
        lines_fake = ["{} {}".format(l.strip().split(' ')[0], int(l.strip().split(' ')[1]) + fake_label_start) for l in lines_fake]

        lines = lines_real + lines_fake

        pid_container = set()
        for line in lines:
            pid = int(line.strip().split(' ')[1])
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        img_list_R = []
        for line in lines_real:
            img_name, pid = line.strip().split(' ')
            label = pid2label[int(pid)]
            domain = 0 if 'VIS' in img_name else 1
            img_list_R.append((os.path.join(self.img_root_R + img_name), label, domain))
        
        img_list_F = []
        for line in lines_fake:
            img_name, pid = line.strip().split(' ')
            label = pid2label[int(pid)]

            # if label in [8192,1984,2110,6344,8566,8589,9362]:         # only with single image pair
            #     print(img_name)
            domain = 0 if 'VIS' in img_name else 1
            img_list_F.append((os.path.join(self.img_root_F + img_name), label, domain))
        
        img_list = img_list_R + img_list_F

        return img_list, len(pid_container)


from torch.utils.data.sampler import Sampler
import numpy as np

def GenIdx(train_vis_label, train_nir_label):
    def get_idx_from_label(train_label):
        pos = []
        unique_train_label = np.unique(train_label)
        for ul in unique_train_label:
            tmp = np.argwhere(train_label == ul).squeeze().tolist()
            if isinstance(tmp,int):
                tmp = [tmp]
            pos.append(tmp)
        return pos
    
    vis_pos = get_idx_from_label(train_vis_label)
    nir_pos = get_idx_from_label(train_nir_label)

    return vis_pos, nir_pos


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, num_img_per_id = 4):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        
        num_id_per_batch = batchSize / num_img_per_id

        for j in range(N//batchSize+1):
            batch_idx = np.random.choice(uni_label, int(num_id_per_batch), replace=False)
            
            for s, i in enumerate(range(0, batchSize, num_img_per_id)):
                sample_flag = True if len(color_pos[batch_idx[s]]) < num_img_per_id or len(thermal_pos[batch_idx[s]]) < num_img_per_id else False
                
                sample_color[i:i+num_img_per_id]  = np.random.choice(color_pos[batch_idx[s]], num_img_per_id, replace=sample_flag)
                sample_thermal[i:i+num_img_per_id] = np.random.choice(thermal_pos[batch_idx[s]], num_img_per_id, replace=sample_flag)
            
            if j ==0:
                index1= sample_color
                index2= sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        
        self.visIndex = index1
        self.nirIndex = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.visIndex)))

    def __len__(self):
        return self.N 