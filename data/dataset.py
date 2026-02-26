import os
import random

import torch
import torch.multiprocessing
import torch.utils.data as data
from PIL import Image, ImageFilter
from PIL import ImageOps
from tifffile import imread
from torchvision import transforms


def find_max_number(folder_path):
    max_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            filename = filename[:-4]
        if not filename.isdigit():
            continue
        filename = int(filename)

        number = int(filename)
        max_number = max(max_number, number)
    return max_number


def find_max_folder_number(folder_path):
    max_number = 0
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            if not folder_name.isdigit():
                continue
            folder_name = int(folder_name)
            number = int(folder_name)
            max_number = max(max_number, number)
    return max_number


def pil_loader(path):
    return Image.open(path).convert('L')


def pil_loader_noL(path):
    return Image.open(path)


def invert(tensor):
    return 1 - tensor


class EMDiffusenDataset(data.Dataset):  # Denoise and super-resolution Dataset
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 loader=pil_loader, interpolation='bicubic'):
        self.data_root = data_root
        self.phase = phase
        self.img_paths, self.gt_paths = self.read_dataset(self.data_root)
        
        if interpolation == 'bicubic':
            interp_mode = transforms.InterpolationMode.BICUBIC
        elif interpolation == 'bilinear':
            interp_mode = transforms.InterpolationMode.BILINEAR
        elif interpolation == 'nearest':
            interp_mode = transforms.InterpolationMode.NEAREST
        else:
            interp_mode = transforms.InterpolationMode.BICUBIC

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1]), interpolation=interp_mode),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = self.img_paths[index]
        gt_file_name = self.gt_paths[index]
        img = self.loader(file_name)
        gt = self.loader(gt_file_name)
        if self.phase == 'train':
            img, gt = self.aug(img, gt)
        img = self.tfs(img)
        gt = self.tfs(gt)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.img_paths)

    def aug(self, img, gt):
        if random.random() < 0.5:
            img = ImageOps.flip(img)
            gt = ImageOps.flip(gt)
        if random.random() < 0.5:
            img = img.rotate(90)
            gt = gt.rotate(90)
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
        return img, gt

    def read_dataset(self, data_root):
        import os
        img_paths = []
        gt_paths = []
        for cell_num in os.listdir(data_root):
            if cell_num == '.DS_Store':
                continue
            cell_path = os.path.join(data_root, cell_num)
            for noise_level in os.listdir(cell_path):
                for img in sorted(os.listdir(os.path.join(cell_path, noise_level))):
                    if 'tif' in img:
                        img_paths.append(os.path.join(cell_path, noise_level, img))
                        gt_paths.append(os.path.join(cell_path, noise_level, img).replace('wf', 'gt'))
        return img_paths, gt_paths


class vEMDiffuseTrainingDatasetPatches(
    data.Dataset):  # Dataset for vEMDiffuse training using the downloaded subvolume
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 method='vEMDiffuse-i',
                 z_times=6, loader=pil_loader_noL):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.gt_paths, self.max_nums = self.read_dataset(self.data_root)
        self.percent = percent
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        # sub_volume_index = random.randint(0, len(self.gt_paths) - 1)

        gt_file_name = self.gt_paths[index]
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
        file_index = random.randint(upper_bound, self.max_nums[index] - lower_bound)
        gt_file_name = os.path.join(gt_file_name, str(file_index) + '.tif')
        img_up = self.loader(os.path.join(os.path.dirname(gt_file_name), str(file_index - upper_bound) + '.tif'))
        img_below = self.loader(os.path.join(os.path.dirname(gt_file_name), str(file_index + lower_bound) + '.tif'))
        gts = []
        for i in range(file_index - upper_bound + 1, file_index + lower_bound):
            gts.append(self.loader(os.path.join(os.path.dirname(gt_file_name), str(i) + '.tif')))
        img_up, img_below, gts = self.aug(img_up, img_below, gts)
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(gt_file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.gt_paths)

    def aug(self, img_up, img_below, gts):
        if random.random() < 0.5:
            img_up = img_up.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_below = img_below.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_up = ImageOps.flip(img_up)
            img_below = ImageOps.flip(img_below)
            for i in range(len(gts)):
                gt_tmp = ImageOps.flip(gts[i])
                gts[i] = gt_tmp
        if random.random() < 0.5:
            img_up = img_up.rotate(90)
            img_below = img_below.rotate(90)
            for i in range(len(gts)):
                gt_tmp = gts[i].rotate(90)
                gts[i] = gt_tmp
        return img_up, img_below, gts

    def read_dataset(self, data_root):
        gt_paths = []  # subfolder
        max_nums = []  # z_depth in subfolder
        for cell_num in os.listdir(data_root):
            cell_path = os.path.join(data_root, cell_num)
            gt_paths.append(cell_path)
            max_nums.append(find_max_number(os.path.join(cell_path)))
        return gt_paths, max_nums


class vEMDiffuseTrainingDatasetVolume(data.Dataset):  # Dataset for vEMDiffuse training using the whole isotropic volume
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 method='vEMDiffuse-i',
                 z_times=6, loader=pil_loader_noL):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.method = method
        self.percent = percent
        self.tfs = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.depth = find_max_number(self.data_root)
        print('the number of layers: ', self.depth + 1)
        self.height, self.width = imread(os.path.join(data_root, str(self.depth) + '.tif')).shape
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        x_index = random.randint(0, self.width - self.image_size[0])
        y_index = random.randint(0, self.height - self.image_size[1])
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
        z_index = random.randint(upper_bound, self.depth - lower_bound - 1)

        img_up = Image.fromarray(imread(os.path.join(self.data_root, str(z_index - upper_bound) + '.tif'))[
                                 y_index:y_index + self.image_size[1],
                                 x_index:x_index + self.image_size[0]])
        img_below = Image.fromarray(imread(os.path.join(self.data_root, str(z_index + lower_bound) + '.tif'))[
                                    y_index:y_index + self.image_size[1],
                                    x_index:x_index + self.image_size[0]])
        gts = []
        for i in range(z_index - upper_bound + 1, z_index + lower_bound):
            gts.append(Image.fromarray(
                imread(os.path.join(self.data_root, str(i) + '.tif'))[y_index:y_index + self.image_size[1],
                x_index:x_index + self.image_size[0]]))
        img_up, img_below, gts = self.aug(img_up, img_below, gts)
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = str(y_index) + '_' + str(x_index) + '_' + str(z_index) + '.tif'
        return ret

    def __len__(self):
        if self.phase == 'train':
            return 20000

    def aug(self, img_up, img_below, gts):
        if random.random() < 0.5:
            img_up = img_up.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_below = img_below.filter(ImageFilter.GaussianBlur(radius=3))
        if random.random() < 0.5:
            img_up = ImageOps.flip(img_up)
            img_below = ImageOps.flip(img_below)
            for i in range(len(gts)):
                gt_tmp = ImageOps.flip(gts[i])
                gts[i] = gt_tmp
        if random.random() < 0.5:
            img_up = img_up.rotate(90)
            img_below = img_below.rotate(90)
            for i in range(len(gts)):
                gt_tmp = gts[i].rotate(90)
                gts[i] = gt_tmp
        return img_up, img_below, gts


class CachedvEMDiffuseDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 method='vEMDiffuse-i',
                 z_times=6, loader=pil_loader_noL):
        self.data_root = data_root
        self.phase = phase
        self.image_size = image_size
        
        cache_file_path = os.path.join(self.data_root, 'cache_500.pt')
        print(f"Loading cached dataset from {cache_file_path}...")
        
        # Load the entire dataset into memory
        data_dict = torch.load(cache_file_path)
        self.gt_images = data_dict['gt_images']
        self.cond_images = data_dict['cond_images']
        self.paths = data_dict['paths']
        
        self.length = len(self.paths)
        print(f"Loaded {self.length} samples.")

    def __getitem__(self, index):
        ret = {}
        ret['gt_image'] = self.gt_images[index]
        ret['cond_image'] = self.cond_images[index]
        ret['path'] = self.paths[index]
        return ret

    def __len__(self):
        return self.length


class vEMDiffuseTestIsotropic(
    data.Dataset):  # dataset for testing vEMDiffuse-i. Given an isotropic volume, first downsample it and then reconstruct it.
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 z_times=6,
                 loader=pil_loader):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.is_flat = False # Flag to indicate if data is in flat structure
        self.gt_paths = self.read_dataset(self.data_root)
        self.percent = percent

        self.tfs = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        gt_file_name = self.gt_paths[index]
        
        if self.is_flat:
             # gt_file_name is absolute path like .../6.tif
             # We need to extract index from filename
             basename = os.path.basename(gt_file_name)
             file_index = int(os.path.splitext(basename)[0])
             
             upper_bound = self.z_times // 2
             lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
             
             # Construct paths for up, below, and gts
             # Assuming files are named {index}.tif in data_root
             
             path_up = os.path.join(self.data_root, f"{file_index - upper_bound}.tif")
             path_below = os.path.join(self.data_root, f"{file_index + lower_bound}.tif")
             
             img_up = self.loader(path_up)
             img_below = self.loader(path_below)
             
             gts = []
             for i in range(file_index - upper_bound + 1, file_index + lower_bound):
                 gts.append(self.loader(os.path.join(self.data_root, f"{i}.tif")))
                 
        else:
            file_index = int(gt_file_name.split(os.sep)[-2])
            upper_bound = self.z_times // 2
            lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1
            img_up = self.loader(os.path.join(self.data_root, str(file_index - upper_bound), gt_file_name.split('/')[-1]))
            img_below = self.loader(
                os.path.join(self.data_root, str(file_index + lower_bound), gt_file_name.split(os.sep)[-1]))
            gts = []
            for i in range(file_index - upper_bound + 1, file_index + lower_bound):
                gts.append(self.loader(os.path.join(self.data_root, str(i), gt_file_name.split(os.sep)[-1])))
                
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(gt_file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.gt_paths)

    def read_dataset(self, data_root):
        import os
        
        # Check if flat structure (files in root) or nested (folders in root)
        # We assume if we find 0.tif (or similar) in root, it is flat.
        # But find_max_folder_number checks for folders.
        
        # Try to find max number from files first
        max_file_num = find_max_number(data_root)
        max_folder_num = find_max_folder_number(data_root)
        
        gt_paths = []
        upper_bound = self.z_times // 2
        lower_bound = self.z_times // 2 if self.z_times % 2 == 0 else self.z_times // 2 + 1

        if max_file_num > 0 and max_folder_num == 0:
            self.is_flat = True
            z_depth = max_file_num
            # For flat structure, we iterate through indices that serve as "centers" or "anchors"
            # The logic in original code was: iterate i in steps of z_times.
            # i corresponds to the folder name, which seems to be the Z index?
            # Original: range(upper_bound, z_depth - lower_bound, self.z_times)
            # If we have 0..N tifs.
            # We want to sample pairs (0, 6) -> reconstruct 1..5.
            # The "center" of reconstruction is 3.
            # But the code uses `file_index` to determine up/below.
            # img_up = file_index - upper_bound
            # img_below = file_index + lower_bound
            # So if z_times=6, upper=3, lower=3.
            # file_index needs to be such that file_index-3 >= 0 and file_index+3 <= max.
            # So file_index should start at 3.
            # loop range: upper_bound to z_depth - lower_bound.
            
            # Note: The original loop was step=self.z_times.
            # This implies non-overlapping chunks?
            # Yes, likely.
            
            for i in range(upper_bound, z_depth - lower_bound + 1, self.z_times):
                 # We need to add a "path" representing this chunk.
                 # Since it's flat, we can just use the path to the "center" slice?
                 # Or just the index?
                 # The __getitem__ expects gt_file_name.
                 # Let's use the path to the i-th tif.
                 # Check if file exists
                 if os.path.exists(os.path.join(data_root, f"{i}.tif")):
                     gt_paths.append(os.path.join(data_root, f"{i}.tif"))
                     
        else:
            self.is_flat = False
            z_depth = max_folder_num
            for i in range(upper_bound, z_depth - lower_bound, self.z_times):  # Downsample
                if os.path.isdir(os.path.join(data_root, str(i))):
                    for file in os.listdir(os.path.join(data_root, str(i))):
                        gt_paths.append(os.path.join(data_root, str(i), file))
                        # below_paths.append(os.path.join(data_root, str(i + 1), file))
        return gt_paths


class vEMDiffuseTestAnIsotropic(
    data.Dataset):  # dataset for testing vEMDiffuse-i. Given an anisotropic volume, reconstruct it.
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256],
                 z_times=6,
                 loader=pil_loader):
        self.data_root = data_root
        self.phase = phase
        self.z_times = z_times
        self.gt_paths, self.below_paths = self.read_dataset(self.data_root)
        self.percent = percent

        self.tfs = transforms.Compose([
            # transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        gt_file_name = self.gt_paths[index]
        img_up = self.loader(gt_file_name)
        img_below = self.loader(self.below_paths[index])
        gts = []
        for i in range(self.z_times - 1):
            gts.append(self.loader(gt_file_name))
        img_up = self.tfs(img_up)
        img_below = self.tfs(img_below)
        out_gts = []
        for i in range(len(gts)):
            gt_tmp = self.tfs(gts[i])
            out_gts.append(gt_tmp)
        img = torch.cat([img_below, img_up], dim=0)
        gt = torch.cat(out_gts, dim=0)
        ret['gt_image'] = gt
        ret['cond_image'] = img
        ret['path'] = '_'.join(gt_file_name.split(os.sep)[-3:])
        return ret

    def __len__(self):
        return len(self.gt_paths)

    def read_dataset(self, data_root):
        import os
        z_depth = find_max_folder_number(data_root)
        gt_paths = []
        below_paths = []
        for i in range(0, z_depth):
            for file in os.listdir(os.path.join(data_root, str(i))):
                gt_paths.append(os.path.join(data_root, str(i), file))
                below_paths.append(os.path.join(data_root, str(i + 1), file))
                # below_paths.append(os.path.join(data_root, str(i + 1), file))
        return gt_paths, below_paths
