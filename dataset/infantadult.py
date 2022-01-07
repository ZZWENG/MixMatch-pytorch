import csv, json, os
from collections import defaultdict
from glob import glob
import numpy as np
import torch
import scipy.io as sio
from PIL import Image
import cv2

from im_utils import read_keypoints, cropout_openpose_one_third, crop

infant_root = r'/home/groups/syyeung/zzweng/code/Infant-Pose-Estimation/data/coco'
infant_anns_path = os.path.join(infant_root, r'annotations/200R_1000S/person_keypoints_train_infant.json')
infant_img_path = os.path.join(infant_root, r'images/train_infant')

mpii_img_path = r'/scratch/users/zzweng/datasets/pose_estimation/mpii_human_pose/images'
mpii_anns_path = r'/scratch/users/zzweng/datasets/pose_estimation/mpii_human_pose/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'


class LabeledDataset(object):
    def __init__(self, transform, num_per_class=200, include_synthetic=False):
        """[Loads infants from Infant-Pose-Estimation, and adults from MPII]

        Args:
            transform ([type]): [description]
            include_synthetic (bool, optional): [Whether include the synthetic infant images.]. Defaults to False.
        """
        self.infant_root = infant_root
        self.transforms = transform

        # Load infant meta data.
        infant_imgs = {}
        with open(os.path.join(self.infant_root, infant_anns_path), 'r') as f:
            infant_anns = json.load(f)

        for image_meta in infant_anns['images']:
            infant_imgs[image_meta['id']] = image_meta

        if not include_synthetic:
            self.infant_idxs = []
            for i, ann in enumerate(infant_anns['annotations']):
                if not infant_imgs[ann['id']]['is_synthetic']:
                    self.infant_idxs.append(i)
        else:
            self.infant_idxs = list(range(len(infant_anns)))

        self.infant_imgs = infant_imgs
        self.infant_anns = infant_anns['annotations'][:num_per_class]
        self.num_infant = len(self.infant_idxs)
        print('loaded {} infants'.format(self.num_infant))

        # Load adult meta data
        decoded1 = sio.loadmat(mpii_anns_path, struct_as_record=False)["RELEASE"]
        self.dataset_obj = generate_dataset_obj(decoded1)
        valid_idx = set()  # only use the images with keypoints annotations
        for i in range(len(self.dataset_obj['annolist'])):
            annorect = self.dataset_obj['annolist'][i]['annorect']
            if len(annorect) > 0 and 'annopoints' in annorect[0] and len(annorect[0]['annopoints']) > 0:
                valid_idx.add(i)
        self.mpii_idxs = list(valid_idx)[:num_per_class]
        self.num_adult = len(self.mpii_idxs)
        print('loaded {} adults'.format(self.num_adult))
    
    def __getitem_infant__(self, idx):
        anns = self.infant_anns[self.infant_idxs[idx]]
        img_id = anns['id']
        img_fn = os.path.join(infant_img_path, self.infant_imgs[img_id]['file_name'])
        img = np.asarray(Image.open(img_fn).convert("RGB"))
        # get bounding box coordinates
        bbox = anns['bbox']
        xmin = np.min(int(bbox[0]))
        xmax = np.max(int(bbox[0])+int(bbox[2]))
        ymin = np.min(int(bbox[1]))
        ymax = np.max(int(bbox[1])+int(bbox[3]))
        center = [(xmin+xmax)/2, (ymin+ymax)/2]
        scale = max(xmax-xmin, ymax-ymin) * 1.2
        # print(center, scale, img.shape)
        img = crop(img, center, scale, [224,224], edge_padding=True).astype(np.uint8)
        img = Image.fromarray(img)
        return img

    def __getitem_adult__(self, idx):
        idx = self.mpii_idxs[idx]
        anno = self.dataset_obj["annolist"][idx]
        if 'annorect' in anno:
            num_persons = len(anno['annorect'])
            for i in range(num_persons):
                try:
                    points = anno['annorect'][i]['annopoints']['point']
                except:
                   print(anno['annorect'][i])
                   continue
                xmin = min([p['x'] for p in points])
                xmax = max([p['x'] for p in points])
                ymin = min([p['y'] for p in points])
                ymax = max([p['y'] for p in points])
                break # only reads one person from each image
        img_path = os.path.join(mpii_img_path, anno['image']['name'])
        img = np.asarray(Image.open(img_path).convert("RGB"))
        center = [(xmin+xmax)/2, (ymin+ymax)/2]
        scale = max(xmax-xmin, ymax-ymin) * 1.2
        # print(center, scale, img.shape)
        img = crop(img, center, scale, [224,224], edge_padding=True).astype(np.uint8)
        img = Image.fromarray(img)
        return img
    
    def __len__(self):
        return self.num_infant + self.num_adult

    def __getitem__(self, idx):
        if idx < self.num_infant:
            img = self.__getitem_infant__(idx)
            target = 1
        else:
            img = self.__getitem_adult__(idx - self.num_infant)
            target = 0
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target  # 0 = adult, 1 = infant


class UnLabeledDataset(object):
    """ Loads unlabeled crops from SEEDLingS.
    """
    def __init__(self, transform, seed=0):
        self.root = r'/scratch/groups/syyeung/seedlings'
        k_per_recording = 30  # 30 -> 12,400 images
        np.random.seed(seed)
        sampled_frames, kps = [], []
        for month in range(6, 15):
            recordings = glob.glob(os.path.join(self.root, 'month{}/*_fps1'.format(month)))
            for recording in sorted(recordings):
                images = glob.glob(os.path.join(recording, 'images/*.jpg'))
                images = list(sorted(images))
                if len(images) == 0:
                    print('Month {}, recording {} is empty'.format(month, recording.split('/')[-1][:2]))
                    continue
                sampled_frame = np.random.choice(images, size=k_per_recording, replace=False)
                sampled_frames.append(sampled_frame)
                sampled_frame_name = os.path.splitext(os.path.split(sampled_frame)[1])[0] # frameXXXXXX
                kps.append(os.path.join(
                    os.path.split(sampled_frame.replace('images', 'keypoints'))[0],
                    sampled_frame_name+'_keypoints.json'
                ))
        self.images_path = sampled_frames # full paths of the images
        self.kp_path = kps

        person_to_img_map = {}
        num_persons = 0
        for img_idx, img_name in enumerate(self.images_path):
            img_name = img_name.split('/')[-1]
            kp_name = self.kp_path[img_idx]
            kp = read_keypoints(kp_name)
            for p_i in range(len(kp)):
                # maps dataset entry id to the img name and the person idx in this img.
                person_to_img_map[num_persons+p_i] = (img_name, kp_name, p_i)
            num_persons += len(kp)
        self.person_to_img_map = person_to_img_map
        self.num_persons = num_persons

        print('loaded {} unlabeled persons'.format(num_persons))

    def __getitem__(self, idx):
        img_name, kp_name, p_i = self.person_to_img_map[idx]
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        kp = read_keypoints(kp_name)[p_i]  # (118, 3)
        crop_out = cropout_openpose_one_third(img, kp)
        cropped_image = crop_out['cropped_image']  # cropped image
        img = Image.fromarray(cropped_image)
        target = -1
        return img, target

    def __len__(self):
        return self.num_persons


def generate_dataset_obj(obj):
    must_be_list_fields = ["annolist", "annorect", "point", "img_train", "single_person", "act", "video_list"]
    if type(obj) == np.ndarray:
        dim = obj.shape[0]
        if dim == 1:
            ret = generate_dataset_obj(obj[0])
        else:
            ret = []
            for i in range(dim):
                ret.append(generate_dataset_obj(obj[i]))

    elif type(obj) == sio.matlab.mio5_params.mat_struct:
        ret = {}
        for field_name in obj._fieldnames:
            field = generate_dataset_obj(obj.__dict__[field_name])
            if field_name in must_be_list_fields and type(field) != list:
                field = [field]
            ret[field_name] = field

    else:
        ret = obj

    return ret


def print_dataset_obj(obj, depth = 0, maxIterInArray = 20):
    prefix = "  "*depth
    if type(obj) == dict:
        for key in obj.keys():
            print("{}{}".format(prefix, key))
            print_dataset_obj(obj[key], depth + 1)
    elif type(obj) == list:
        for i, value in enumerate(obj):
            if i >= maxIterInArray:
                break
            print("{}{}".format(prefix, i))
            print_dataset_obj(value, depth + 1)
    else:
        print("{}{}".format(prefix, obj))


class ValidationDataset(object):
    def __init__(self, transform):
        self.root = '/home/groups/syyeung/zzweng/seedlings_body_type_anns'
        self.transform = transform
        recordings = [
            '01_07_images', '02_07_images', '03_07_images', 
            '13_07_images', '14_07_images', '22_07_images', 
            '23_07_images', '43_06_images', '43_07_images']
   
        recording_anns = {}
        person_to_img_map = {}
        num_persons = 0
        for recording in recordings:
            with open(os.path.join(self.root, recording, 'labels.txt')) as f:
               lines = csv.reader(f)
               recording_anns[recording] = defaultdict(list)
               for line in lines:
                   recording_anns[recording][line[5]].append(line)

            for img_name in recording_anns[recording]:
                full_img_path = os.path.join(self.root, recording, img_name)
                anns = recording_anns[recording][img_name]
                for p_i in range(len(anns)):
                    person_to_img_map[num_persons] = (full_img_path, recording, p_i)
                    num_persons += 1
        self.person_to_img_map = person_to_img_map
        self.num_persons = num_persons
        
        self.anns = recording_anns
        print('Initialized SEEDLingS dataset with {} images'.format(self.num_persons))

    def __len__(self):
        return self.num_persons

    def __getitem__(self, idx):
        # load images and masks
        full_img_path, recording, p_i = self.person_to_img_map[idx]
        img = np.asarray(Image.open(full_img_path).convert("RGB"))
        frame_name = full_img_path.split('/')[-1]
        anns = self.anns[recording][frame_name][p_i]
         
        xmin = np.min(int(anns[1]))
        xmax = np.max(int(anns[1])+int(anns[3]))
        ymin = np.min(int(anns[2]))
        ymax = np.max(int(anns[2])+int(anns[4]))
        center = [(xmin+xmax)/2, (ymin+ymax)/2]
        scale = max(xmax-xmin, ymax-ymin) * 1.2
        # print(center, scale, img.shape)
        img = crop(img, center, scale, [224,224], edge_padding=True).astype(np.uint8)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if anns[0] == 'infant': 
            target = 1
        elif anns[0] == 'adult':
            target = 0
        else:
            print(anns)
            raise Exception()
        return img, target


if __name__ == '__main__':
    dataset = ValidationDataset(None)
    # unlabeled_dataset = UnLabeledDataset(None)
    print(len(dataset), 'persons in total.')
    import pdb; pdb.set_trace()
    print(dataset[0])
    # dataset[120][0].save("fig.png")
