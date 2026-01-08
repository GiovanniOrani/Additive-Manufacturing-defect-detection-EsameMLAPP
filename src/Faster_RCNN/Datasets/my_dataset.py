import os
import torch
from PIL import Image
import json
from torchvision import transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split= 'train', augmentation=False):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.annotations = json.load(open(os.path.join(root, f"annotations/{self.split}_annotations.json")))

        ## Data Augmentation
        bright_t = transforms.ColorJitter(brightness=[0.7,1.5])
        contrast_t = transforms.ColorJitter(contrast = [0.7,1.5])

        aug_transformation = transforms.Compose([bright_t, contrast_t])

        if augmentation:
            self.aug_pipeline = transforms.Compose([
                transforms.RandomApply([aug_transformation], p = 0.5),
                transforms.ToTensor()
            ])
        else:
            self.aug_pipeline = transforms.Compose([
                transforms.ToTensor()
            ])

        self.boxes = {}
        self.labels = {}
        self.imageList = []
        self.images_paths = {img["id"]: os.path.join(self.root,f"images/{self.split}" ,img["file_name"]) for img in self.annotations["images"]}

        for image in self.annotations['images']:
            self.imageList.append(image['id'])
            self.boxes[image['id']] = []
            self.labels[image['id']] = []
        
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']

            if image_id not in self.boxes:
                print(f"found an unkown image_id:{image_id}")
                
            x, y, w, h = ann["bbox"]
            self.boxes[image_id].append([x, y, x + w, y + h])
            self.labels[image_id].append(ann['category_id'])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_id = self.imageList[idx]
        img_path = self.images_paths[image_id]
        img = Image.open(img_path).convert("RGB")

        if len(self.boxes[image_id])==0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0, ), dtype=torch.int64)
        else:
            boxes = torch.tensor(self.boxes[image_id], dtype=torch.float32)
            labels = torch.tensor(self.labels[image_id], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        img = self.aug_pipeline(img)

        return img, target
