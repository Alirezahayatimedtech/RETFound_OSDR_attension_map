%%writefile "/content/drive/My Drive/Colab Notebooks/RETFound_MAE/util/datasets.py"
import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Custom ImageFolder class to return (image, target, path)
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # Get the original (image, target) from ImageFolder
        image, target = super().__getitem__(index)
        # Get the file path for this index
        path = self.imgs[index][0]
        return image, target, path

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train == 'train' else 'val' if is_train == 'val' else 'test')
    # Use ImageFolderWithPaths for visualization mode, otherwise use regular ImageFolder
    if args.visualize:
        dataset = ImageFolderWithPaths(root, transform=transform)
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
    
    print(f"Dataset for {is_train}: {root}")
    print("Classes:", dataset.classes)
    print("Class to index mapping:", dataset.class_to_idx)
    print("Number of samples:", len(dataset))

    # Debug the first few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        if args.visualize:
            image, target, path = sample
            print(f"Sample {i} - Image shape: {image.shape}, Target: {target}, Path: {path}")
        else:
            image, target = sample
            print(f"Sample {i} - Image shape: {image.shape}, Target: {target}")

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
