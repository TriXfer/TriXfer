import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as FT
import timm as timm
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os





def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', choices=['vit_base_patch16_224', 'vit_small_patch14_dinov2.lvd142m', 'cait_xxs24_224', 'tiny_vit_5m_224', 'deit3_base_patch16_224'])
    parser.add_argument('--dataset',type = str)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--save_name', type=str, default='model.pth')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--save_last', type=bool, default=True)
    return parser.parse_args()

args = parse_arguments()

if args.dataset == 'cifar10':
    sample_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
elif args.dataset == 'cifar100':
    sample_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
else:
    raise ValueError(f"Invalid dataset: {args.dataset}")

sample_loader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Using model: {args.model_name}")
print(f"Using Dataset: {args.data}")
def load_pretrained_model(model_name, num_classes=None, pretrained=True):
    """
    Load pretrained models using timm
    
    Args:
        model_name: timm model identifier
        num_classes: Number of output classes (None keeps original ImageNet-1k classes)
        pretrained: Whether to load pretrained weights
    """
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes if num_classes else 1000
    )
    return model

# 1. ViT-B/16 (ImageNet-1k pretrained)
vit_b16 = load_pretrained_model('vit_base_patch16_224', pretrained=True)

# 2. PyramidAT + ViT-B/16 (adversarial pretrained)

vit_b16_pyramidat = timm.create_model('vit_base_patch16_224', pretrained=False)


# 3. DeiT III (Base) - supervised baseline
deit3_base = load_pretrained_model('deit3_base_patch16_224', pretrained=True)

# 4. DINOv2 ViT-S - self-supervised baseline
dinov2_vits = load_pretrained_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)

# 5. CaiT (Class-Attention Transformer)
cait_xxs24_224 = load_pretrained_model('cait_xxs24_224', pretrained=True)
# Or other variants: cait_xs24_224, cait_s24_224, cait_s36_384

# 6. TinyViT
tinyvit_5m_224 = load_pretrained_model('tiny_vit_5m_224', pretrained=True)
# Or other variants: tiny_vit_11m_224, tiny_vit_21m_224, tiny_vit_21m_384
