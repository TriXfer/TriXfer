from typing import Optional
import timm as timm
import argparse
import os
import sys
import re
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
tarf_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if tarf_root not in sys.path:
    sys.path.insert(0, tarf_root)
from tarfadatasets.data_utils import *
from adaptation.utils import *
from adaptation.Adaptation import fine_tune_setup


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='vit_base_patch16_224', choices=['vit_base_patch16_224', 'vit_small_patch16_224.dino', 'cait_xxs24_224', 'tiny_vit_21m_224.dist_in22k_ft_in1k', 'deit3_base_patch16_224.fb_in22k_ft_in1k','swin_base_patch4_window7_224'])
    parser.add_argument('adaptation_method', type=str, default='full_finetuning',
                       choices=['linear_probing', 'full_finetuning', 'adapter', 'lora', 
                               'prefix_tuning', 'prompt_tuning', 'pyramid_adv_attack'],
                       help='Fine-tuning method to use')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--adapter_dim', type=int, default=64, 
                       help='Adapter dimension (for adapter method)')
    parser.add_argument('--adapter_type', type=str, default='ia3', 
                       choices=['ia3', 'adalora'],
                       help='Adapter type: ia3 or adalora')
    parser.add_argument('--lora_rank', type=int, default=8, 
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, 
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, 
                       help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, nargs='+', 
                       default=['qkv', 'proj', 'fc1', 'fc2'],
                       help='Target modules for LoRA/Adapter')
    parser.add_argument('--num_prefix_tokens', type=int, default=10, 
                       help='Number of prefix tokens (for prefix_tuning)')
    parser.add_argument('--num_prompts', type=int, default=10, 
                       help='Number of prompt tokens (for prompt_tuning)')
    parser.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')

    parser.add_argument('dataset',type = str)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='/home/huan1932/TARFA/log')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--save_name', type=str, default='model.pth')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--save_last', type=bool, default=True)


    parser.add_argument('--world_size', type=int, default=-1, 
                       help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=-1, 
                       help='node rank for distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                       help='distributed backend')
   
   
    return parser.parse_args()


def main_worker(args):
    """Main training function for each process"""

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    def get_model_for_saving(model):
        if isinstance(model, DDP):
            return model.module
        return model
    # Setup distributed training
    if world_size > 1:
        setup_distributed()
    else:
        rank = 0
        world_size = 1
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Print only from rank 0
    def print_once(*args_print, **kwargs):
        if rank == 0:
            print(*args_print, **kwargs)
    
    print_once(f"Rank {rank} using device {device}")
    print_once(f"Model: {args.model_name}, Dataset: {args.dataset}")
    
    # Load dataset
    train_dataset, val_dataset, test_dataset = get_dataset(args.dataset)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True, 
        seed=args.seed
    )
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size,  # Divide batch size by world_size
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // world_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // world_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    if args.model_name == 'vit_base_patch16_224':
        if not args.pretrained:
            checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'vit_b16_imagenet1k.pth')
        else:
            checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'vit_base')
    elif args.model_name == 'vit_small_patch16_224.dino':
        checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'dinov2_vits.pth')
    elif args.model_name == 'cait_xxs24_224':
        checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'cait_xxs24_224_imagenet1k.pth')
    elif args.model_name == 'tiny_vit_21m_224.dist_in22k_ft_in1k':
        checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'tinyvit_5m_224_imagenet1k.pth')
    elif args.model_name == 'deit3_base_patch16_224.fb_in22k_ft_in1k':
        checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'deit3_base_imagenet1k.pth')
    elif args.model_name == 'swin_base_patch4_window7_224':
        if not args.pretrained:
            checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'swin_base_patch4_window7_224_imagenet1k.pth')
        else:
            checkpoint_path = os.path.join(tarf_root, 'pretrained_models', 'swin')   
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    # Load model
    model = load_pretrained_model(
        args.model_name, device, 
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        checkpoint_path=checkpoint_path
    )
    
    # inspect_model_quick(model, args.model_name)

    # raise ValueError("Stop here")
    
    model = fine_tune_setup(model, adaptation_method=args.adaptation_method,
    adapter_dim=args.adapter_dim,
    adapter_type=args.adapter_type,
    lora_rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.target_modules,
    num_prefix_tokens=args.num_prefix_tokens,
    num_prompts=args.num_prompts)
    model = model.to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank],output_device=rank, find_unused_parameters=False)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True
    )
    
    # Learning rate scheduler 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # TensorBoard writer 
    writer = None
    if rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print_once(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        if world_size > 1:
            actual_model = get_model_for_saving(model)
            actual_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Resume from correct epoch
        start_epoch = checkpoint['epoch'] + 1
        filename = os.path.basename(args.resume)
        match = re.search(r'epoch_\d+_(\d+\.\d+)_', filename)
        if match:
            best_acc = float(match.group(1))
            print_once(f"Extracted best_acc {best_acc:.2f}% from filename")
        
        print_once(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}%")
    elif args.resume:
        print_once(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    patience = 0
    for epoch in range(args.epochs):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, rank,
            args.adaptation_method, args.epsilon, args.n_steps
        )
        if world_size > 1:
            train_acc_tensor = torch.tensor(train_acc).to(device)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            train_acc = (train_acc_tensor.item() / world_size)  # Average across ranks
        # Validate
        test_loss, test_acc = validate(
            model, test_loader, criterion, device, world_size
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print results (only rank 0)
        print_once(f'Epoch {epoch+1}/{args.epochs}:')
        print_once(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print_once(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print_once(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Log to TensorBoard (only rank 0)
        if writer is not None:
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Acc', train_acc, epoch)
            writer.add_scalar('Test/Loss', test_loss, epoch)
            writer.add_scalar('Test/Acc', test_acc, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        should_stop = False
        # Save checkpoint
        if rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            
            # Save best model
            if args.save_best and test_acc > best_acc:
                patience = 0
                best_acc = test_acc
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': get_model_for_saving(model).state_dict(),  # Note: model.module for DDP
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(args.save_dir, 'best_' + args.save_name +'.pth'))
                print_once(f'Saved best model with acc: {best_acc:.2f}%')
            else:
                patience += 1
            if patience > 7:
                should_stop = True
                print_once(f"Early stopping at epoch {epoch+1}, best_acc: {best_acc:.2f}%")
            if "." in args.model_name:
                model_name = args.model_name.replace(".", "_")
            else:
                model_name = args.model_name
            if (epoch + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': get_model_for_saving(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   
                }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}_{test_acc:.2f}_{model_name}_{args.dataset}_{args.adaptation_method}.pth'))
            
            # Save last model
            if args.save_last and epoch == args.epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': get_model_for_saving(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }, os.path.join(args.save_dir, 'last_' + args.save_name+f"{test_acc:.2f}_.pth"))

        if world_size > 1:
            should_stop_tensor = torch.tensor(1 if should_stop else 0, dtype=torch.int, device=device)
            dist.broadcast(should_stop_tensor, src=0)
            should_stop = should_stop_tensor.item() == 1
        if should_stop:
            break
    if writer is not None:
        writer.close()
    
    # Cleanup
    cleanup_distributed()





def main():
    args = parse_arguments()
    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    main_worker(args)

if __name__ == '__main__':
    main()