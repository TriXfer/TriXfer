import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os
import torch.nn.functional as F
import timm
from adaptation.attacks import AdaEA_FGSM, AdaEA_IFGSM, AdaEA_MIFGSM, AdaEA_DIFGSM, AdaEA_TIFGSM, Trixfer_FGSM, Trixfer_IFGSM, Trixfer_MIFGSM, Trixfer_DIFGSM, Trixfer_TIFGSM

class ModelWithCoarseHead(nn.Module):
    """Model with extra linear layer for coarse classification"""
    def __init__(self, base_model, num_fine_classes, num_coarse_classes, dataset_type='oxford-iiit-pet'):
        super().__init__()
        self.base_model = base_model
        self.dataset_type = dataset_type
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes
        
        # Extra linear layer: fine classes -> coarse classes
        self.extra_linear = nn.Linear(num_fine_classes, num_coarse_classes)
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Only train coarse head
        for param in self.extra_linear.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Get fine-grained logits from base model
        fine_logits = self.base_model(x)
        # Map to coarse classes
        coarse_logits = self.extra_linear(fine_logits)
        return coarse_logits
    
    def get_coarse_label(self, fine_label):
        """Convert fine label to coarse label"""
        if self.dataset_type == 'oxford-iiit-pet':
            return get_coarse_label_oxford_pet(fine_label)
        elif self.dataset_type == 'combined':
            return get_coarse_label_combined(fine_label)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
 
    
def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    print(f"Rank {rank}: Initializing process group (world_size={world_size}, port={master_port})...")
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )
    
    print(f"Rank {rank}: ✓ Process group initialized!")
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    return rank, world_size


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def convert_labels_to_coarse(model, labels, device):
    """
    Convert fine-grained labels to coarse labels if model has coarse head.
    Handles both regular models and DDP-wrapped models.
    """
    # Check if model has coarse head (handle DDP wrapping)
    actual_model = model.module if hasattr(model, 'module') else model
    
    if isinstance(actual_model, ModelWithCoarseHead):
        # Convert fine labels to coarse labels
        coarse_labels = torch.zeros_like(labels).to(device)
        for i, fine_label in enumerate(labels):
            coarse_labels[i] = actual_model.get_coarse_label(fine_label.item())
        return coarse_labels
    else:
        # No coarse head, return original labels
        return labels

def prepare_model(model, method='full_finetuning'):
    """
    Prepare timm model for different adaptation methods
    
    Args:
        model: timm model
        method: 'linear_probing', 'full_finetuning', 'adapter', 'lora', 'prefix_tuning', 'prompt_tuning'
        **kwargs: method-specific arguments
    """
    if method == 'linear_probing':
        # Freeze all parameters except classifier
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        return model


    elif method == 'full_finetuning':
        # All parameters trainable
        for param in model.parameters():
            param.requires_grad = True
        return model

def filter_head_layers(state_dict, model):
    """
    Remove all final FC/classifier layer weights from state_dict
    Works with various head types: MlpClassifierHead, ClassifierHead, 
    NormMlpClassifierHead, and direct Linear classifiers
    """
    keys_to_remove = []
    
    for key in list(state_dict.keys()):
        # Check for various head/classifier patterns
        if any(pattern in key for pattern in [
            'head.fc',           # NormMlpClassifierHead: head.fc
            'head.fc1',          # MlpClassifierHead: head.fc1
            'head.fc2',          # MlpClassifierHead: head.fc2
            'head.classifier',   # ClassifierHead: head.classifier
            'head.in_conv',      # ClassifierHead: head.in_conv
            'classifier.',       # Direct classifier
        ]):
            keys_to_remove.append(key)
    
    # Remove the keys
    for key in keys_to_remove:
        del state_dict[key]
        print(f"Removed head layer: {key}")
    
    return state_dict

# Add this function to utils.py (after filter_head_layers, around line 147)

def remove_ddp_prefix(state_dict):
    """
    Remove 'module.' prefix from state dict keys (for DDP checkpoints).
    
    Args:
        state_dict: State dictionary that may have 'module.' prefix
    
    Returns:
        State dictionary with 'module.' prefix removed
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if present
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' (7 characters)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def load_pretrained_surrogate_model(model_name, device, num_classes=None, pretrained=False, checkpoint_path=None,feature_only=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    saved_num_classes = checkpoint.get('num_classes', 1000)
    
    if num_classes is None:
        num_classes = saved_num_classes
    
    # Create model
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        features_only=feature_only,
        num_classes=num_classes
    )
    if not pretrained:
        state_dict = checkpoint['model_state_dict']
        
        # Always filter head layers to avoid size mismatches
        # This ensures we only load backbone weights
        state_dict = filter_head_layers(state_dict, model)
        
        # Load state dict (strict=False allows missing keys)
        model.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded pretrained weights (excluding head layers)")
        print(f"Model initialized with {num_classes} classes")
    else:
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights")
        print(f"Model initialized with {num_classes} classes")
    return model
def load_pretrained_model(model_name, device, num_classes=None, pretrained=False,checkpoint_path=None):

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    saved_num_classes = checkpoint.get('num_classes', 1000)
    yes = False
    if num_classes is None:
        num_classes = saved_num_classes
    if model_name == 'AT_vit_base_patch16_224':
        model_name = 'vit_base_patch16_224'
        yes = True
    elif model_name == 'AT_swin_base_patch4_window7_224':
        model_name = 'swin_base_patch4_window7_224'
        yes = True
    # Create model
    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,  # We're loading from checkpoint
        num_classes=num_classes
    )
    if not pretrained:
        if yes:
             state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint['model_state_dict']
        
        # If num_classes differs, we need to handle the head layer
        if num_classes != saved_num_classes:
            # Remove head-related keys from state dict if sizes differ
            # For TinyViT: head.fc.weight and head.fc.bias
            # For other models: might be head.weight, head.bias, classifier.weight, etc.
            keys_to_remove = []
            for key in state_dict.keys():
                if 'head.fc' in key or 'head.weight' in key or 'head.bias' in key:
                    keys_to_remove.append(key)
                elif 'head_dist' in key: 
                    keys_to_remove.append(key)
                elif 'classifier' in key.lower() and ('weight' in key or 'bias' in key):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del state_dict[key]
        
        # Load state dict (strict=False allows missing keys)
        model.load_state_dict(state_dict, strict=False)
        
        # Now replace the classifier head if needed
        if num_classes != saved_num_classes:
            if hasattr(model, 'head'):
                # For TinyViT: head is NormMlpClassifierHead with head.fc
                if hasattr(model.head, 'fc'):
                    # Replace the fc layer
                    in_features = model.head.fc.in_features
                    model.head.fc = nn.Linear(in_features, num_classes)
                    print(f"Replaced head.fc: {in_features} -> {num_classes} classes")
                elif hasattr(model.head,'classifier'):
                    if isinstance(model.head.classifier,nn.Sequential):
                        for i, layer in enumerate(model.head.classifier):
                            if isinstance(layer, nn.Linear) and i+1 == len(model.head.classifier):
                                in_features = layer.in_features
                                model.head.classifier[i] = nn.Linear(in_features, num_classes)
                                print(f"Replaced classifier[{i}]: {in_features} -> {num_classes} classes")
                                break
                else:
                    # Some models might have head as a direct Linear layer
                    if isinstance(model.head, nn.Linear):
                        in_features = model.head.in_features
                        model.head = nn.Linear(in_features, num_classes)
                        print(f"Replaced head: {in_features} -> {num_classes} classes")
            elif hasattr(model, 'classifier'):
                # Handle classifier (for other model types)
                if isinstance(model.classifier, nn.Sequential):
                    # Find the Linear layer in the classifier
                    for i, layer in enumerate(model.classifier):
                        if isinstance(layer, nn.Linear):
                            in_features = layer.in_features
                            model.classifier[i] = nn.Linear(in_features, num_classes)
                            print(f"Replaced classifier[{i}]: {in_features} -> {num_classes} classes")
                            break
                elif isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
                    print(f"Replaced classifier: {in_features} -> {num_classes} classes")
    else:
        state_dict = checkpoint['state_dict']
        
        # If num_classes differs, we need to handle the head layer
        if num_classes != saved_num_classes:
            # Remove head-related keys from state dict if sizes differ
            # For TinyViT: head.fc.weight and head.fc.bias
            # For other models: might be head.weight, head.bias, classifier.weight, etc.
            keys_to_remove = []
            for key in state_dict.keys():
                if 'head.fc' in key or 'head.weight' in key or 'head.bias' in key:
                    keys_to_remove.append(key)
                elif 'head_dist' in key: 
                    keys_to_remove.append(key)
                elif 'classifier' in key.lower() and ('weight' in key or 'bias' in key):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del state_dict[key]
        
        # Load state dict (strict=False allows missing keys)
        model.load_state_dict(state_dict, strict=False)
        
        # Now replace the classifier head if needed
        if num_classes != saved_num_classes:
            if hasattr(model, 'head'):
                # For TinyViT: head is NormMlpClassifierHead with head.fc
                if hasattr(model.head, 'fc'):
                    # Replace the fc layer
                    in_features = model.head.fc.in_features
                    model.head.fc = nn.Linear(in_features, num_classes)
                    print(f"Replaced head.fc: {in_features} -> {num_classes} classes")
                else:
                    # Some models might have head as a direct Linear layer
                    if isinstance(model.head, nn.Linear):
                        in_features = model.head.in_features
                        model.head = nn.Linear(in_features, num_classes)
                        print(f"Replaced head: {in_features} -> {num_classes} classes")
            elif hasattr(model, 'classifier'):
                # Handle classifier (for other model types)
                if isinstance(model.classifier, nn.Sequential):
                    # Find the Linear layer in the classifier
                    for i, layer in enumerate(model.classifier):
                        if isinstance(layer, nn.Linear):
                            in_features = layer.in_features
                            model.classifier[i] = nn.Linear(in_features, num_classes)
                            print(f"Replaced classifier[{i}]: {in_features} -> {num_classes} classes")
                            break
                elif isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
                    print(f"Replaced classifier: {in_features} -> {num_classes} classes")
    return model

def get_coarse_label_oxford_pet(fine_label):
   
    cat_indices = {0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33}
    
    if fine_label in cat_indices:
        return 0  # Cat
    else:
        return 1  # Dog

def get_coarse_label_combined(fine_label):
    """
    Combined dataset: 306 classes -> 3 coarse classes
    Stanford Cars (0-195) -> 0 (Cars)
    ImageWoof (196-205) -> 1 (Dogs)  
    FGVC-Aircraft (206-305) -> 2 (Aircraft)
    """
    if fine_label < 196:
        return 0  # Cars
    elif fine_label < 206:
        return 1  # Dogs
    else:
        return 2  # Aircraft

# Coarse class names
OXFORD_PET_COARSE_CLASSES = ['Cat', 'Dog']
COMBINED_COARSE_CLASSES = ['Car', 'Dog', 'Aircraft']


def prepare_coarse_tune(model, num_fine_classes, num_coarse_classes, dataset_type='oxford-iiit-pet'):
    """Wrap model with coarse classification head"""
    return ModelWithCoarseHead(model, num_fine_classes, num_coarse_classes, dataset_type)


def get_attack(args, device, trixfer = False,ens_models = None,fine_model=None,coarse_model=None):
    # AdaEA
    if not trixfer:
        if args.attack_method == 'AdaEA_FGSM':
            attack_method = AdaEA_FGSM.AdaEA_FGSM(
                ens_models, eps=args.eps, max_value=args.max_value, min_value=args.min_value, threshold=args.threshold,
                beta=args.beta, device=device)
        elif args.attack_method == 'AdaEA_IFGSM':
            attack_method = AdaEA_IFGSM.AdaEA_IFGSM(
                ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
        elif args.attack_method == 'AdaEA_MIFGSM':
            attack_method = AdaEA_MIFGSM.AdaEA_MIFGSM(
                ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
                momentum=args.momentum)
        elif args.attack_method == 'AdaEA_DIFGSM':
            attack_method = AdaEA_DIFGSM.AdaEA_DIFGSM(
                ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, threshold=args.threshold, device=device, beta=args.beta,
                momentum=args.momentum, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob)
        elif args.attack_method == 'AdaEA_TIFGSM':
            attack_method = AdaEA_TIFGSM.AdaEA_TIFGSM(
                ens_models, eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device)
        else:
            raise NotImplemented
    else:
        if args.attack_method == 'Trixfer_FGSM':
            attack_method = Trixfer_FGSM.Trixfer_FGSM(
                eps=args.eps, max_value=args.max_value,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device,fine_model=fine_model,coarse_model=coarse_model,lambda_fine=args.lambda_fine,lambda_coarse=args.lambda_coarse,lambda_sim=args.lambda_sim,model_name=args.fine_model_name,layer_level=args.layer_level)
        elif args.attack_method == 'Trixfer_IFGSM':
            attack_method = Trixfer_IFGSM.Trixfer_IFGSM(
                eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device,fine_model=fine_model,coarse_model=coarse_model,lambda_fine=args.lambda_fine,lambda_coarse=args.lambda_coarse,lambda_sim=args.lambda_sim,model_name=args.fine_model_name,layer_level=args.layer_level)
        elif args.attack_method == 'Trixfer_MIFGSM':
            attack_method = Trixfer_MIFGSM.Trixfer_MIFGSM(
                eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device,fine_model=fine_model,coarse_model=coarse_model,lambda_fine=args.lambda_fine,lambda_coarse=args.lambda_coarse,lambda_sim=args.lambda_sim,model_name=args.fine_model_name,layer_level=args.layer_level)
        elif args.attack_method == 'Trixfer_DIFGSM':
            attack_method = Trixfer_DIFGSM.Trixfer_DIFGSM(
                eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,resize_rate=args.resize_rate,diversity_prob=args.diversity_prob,momentum=args.momentum,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device,fine_model=fine_model,coarse_model=coarse_model,lambda_fine=args.lambda_fine,lambda_coarse=args.lambda_coarse,lambda_sim=args.lambda_sim,model_name=args.fine_model_name,layer_level=args.layer_level)
        elif args.attack_method == 'Trixfer_TIFGSM':
            attack_method = Trixfer_TIFGSM.Trixfer_TIFGSM(
                eps=args.eps, alpha=args.alpha, iters=args.iters, max_value=args.max_value,
                min_value=args.min_value, beta=args.beta, threshold=args.threshold, device=device,fine_model=fine_model,coarse_model=coarse_model,lambda_fine=args.lambda_fine,lambda_coarse=args.lambda_coarse,lambda_sim=args.lambda_sim,model_name=args.fine_model_name,layer_level=args.layer_level)
        else:
            raise NotImplementedError(f"Attack method {args.attack_method} not implemented")

    return attack_method



def train_epoch(model, train_loader, criterion, optimizer, device, rank, method = None, epsilon=None, n_steps=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = convert_labels_to_coarse(model, target, device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if rank == 0 and batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device, world_size):
    """Validate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = convert_labels_to_coarse(model, target, device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Aggregate results across all processes
    if world_size > 1:
        test_loss = torch.tensor(test_loss).to(device)
        correct = torch.tensor(correct).to(device)
        total = torch.tensor(total).to(device)
        
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        
        test_loss = test_loss.item() / len(test_loader)
        test_acc = 100. * correct.item() / total.item()
    else:
        # Single GPU - no reduction needed
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
    
    return test_loss, test_acc


