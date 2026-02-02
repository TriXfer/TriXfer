import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from time import time

# Add project root to path
tarf_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if tarf_root not in sys.path:
    sys.path.insert(0, tarf_root)
from tarfadatasets.data_utils import get_dataset, inspect_model_quick
from adaptation.utils import *
# Add this import after line 23
from adaptation.label_mapping import (
    map_source_to_victim_label,
    map_combined_to_victim_label,
    are_semantically_similar,
    get_source_dataset_classes,
    get_victim_dataset_classes,
)
from adaptation.Adaptation import fine_tune_setup
def get_coarse_labels(fine_labels, dataset_type,device):
    if dataset_type == 'oxford-iiit-pet':
        # Convert batch of labels
        coarse_labels = torch.zeros_like(fine_labels).to(device)
        for i, fine_label in enumerate(fine_labels):
            coarse_labels[i] = get_coarse_label_oxford_pet(fine_label.item())
        return coarse_labels
    elif dataset_type == 'combined':
        # Convert batch of labels using vectorized operations
        coarse_labels = torch.zeros_like(fine_labels).to(device)
        # Stanford Cars (0-195) -> 0 (Cars)
        # ImageWoof (196-205) -> 1 (Dogs)
        # FGVC-Aircraft (206-305) -> 2 (Aircraft)
        coarse_labels[fine_labels < 196] = 0
        coarse_labels[(fine_labels >= 196) & (fine_labels < 206)] = 1
        coarse_labels[fine_labels >= 206] = 2
        return coarse_labels
    else:
        # No coarse labels, return original
        return fine_labels.to(device)

def parse_arguments():
    parser = argparse.ArgumentParser(description='MI-FGSM Attack with Fine and Coarse Grained Losses')
    
    # Model arguments
    
    parser.add_argument('--fine_model_name', type=str, required=True,
                       help='Name of fine-grained model (timm model name)')
    parser.add_argument('--coarse_model_name', type=str, default=None,
                       help='Name of coarse-grained model (timm model name)')
    parser.add_argument('--fine_checkpoint', type=str,
                       help='Path to fine-grained model checkpoint')
    parser.add_argument('--coarse_checkpoint', type=str, default=None,
                       help='Path to coarse-grained model checkpoint')
    parser.add_argument('--num_fine_classes', type=int, required=True,
                       help='Number of fine-grained classes')
    parser.add_argument('--num_coarse_classes', type=int, default=None,
                       help='Number of coarse-grained classes')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['stl10', 'oxford-iiit-pet', 'cinic10', 'combined'],
                       help='Dataset name')
    parser.add_argument('--adaptation_method', type=str, default='full_finetuning',
                       choices=['full_finetuning', 'linear_probing', 'adapter', 'lora', 
                               'prefix_tuning', 'prompt_tuning', 'pyramid_adv_attack'],
                       help='Adaptation method')
    parser.add_argument('--p_save', type=bool, default=False,
                       help='Save evaluation results')
    parser.add_argument('--victim_dataset', type=str, default=None,
                   choices=['cifar10', 'cifar100', 'imagenet', 'imagenet1k'],
                   help='Dataset that victim model was trained on (required for evaluation)')
    parser.add_argument('--attack_method', type=str, required=True,
                       choices=['AdaEA_FGSM', 'AdaEA_IFGSM', 'AdaEA_MIFGSM', 'AdaEA_DIFGSM', 'AdaEA_TIFGSM', 'Trixfer_FGSM', 'Trixfer_IFGSM', 'Trixfer_MIFGSM', 'Trixfer_DIFGSM', 'Trixfer_TIFGSM'],
                       help='Attack method')
    parser.add_argument('--victim_model_name', type=str, default=None,
                       help='Name of victim model for evaluation (timm model name)')
    parser.add_argument('--victim_checkpoint', type=str, default=None,
                       help='Path to victim model checkpoint for evaluation')
    parser.add_argument('--down_stream_checkpoint', type=str, default=None,
                       help='Path to down-stream model checkpoint for evaluation')
    parser.add_argument('--ae_path', type=str, default=None,
                       help='Path to pre-generated adversarial examples .pt file')
    parser.add_argument('--trixfer', type=bool, default=True,
                       help='Use Trixfer attack method')
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--min_value', type=float, default=0.0,
                       help='Minimum value (default: 0.0)')
    parser.add_argument('--max_value', type=float, default=1.0,
                       help='Maximum value (default: 1.0)')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Threshold (default: 0.0)')
    parser.add_argument('--beta', type=float, default=10,
                       help='Beta (default: 10)')
    parser.add_argument('--resize_rate', type=float, default=0.9,
                       help='Resize rate (default: 0.9)')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                       help='Diversity probability (default: 0.5)')
    parser.add_argument('--iters', type=int, default=20,
                       help='Number of iterations (default: 20)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum factor (default: 0.9)')
    parser.add_argument('--lambda_fine', type=float, default=1.0,
                       help='Weight for fine-grained loss (default: 1.0)')
    parser.add_argument('--lambda_coarse', type=float, default=1.0,
                       help='Weight for coarse-grained loss (default: 1.0)')
    parser.add_argument('--lambda_sim', type=float, default=1.0,
                       help='Weight for similarity loss (default: 1.0)')
    # Attack parameters
    parser.add_argument('--eps', type=float, default=8/255,
                       help='Maximum perturbation budget (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=2/255,
                       help='Step size (default: 2/255)')
    parser.add_argument('--layer_level', type=float, default=0.33,
                       help='Layer level (default: 0.33)')
    # Data arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')

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
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='attack/AE',
                       help='Directory to save adversarial examples (default: ./adversarial_examples)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--save', type=bool, default=False,
                       help='Save evaluation results (default: False)')
    parser.add_argument('--subset_size', type=float, default=1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset, val_dataset, test_dataset = get_dataset(args.dataset)
    generator = torch.Generator().manual_seed(args.seed)
    
    # Get total dataset size
    total_size = len(test_dataset)
    
    # Generate random indices
    indices = torch.randperm(total_size, generator=generator)[:int(total_size*args.subset_size)].tolist()
    
    # Create subset
    test_dataset = torch.utils.data.Subset(test_dataset, indices)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    if args.evaluate:
        # Load victim model
        if args.victim_model_name is None:
            raise ValueError("--victim_model_name must be provided when --evaluate is True")
        if args.victim_checkpoint is None:
            raise ValueError("--victim_checkpoint must be provided when --evaluate is True")
        if args.victim_dataset is None:
            raise ValueError("--victim_dataset must be provided when --evaluate is True")
        
        print(f"Loading victim model: {args.victim_model_name}")
        print(f"Victim model trained on: {args.victim_dataset}")
        victim_model = load_pretrained_model(
            args.victim_model_name,
            device,
            num_classes=args.num_fine_classes,
            pretrained=False,
            checkpoint_path=args.victim_checkpoint
        )
        if args.down_stream_checkpoint is not None:
            victim_model = fine_tune_setup(victim_model, adaptation_method=args.adaptation_method,
            adapter_dim=args.adapter_dim,
            adapter_type=args.adapter_type,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            num_prefix_tokens=args.num_prefix_tokens,
            num_prompts=args.num_prompts)
            downstream_checkpoint = torch.load(args.down_stream_checkpoint, map_location=device)
            if 'model_state_dict' in downstream_checkpoint:
                downstream_state_dict = remove_ddp_prefix(downstream_checkpoint['model_state_dict'])
            else:
                downstream_state_dict = remove_ddp_prefix(downstream_checkpoint)

            victim_model.load_state_dict(downstream_state_dict, strict=False)
        victim_model = victim_model.to(device)
        victim_model.eval()
        # Evaluate clean accuracy on source dataset
        print("\n" + "="*60)
        print("Evaluating Clean Accuracy on Source Dataset")
        print("="*60)
        clean_correct = 0
        clean_total = 0
        
        # Get class names for mapping
        source_classes = get_source_dataset_classes(args.dataset)
        victim_classes = get_victim_dataset_classes(args.victim_dataset)
        # For ImageNet, load class names
        imagenet_dict = None
        if args.victim_dataset in ['imagenet', 'imagenet1k']:
            try:
                import json
                imagenet_mapping_path = '/home/newdrive/huan1932/data/ImageNet2012/info/imagenet_class_index.json'
                if os.path.exists(imagenet_mapping_path):
                    with open(imagenet_mapping_path, 'r') as f:
                        imagenet_dict = json.load(f)
                    victim_classes = [imagenet_dict[str(i)][1] for i in range(1000)]
                    print(f"Loaded {len(victim_classes)} ImageNet class names from mapping file")
            except Exception as e:
                print(f"Warning: Could not load ImageNet class names: {e}")
                victim_classes = None
        
        with torch.no_grad():
            for images, source_labels in tqdm(test_dataloader, desc="Clean Accuracy"):
                images = images.to(device)
                source_labels = source_labels.to(device)
                
                # Get predictions
                preds = victim_model(images).argmax(dim=1)
                
                # Map source labels to victim labels and check accuracy
                for j in range(len(source_labels)):
                    source_label = source_labels[j].item()
                    pred_label = preds[j].item()
                    
                    # Map source label to victim label
                    if args.dataset == 'combined':
                        victim_label, label_exists, source_class_name = map_combined_to_victim_label(
                            source_label, args.victim_dataset, victim_classes
                        )
                    else:
                        victim_label, label_exists, source_class_name = map_source_to_victim_label(
                            source_label, args.dataset, args.victim_dataset,
                            source_classes, victim_classes
                        )
                    
                    clean_total += 1
                    
                    if victim_label is not None:
                        # Label exists in victim model
                        if label_exists:
                            # Exact match - check if prediction matches
                            if pred_label == victim_label:
                                clean_correct += 1
                        else:
                            # Semantic similarity mapping - check if semantically similar
                            if victim_classes is not None and pred_label < len(victim_classes):
                                pred_class_name = victim_classes[pred_label]
                            else:
                                pred_class_name = f"class_{pred_label}"
                            
                            if victim_classes is not None and victim_label < len(victim_classes):
                                victim_class_name = victim_classes[victim_label]
                            else:
                                victim_class_name = f"class_{victim_label}"
                            
                            if are_semantically_similar(victim_class_name, pred_class_name, args.victim_dataset, args.dataset, source_label if args.dataset == 'combined' else None):
                                clean_correct += 1
                            elif pred_label == victim_label:
                                clean_correct += 1
                                
                            
                    else:
                        
                        # Semantic similarity mapping - check if prediction is semantically similar
                        if victim_classes is not None and pred_label < len(victim_classes):
                            pred_class_name = victim_classes[pred_label]
                        else:
                            pred_class_name = f"class_{pred_label}"
                        
                        if are_semantically_similar(source_class_name, pred_class_name, args.victim_dataset, args.dataset, None):
                            clean_correct += 1
        
        clean_accuracy = 100. * clean_correct / clean_total if clean_total > 0 else 0.0
        print(f"Clean Accuracy: {clean_accuracy:.4f}% ({clean_correct}/{clean_total})")
        print("="*60 + "\n")
        if args.ae_path is None:
            # Try to construct path from output_dir
            ae_filename = f'{args.attack_method}_{args.dataset}_{args.fine_model_name}_adversarial_examples.pt'
            args.ae_path = os.path.join(args.output_dir, ae_filename)
        
        if not os.path.exists(args.ae_path):
            raise FileNotFoundError(f"Adversarial examples file not found: {args.ae_path}")
        
        print(f"Loading adversarial examples from: {args.ae_path}")
        ae_data = torch.load(args.ae_path, map_location=device)
        if 'adversarial_examples' in ae_data:
            adv_images = ae_data['adversarial_examples'].to(device)
            fine_labels = ae_data['labels'].to(device)
        else:
            successful_adv_images = ae_data['successful_adv_images']
            successful_source_labels = ae_data['successful_source_labels']
            
            # Convert lists to tensors if needed
            if isinstance(successful_adv_images, list):
                adv_images = torch.stack(successful_adv_images).to(device)
            else:
                adv_images = successful_adv_images.to(device)
            
            if isinstance(successful_source_labels, list):
                fine_labels = torch.tensor(successful_source_labels).to(device)
            else:
                fine_labels = successful_source_labels.to(device)
        coarse_labels = ae_data.get('coarse_labels', None)
        if coarse_labels is not None:
            coarse_labels = coarse_labels.to(device)
        
        print(f"Loaded {len(adv_images)} adversarial examples")
        print(f"Source dataset: {args.dataset}")
        
        # Evaluate victim model on adversarial examples with proper label mapping
        print("Evaluating victim model on adversarial examples with label mapping...")
        total_samples = len(adv_images)
        successful_attacks = 0
        correct_predictions = 0
        unmapped_samples = 0
        semantically_similar_failures = 0
        successful_adv_images = []
        successful_source_labels = []
        # Process in batches
        batch_size = args.batch_size
        with torch.no_grad():
            for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
                batch_end = min(i + batch_size, total_samples)
                batch_adv = adv_images[i:batch_end]
                batch_source_labels = fine_labels[i:batch_end]
                
                # Get predictions
                preds = victim_model(batch_adv).argmax(dim=1)
                
                # Map source labels to victim labels and evaluate
                for j in range(len(batch_source_labels)):
                    source_label = batch_source_labels[j].item()
                    pred_label = preds[j].item()
                    
                    # Map source label to victim label
                    if args.dataset == 'combined':
                        victim_label, label_exists, source_class_name = map_combined_to_victim_label(
                            source_label, args.victim_dataset, victim_classes
                        )
                    else:
                        victim_label, label_exists, source_class_name = map_source_to_victim_label(
                            source_label, args.dataset, args.victim_dataset,
                            source_classes, victim_classes
                        )
                    
                    is_successful_attack = False
                    
                    if victim_label is None:
                        # Label doesn't exist in victim model - use semantic similarity
                        unmapped_samples += 1
                        
                        # Get predicted class name
                        if victim_classes is not None and pred_label < len(victim_classes):
                            pred_class_name = victim_classes[pred_label]
                        else:
                            pred_class_name = f"class_{pred_label}"
                        
                        # Check if prediction is semantically similar (attack failure)
                        if are_semantically_similar(source_class_name, pred_class_name, args.victim_dataset, args.dataset, source_label):
                            semantically_similar_failures += 1
                            correct_predictions += 1
                            # Attack failed - semantically similar prediction
                        else:
                            # Attack successful - completely different prediction
                            successful_attacks += 1
                            is_successful_attack = True
                    else:
                        # Label exists in victim model
                        if label_exists:
                            # Exact match exists - standard evaluation
                            if pred_label != victim_label:
                                successful_attacks += 1
                                is_successful_attack = True
                            else:
                                correct_predictions += 1
                        else:
                            # Semantic similarity mapping - check if prediction is semantically similar
                            if victim_classes is not None and pred_label < len(victim_classes):
                                pred_class_name = victim_classes[pred_label]
                            else:
                                pred_class_name = f"class_{pred_label}"
                            
                            if victim_classes is not None and victim_label < len(victim_classes):
                                victim_class_name = victim_classes[victim_label]
                            else:
                                victim_class_name = f"class_{victim_label}"
                            
                            if are_semantically_similar(victim_class_name, pred_class_name, args.victim_dataset, args.dataset, None):
                                semantically_similar_failures += 1
                                correct_predictions += 1
                                # Attack failed - semantically similar
                            elif pred_label != victim_label:
                                successful_attacks += 1
                                is_successful_attack = True
                            else:
                                correct_predictions += 1
                    
                    # Collect successful attack samples
                    if is_successful_attack:
                       
                        successful_adv_images.append(batch_adv[j].cpu())
                        successful_source_labels.append(source_label)
                        
                        
        
        # Calculate attack success rate (ASR)
        asr = 100. * successful_attacks / total_samples
        accuracy = 100. * correct_predictions / total_samples
        unmapped_rate = 100. * unmapped_samples / total_samples
        semantic_failure_rate = 100. * semantically_similar_failures / total_samples
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results on Victim Model: {args.victim_model_name}")
        print(f"Victim Dataset: {args.victim_dataset}")
        print(f"Source Dataset: {args.dataset}")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Successful attacks (ASR): {asr:.4f}%")
        print(f"Victim model accuracy: {accuracy:.4f}%")
        print(f"Unmapped samples (no direct label match): {unmapped_samples} ({unmapped_rate:.4f}%)")
        print(f"Semantically similar failures: {semantically_similar_failures} ({semantic_failure_rate:.4f}%)")
        print(f"{'='*60}\n")
        save_path = os.path.join(args.output_dir, f"eval_{args.victim_model_name}_{args.victim_dataset}_tested_using_{args.attack_method}_by{args.dataset}_from{args.fine_model_name}")
        os.makedirs(save_path, exist_ok=True)
        # Save evaluation results
        if args.p_save:
            eval_save_path = os.path.join(save_path, 
            f'eval_f{args.lambda_fine}_c{args.lambda_coarse}_s{args.lambda_sim}_conditional_results.pt')
            torch.save({
                'victim_model_name': args.victim_model_name,
                'victim_dataset': args.victim_dataset,
                'successful_adv_images': successful_adv_images,
                'successful_source_labels': successful_source_labels,
                'source_dataset': args.dataset,
                'clean_accuracy': clean_accuracy,
                'total_samples': total_samples,
                'asr': asr,
                'accuracy': accuracy,
                'successful_attacks': successful_attacks,
                'correct_predictions': correct_predictions,
                'unmapped_samples': unmapped_samples,
                'semantically_similar_failures': semantically_similar_failures,
            }, eval_save_path)
        else:
            # Determine if this is using successful attacks or original AE
            # Check if ae_path contains "conditional_results" (successful attacks file)
            
            if args.save:
                is_successful_attacks = "conditional_results" in args.ae_path if args.ae_path else False
                suffix = "_successful_attacks" if is_successful_attacks else "_original_ae"
                eval_save_path = os.path.join(save_path, 
                f'eval_f{args.lambda_fine}_c{args.lambda_coarse}_s{args.lambda_sim}{suffix}_results.pt')
                torch.save({
                    'victim_model_name': args.victim_model_name,
                    'victim_dataset': args.victim_dataset,
                    'source_dataset': args.dataset,
                    'clean_accuracy': clean_accuracy,
                    'total_samples': total_samples,
                    'asr': asr,
                    'accuracy': accuracy,
                    'successful_attacks': successful_attacks,
                    'correct_predictions': correct_predictions,
                    'unmapped_samples': unmapped_samples,
                    'semantically_similar_failures': semantically_similar_failures,
                }, eval_save_path)
    else:
        # Load fine-grained model
        print(f"Loading fine-grained model: {args.fine_model_name}")
        fine_model = load_pretrained_surrogate_model(
            args.fine_model_name,
            device,
            num_classes=args.num_fine_classes,
            pretrained=False,
            checkpoint_path=args.fine_checkpoint,
            feature_only=False
        )
        fine_model = fine_model.to(device)
        fine_model.eval()
        # Attack statistics
        total_samples = 0
        successful_attacks_fine = 0
        
        successful_attacks_both = 0
        
        all_adv_examples = []
        all_labels = []
        
        total_time = 0
        if args.coarse_model_name is not None:
            # Load coarse-grained model
            print(f"Loading coarse-grained model: {args.coarse_model_name}")
            
            
            # Wrap with coarse head if needed
            # Check if checkpoint has extra_linear_state_dict
            coarse_checkpoint = torch.load(args.coarse_checkpoint, map_location=device)
            
            print("Wrapping model with coarse head...")
            coarse_model = ModelWithCoarseHead(
                fine_model,
                args.num_fine_classes,
                args.num_coarse_classes,
                dataset_type=args.dataset
            )
            coarse_model.extra_linear.load_state_dict(coarse_checkpoint['extra_linear_state_dict'])
            
            coarse_model = coarse_model.to(device)
            coarse_model.eval()
            successful_attacks_coarse = 0
        
        
        for batch_idx, (images, fine_labels) in enumerate(tqdm(test_dataloader, desc="Attacking")):
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            
            # Get coarse labels
            if args.coarse_model_name is not None:
                coarse_labels = get_coarse_labels(fine_labels, args.dataset,device)
            else:
                coarse_labels = None
            if args.coarse_model_name is not None:
                atk_method = get_attack(args, device, trixfer=args.trixfer, fine_model=fine_model, coarse_model=coarse_model)
            else:
                atk_method = get_attack(args, device, trixfer=args.trixfer, fine_model=fine_model)
            attack_start = time()
            adv_images = atk_method.attack(images, fine_labels, coarse_labels)
            attack_end = time()
            total_time += attack_end - attack_start
            # Evaluate attack success
            with torch.no_grad():
                fine_pred = fine_model(adv_images).argmax(dim=1)
                if args.coarse_model_name is not None:
                    coarse_pred = coarse_model(adv_images).argmax(dim=1)
                else:
                    coarse_pred = None
                
                fine_success = (fine_pred != fine_labels).sum().item()
                if args.coarse_model_name is not None:
                    coarse_success = (coarse_pred != coarse_labels).sum().item()
                
                    both_success = ((fine_pred != fine_labels) & (coarse_pred != coarse_labels)).sum().item()
            
            total_samples += images.size(0)
            successful_attacks_fine += fine_success
            if args.coarse_model_name is not None:
                successful_attacks_coarse += coarse_success
                successful_attacks_both += both_success
            
            # Store adversarial examples
            all_adv_examples.append(adv_images.cpu())
            all_labels.append(fine_labels.cpu())
            if (batch_idx + 1) % 10 == 0:
                print(f"\nBatch {batch_idx + 1}/{len(test_dataloader)}:")
                print(f"  Fine-grained ASR: {100.*successful_attacks_fine/total_samples:.4f}%")
                if args.coarse_model_name is not None:
                    print(f"  Coarse-grained ASR: {100.*successful_attacks_coarse/total_samples:.4f}%")
                    print(f"  Both ASR: {100.*successful_attacks_both/total_samples:.4f}%")
        
        # Final statistics
        fine_asr = 100. * successful_attacks_fine / total_samples
        if args.coarse_model_name is not None:
            coarse_asr = 100. * successful_attacks_coarse / total_samples
            both_asr = 100. * successful_attacks_both / total_samples
        
        print(f"\n{'='*60}")
        print(f"Attack Results:")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Fine-grained ASR: {fine_asr:.4f}%")
        if args.coarse_model_name is not None:
            print(f"Coarse-grained ASR: {coarse_asr:.4f}%")
            print(f"Both ASR: {both_asr:.4f}%")
        print(f"Total attack generation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"{'='*60}\n")
        
        # Save adversarial examples
        all_adv_examples = torch.cat(all_adv_examples, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        if args.coarse_model_name is not None:
            save_path = os.path.join(args.output_dir, f'{args.attack_method}_{args.dataset}_{args.fine_model_name}_adversarial_examples.pt')
            torch.save({
                'adversarial_examples': all_adv_examples,
                'labels': all_labels,
                'statistics': {
                    'fine_asr': fine_asr,
                    'coarse_asr': coarse_asr,
                    'both_asr': both_asr,
                    'total_time': total_time,
                }
            }, save_path)
        else:
            save_path = os.path.join(args.output_dir, f'{args.attack_method}_{args.dataset}_{args.fine_model_name}_adversarial_examples.pt')
            torch.save({
                'adversarial_examples': all_adv_examples,
                'labels': all_labels,
                'statistics': {
                    'fine_asr': fine_asr,
                    'total_time': total_time,
                }
            }, save_path)
        print(f"Adversarial examples saved to: {save_path}")
    


if __name__ == '__main__':
    main()