import torch
import torch.nn as nn
#--------------------

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    AdaLoraConfig,
    IA3Config,
)
PEFT_AVAILABLE = True


#--------------------


# ==================== Custom implementations for methods not in PEFT ====================

class PromptTuningForVision(nn.Module):
    """Prompt Tuning for Vision Transformers (custom implementation)
    
    Places prompts AFTER CLS token: [cls_token, prompts, patch_tokens]
    """
    def __init__(self, num_prompts=10, embed_dim=768):
        super().__init__()
        self.num_prompts = num_prompts
        # Initialize with smaller values (better for training)
        self.prompts = nn.Parameter(torch.randn(1, num_prompts, embed_dim) * 0.02)
        
    def forward(self, cls_token, patch_tokens):
        """
        Args:
            cls_token: [batch, 1, embed_dim] - CLS token
            patch_tokens: [batch, num_patches, embed_dim] - patch embeddings
        Returns:
            Concatenated sequence: [batch, 1 + num_prompts + num_patches, embed_dim]
        """
        batch_size = cls_token.shape[0]
        prompts = self.prompts.expand(batch_size, -1, -1).to(cls_token.device)
        # Concatenate: [cls_token, prompts, patch_tokens]
        return torch.cat([cls_token, prompts, patch_tokens], dim=1)


class PrefixAttentionWrapper(nn.Module):
    """Wraps attention module to inject prefix keys and values"""
    def __init__(self, attention_module, prefix_k, prefix_v, original_embed_dim=None, original_num_heads=None):
        super().__init__()
        self.attention = attention_module
        self.register_parameter('prefix_k', prefix_k)
        self.register_parameter('prefix_v', prefix_v)
        self.num_prefix = prefix_k.shape[0]
        self.original_embed_dim = original_embed_dim or prefix_k.shape[1]
        self.original_num_heads = original_num_heads
        
       
        self.prefix_k_proj = None
        self.prefix_v_proj = None
        
    def forward(self, x, attn_mask=None, **kwargs):
        """
        Args:
            x: [batch, seq_len, embed_dim] for ViT
            attn_mask: Optional attention mask (for compatibility with timm)
            **kwargs: Additional arguments for compatibility
        Returns:
            Output from attention with prefix injected
        """
        is_4d = x.dim() == 4
        
        if is_4d:
            # TinyViT: [B, C, H, W] 
            B, C, H, W = x.shape
            N = H * W
            # Reshape to 3D for attention computation: [B, H*W, C]
            x_3d = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        else:
            # ViT: [B, N, C]
            B, N, C = x.shape
            x_3d = x
            
        num_heads = getattr(self.attention, 'num_heads', 12)
        head_dim = C // num_heads
        if C != self.original_embed_dim:
            proj_name_k = f'prefix_k_proj_{C}'
            proj_name_v = f'prefix_v_proj_{C}'
            
            if not hasattr(self, proj_name_k):
                # Create and register projection layer
                proj_k = nn.Linear(self.original_embed_dim, C, bias=False).to(x.device)
                proj_v = nn.Linear(self.original_embed_dim, C, bias=False).to(x.device)
                self.register_module(proj_name_k, proj_k)
                self.register_module(proj_name_v, proj_v)
                proj_k.weight.requires_grad = True
                
                proj_v.weight.requires_grad = True
                
                proj_k.train()
                proj_v.train()
               
            else:
                proj_k = getattr(self, proj_name_k)
                proj_v = getattr(self, proj_name_v)
            
            # Project prefix to current layer's dimension
            prefix_k = proj_k(self.prefix_k)  # [num_prefix, C]
            prefix_v = proj_v(self.prefix_v)  # [num_prefix, C]
        else:
            prefix_k = self.prefix_k
            prefix_v = self.prefix_v
        if is_4d and hasattr(self.attention, 'norm'):
            # TinyViT pattern: norm -> qkv
            x_norm = self.attention.norm(x_3d)  # [B, H*W, C]
            qkv = self.attention.qkv(x_norm)
        else:
            # Standard ViT: qkv directly
            qkv = self.attention.qkv(x_3d)
        
        # Reshape qkv: [B, N, 3*C] -> [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        prefix_k = prefix_k.unsqueeze(0)  # [1, num_prefix, embed_dim]
        prefix_k = prefix_k.reshape(1, self.num_prefix, num_heads, head_dim)  # [1, num_prefix, num_heads, head_dim]
        prefix_k = prefix_k.permute(0, 2, 1, 3)  # [1, num_heads, num_prefix, head_dim]
        prefix_k = prefix_k.expand(B, -1, -1, -1)
        
    
        prefix_v = prefix_v.unsqueeze(0)
        prefix_v = prefix_v.reshape(1, self.num_prefix, num_heads, head_dim)
        prefix_v = prefix_v.permute(0, 2, 1, 3)
        prefix_v = prefix_v.expand(B, -1, -1, -1)
        
        # Concatenate prefix with K and V
        k = torch.cat([prefix_k, k], dim=2)  # [B, num_heads, num_prefix + N, head_dim]
        v = torch.cat([prefix_v, v], dim=2)  # [B, num_heads, num_prefix + N, head_dim]
        
        # Compute attention scores
        scale = getattr(self.attention, 'scale', head_dim ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, num_heads, N, num_prefix + N]
        
        if attn_mask is not None:
            # Handle different mask shapes
            if attn_mask.dim() == 3:  # [B, N, N]
                
                prefix_mask = torch.ones(B, N, self.num_prefix, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([prefix_mask, attn_mask], dim=2)  # [B, N, num_prefix + N]
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N, num_prefix + N]
            elif attn_mask.dim() == 4:  # [B, 1, N, N] or [B, num_heads, N, N]
                # Extend mask: [B, 1/num_heads, N, num_prefix + N]
                prefix_mask = torch.ones(
                    attn_mask.shape[0], attn_mask.shape[1], N, self.num_prefix,
                    device=attn_mask.device, dtype=attn_mask.dtype
                )
                attn_mask = torch.cat([prefix_mask, attn_mask], dim=3)  # [B, 1/num_heads, N, num_prefix + N]
            
            # Apply mask (set masked positions to -inf before softmax)
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn = attn.softmax(dim=-1)
        
        # Apply dropout if exists
        if hasattr(self.attention, 'attn_drop'):
            attn = self.attention.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # Apply output projection
        if hasattr(self.attention, 'proj'):
            x = self.attention.proj(x)
        if hasattr(self.attention, 'proj_drop'):
            x = self.attention.proj_drop(x)
        if is_4d:
            # [B, H*W, C] -> [B, C, H, W]
            x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

def add_prefix_tuning_to_attention_blocks(model, num_prefix_tokens=10):
    """
    Add prefix tuning to all attention blocks in the model.
    This modifies the model in-place by wrapping attention modules.
    
    Args:
        model: timm model (ViT or TinyViT)
        num_prefix_tokens: Number of prefix tokens
    
    Returns:
        List of prefix modules (for parameter freezing)
    """
    prefix_modules = []
    
    # Get embed_dim and num_heads from first attention block
    embed_dim = None
    num_heads = None
    
    # Find first attention block to get dimensions
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        # Standard ViT
        first_block = model.blocks[0]
        if hasattr(first_block, 'attn'):
            attn = first_block.attn
            if hasattr(attn, 'qkv'):
                embed_dim = attn.qkv.in_features
                num_heads = getattr(attn, 'num_heads', 12)
    elif hasattr(model, 'stages'):
        # TinyViT
        for stage in model.stages:
          
            if isinstance(stage, (nn.ModuleList, nn.Sequential)):
                for block in stage:
                    if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                        embed_dim = block.attn.qkv.in_features
                        num_heads = getattr(block.attn, 'num_heads', 12)
                        break
                if embed_dim is not None:
                    break
            elif hasattr(stage, 'blocks'):
                for block in stage.blocks:
                # Check if it's a TinyVitBlock (has attn and mlp)
                    if hasattr(block, 'attn') and hasattr(block, 'mlp'):
                        if hasattr(block.attn, 'qkv'):
                            embed_dim = block.attn.qkv.in_features
                            num_heads = getattr(block.attn, 'num_heads', 12)
                            break
                    if embed_dim is not None:
                        break
    elif hasattr(model, 'layers'):  # Swin Transformer
        # Swin has layers containing SwinTransformerStage objects
        for layer in model.layers:
            if hasattr(layer, 'blocks'):  # SwinTransformerStage has blocks
                for block in layer.blocks:
                    if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                        embed_dim = block.attn.qkv.in_features
                        num_heads = getattr(block.attn, 'num_heads', 12)
                        break
                if embed_dim is not None:
                    break
    if embed_dim is None:
        raise ValueError("Could not determine embed_dim from model structure")
    
    if num_heads is None:
        num_heads = 12  # Default
    
    # Create shared prefix parameters (shared across all layers)
    prefix_k = nn.Parameter(torch.randn(num_prefix_tokens, embed_dim))
    prefix_v = nn.Parameter(torch.randn(num_prefix_tokens, embed_dim))
    prefix_modules.append(('prefix_k', prefix_k))
    prefix_modules.append(('prefix_v', prefix_v))
    
    # Apply to all attention blocks
    if hasattr(model, 'blocks'):
        # Standard ViT
        for block in model.blocks:
            if hasattr(block, 'attn'):
                # Wrap attention module
                original_attn = block.attn
                block.attn = PrefixAttentionWrapper(original_attn, prefix_k, prefix_v)
    
    elif hasattr(model, 'stages'):
        # TinyViT
        for stage in model.stages:
            if isinstance(stage, (nn.ModuleList, nn.Sequential)):
                for block in stage:
                    if hasattr(block, 'attn'):
                        original_attn = block.attn
                        block.attn = PrefixAttentionWrapper(original_attn, prefix_k, prefix_v)
            elif hasattr(stage, 'blocks'):
                for block in stage.blocks:
                    if hasattr(block, 'attn'):
                        original_attn = block.attn
                        block.attn = PrefixAttentionWrapper(original_attn, prefix_k, prefix_v)
    elif hasattr(model, 'layers'):  # Swin Transformer
        # Swin has layers containing SwinTransformerStage objects
        for layer in model.layers:
            if hasattr(layer, 'blocks'):  # SwinTransformerStage has blocks
                for block in layer.blocks:
                    if hasattr(block, 'attn'):
                        original_attn = block.attn
                        block.attn = PrefixAttentionWrapper(original_attn, prefix_k, prefix_v)
    
    return prefix_modules

class PromptTuningForTinyViT(nn.Module):
    """Prompt tuning with projection to maintain channel dimensions"""
    def __init__(self, num_prompts=10, embed_dim=768, original_channels=None):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        
        # Prompts as learnable feature maps
        self.prompts = nn.Parameter(
            torch.randn(num_prompts * embed_dim, 1, 1)
        )
        
        # Projection to map concatenated features back to original channels
        # This is trainable and learns how to combine prompts with input
        if original_channels is not None:
            self._create_projection(original_channels)
        else:
            self.projection = None
    
    def _create_projection(self, original_channels):
        prompt_channels = self.num_prompts * self.embed_dim
        total_channels = original_channels + prompt_channels
        
        # Use 1x1 conv to project back to original channels
        self.projection = nn.Sequential(
            nn.Conv2d(total_channels, original_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(original_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size, channels, H, W = x.shape
        
        if self.projection is None:
            self._create_projection(channels)
            self.projection = self.projection.to(x.device)
        
        # Expand prompts
        prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, H, W)
        
        # Concatenate
        x_with_prompts = torch.cat([x, prompts], dim=1)
        
        # Project back to original channels
        x = self.projection(x_with_prompts)
        
        return x


# ==================== PEFT Wrapper for Vision Models ====================
class VisionPEFTWrapper(nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        labels=None,
        **kwargs
    ):
        # Redirect input_ids → pixel_values
        if pixel_values is None and input_ids is not None:
            pixel_values = input_ids

        return self.vision_model(pixel_values)








# ==================== Model Preparation Functions ====================

def prepare_model_for_adaptation(model, method='full_finetuning', **kwargs):
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
        # Only classifier head is trainable
        if hasattr(model, 'stages') or hasattr(model, 'layers'):
            if hasattr(model, 'head'):
                for param in model.head.fc.parameters():
                    param.requires_grad = True
            
            elif hasattr(model, 'classifier'):
                for param in model.classifier.fc.parameters():
                    param.requires_grad = True
        else:
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
        return model, None


    elif method == 'full_finetuning':
        # All parameters trainable
        for param in model.parameters():
            param.requires_grad = True
        return model, None
        
    elif method == 'adapter':
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for adapter method. Install with: pip install peft")
        if hasattr(model, 'stages'):
            target_modules = ["qkv", "proj", "fc1", "fc2"]  # PEFT will find 
        elif hasattr(model, 'layers'):
            target_modules = ["qkv", "proj", "mlp.fc1", "mlp.fc2"]
        else:
            target_modules = ["qkv", "proj", "fc1", "fc2"]
    
        adapter_type = kwargs.get('adapter_type', 'ia3')  # 'ia3' or 'adalora'
        
        if adapter_type == 'ia3':
            if hasattr(model, 'layers'):
                feedforward_modules = ["mlp.fc1", "mlp.fc2"]
            else:
                feedforward_modules = ["fc1", "fc2"]
            peft_config = IA3Config(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,  
                feedforward_modules=feedforward_modules
            )
        else:  # adalora
            peft_config = AdaLoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                init_r=kwargs.get('adapter_dim', 12),
                target_r=kwargs.get('adapter_dim', 8),
                target_modules=target_modules,
            )
        
        # Note: PEFT works best with HuggingFace models
       
        try:
            model = VisionPEFTWrapper(model)
            model = get_peft_model(model, peft_config)
            
            vision_model = model.base_model.model.vision_model
            if hasattr(vision_model, 'stages'):
                if hasattr(vision_model, 'head'):
                    for param in vision_model.head.fc.parameters():
                        param.requires_grad = True
                
                elif hasattr(vision_model, 'classifier'):
                    for param in vision_model.classifier.fc.parameters():
                        param.requires_grad = True

            else:
                if hasattr(vision_model, 'head'):
                    for param in vision_model.head.parameters():
                        param.requires_grad = True
                
                elif hasattr(vision_model, 'classifier'):
                    for param in vision_model.classifier.parameters():
                        param.requires_grad = True
                  
                
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
        except Exception as e:
            print(f"Warning: PEFT adapter failed, using custom adapter: {e}")
            # Fallback to custom adapter implementation
            model = add_custom_adapters(model, adapter_dim=kwargs.get('adapter_dim', 12))
        
        return model, None
        
    elif method == 'lora':
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA method. Install with: pip install peft")
        
        # Use PEFT's LoRA
        if hasattr(model, 'layers'):
            target_modules = ["qkv", "proj", "mlp.fc1", "mlp.fc2"]
        else:
            target_modules = ["qkv", "proj", "fc1", "fc2"]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=kwargs.get('lora_rank', 8),
            lora_alpha=kwargs.get('lora_alpha', 16),
            lora_dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=kwargs.get('target_modules', target_modules),
            bias="none",
        )
        
        try:
            model = VisionPEFTWrapper(model)
            model = get_peft_model(model, lora_config)
            
            vision_model = model.base_model.model.vision_model
            if hasattr(vision_model, 'stages'):
                if hasattr(vision_model, 'head'):
                    for param in vision_model.head.fc.parameters():
                        param.requires_grad = True
                
                elif hasattr(vision_model, 'classifier'):
                    for param in vision_model.classifier.fc.parameters():
                        param.requires_grad = True

            else:
                if hasattr(vision_model, 'head'):
                    print("trainable head")
                    for param in vision_model.head.parameters():
                        param.requires_grad = True
                
                elif hasattr(vision_model, 'classifier'):
                    for param in vision_model.classifier.parameters():
                        param.requires_grad = True
                
                
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
            
        except Exception as e:
            print(f"Warning: PEFT LoRA failed: {e}")
            print("Note: PEFT is designed for HuggingFace models. For timm models,")
            print("you may need to convert the model or use a custom LoRA implementation.")
            raise
        
        return model, None
        
    elif method == 'prefix_tuning':
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for prefix tuning. Install with: pip install peft")
        
        # Use PEFT's Prefix Tuning
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            num_virtual_tokens=kwargs.get('num_prefix_tokens', 10),
        )
        
        try:
            model = get_peft_model(model, prefix_config)
            return model, None
        except Exception as e:
            print(f"Warning: PEFT prefix tuning failed: {e}")
            # Fallback to custom implementation if needed
        num_prefix_tokens = kwargs.get('num_prefix_tokens', 10)
        prefix_modules = add_prefix_tuning_to_attention_blocks(model, num_prefix_tokens)
    
        for param in model.parameters():
            param.requires_grad = False

        for name, module in model.named_modules():
            if isinstance(module, PrefixAttentionWrapper):
                if hasattr(module, 'prefix_k'):
                    module.prefix_k.requires_grad = True
                if hasattr(module, 'prefix_v'):
                    module.prefix_v.requires_grad = True
                for param_name, param in module.named_parameters():
                    if param_name.startswith('prefix_k_proj') or param_name.startswith('prefix_v_proj'):
                        param.requires_grad = True
    
                    

        for name, module in model.named_modules():
            if isinstance(module, PrefixAttentionWrapper):
                model.prefix_k = module.prefix_k
                model.prefix_v = module.prefix_v
                break
            
        if hasattr(model, 'stages') or hasattr(model, 'layers'):
                if hasattr(model, 'head'):
                    for param in model.head.fc.parameters():
                        param.requires_grad = True
                
                elif hasattr(model, 'classifier'):
                    for param in model.classifier.fc.parameters():
                        param.requires_grad = True

        else:
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
        print(f"Added prefix tuning with {num_prefix_tokens} tokens")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        return model, None
        
    elif method == 'prompt_tuning':
        
        if PEFT_AVAILABLE:
            try:
                prompt_config = PromptTuningConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    num_virtual_tokens=kwargs.get('num_prompts', 10),
                    prompt_tuning_init=PromptTuningInit.RANDOM,
                )
                model = get_peft_model(model, prompt_config)
                return model, None
            except Exception as e:
                print(f"Warning: PEFT prompt tuning failed, using custom: {e}")
        
        # Custom implementation for vision transformers
        num_prompts = kwargs.get('num_prompts', 10)
    
        # Get embed_dim from positional embeddings (most reliable)
        if hasattr(model, 'pos_embed'):
            embed_dim = model.pos_embed.shape[-1]
        elif hasattr(model, 'embed_dim'):
            embed_dim = model.embed_dim
        elif hasattr(model, 'blocks') and len(model.blocks) > 0:
            if hasattr(model.blocks[0], 'norm1'):
                embed_dim = model.blocks[0].norm1.normalized_shape[0]
            else:
                embed_dim = 768  # Default
        else:
            embed_dim = 768  # Default
        
        if hasattr(model, 'stages'):
            # TinyViT
            original_channels = None
            if hasattr(model, 'stem'):
                if hasattr(model.stem, 'out_channels'):
                    original_channels = model.stem.out_channels
            else:
                for stage in model.stages:
                    if isinstance(stage, (nn.ModuleList, nn.Sequential)) and len(stage) > 0:
                        first_block = stage[0]
                        if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'qkv'):
                            original_channels = first_block.attn.qkv.in_features
                            break
            
            prompt_module = PromptTuningForTinyViT(
                num_prompts, 
                embed_dim, 
                original_channels=original_channels)
            
        elif hasattr(model, 'layers'):
            # Swin Transformer
            embed_dim = None
            if hasattr(model, 'patch_embed'):
                if hasattr(model.patch_embed, 'out_channels'):
                    embed_dim = model.patch_embed.out_channels
                elif hasattr(model.patch_embed, 'num_features'):
                    embed_dim = model.patch_embed.num_features
                else:
                    # Try to infer from the Conv2d layer
                    for name, module in model.patch_embed.named_modules():
                        if isinstance(module, nn.Conv2d):
                            embed_dim = module.out_channels
                            break
            
            # Fallback to default if not found
            if embed_dim is None:
                embed_dim = 128  # Default for Swin-Base
            
            # Don't set original_channels - let it be created dynamically in forward()
            # This ensures it matches the actual input from patch_embed
            prompt_module = PromptTuningForTinyViT(
                num_prompts, 
                embed_dim,  # Should match patch_embed output channels (e.g., 128 for Swin-Base)
                original_channels=None  # Will be set dynamically in forward() based on actual input
            )
                
        else:
            # Standard ViT
            prompt_module = PromptTuningForVision(num_prompts, embed_dim)
        
        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Make prompt parameters trainable
        for param in prompt_module.parameters():
            param.requires_grad = True
        
        # Make head trainable (if exists)
        if hasattr(model, 'stages') or hasattr(model, 'layers'):
                if hasattr(model, 'head'):
                    for param in model.head.fc.parameters():
                        param.requires_grad = True
                
                elif hasattr(model, 'classifier'):
                    for param in model.classifier.fc.parameters():
                        param.requires_grad = True

        else:
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
            
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
        return model, prompt_module
        
    else:
        raise ValueError(f"Unknown adaptation method: {method}")


def interpolate_pos_embed(pos_embed, num_patches, num_prompts):
    """
    Interpolate positional embeddings and add space for prompts.
    
    Args:
        pos_embed: [1, original_len, D] - original positional embeddings
        num_patches: number of patch tokens in current input
        num_prompts: number of prompt tokens
    
    Returns:
        Extended positional embeddings: [1, 1 + num_prompts + num_patches, D]
        Structure: [cls_pos, prompt_pos (zeros), patch_pos (interpolated)]
    """
    import torch.nn.functional as F
    import math
    
    # Extract CLS token positional embedding
    cls_pos_embed = pos_embed[:, 0:1, :]  # [1, 1, D]
    
    # Extract patch positional embeddings
    patch_pos_embed = pos_embed[:, 1:, :]  # [1, num_patches_orig, D]
    
    original_num_patches = patch_pos_embed.shape[1]
    dim = patch_pos_embed.shape[2]
    
    def find_grid_dims(num_patches):
        sqrt_n = int(math.sqrt(num_patches))
        # Check if it's a perfect square
        if sqrt_n * sqrt_n == num_patches:
            return sqrt_n, sqrt_n
        
        # Find the closest factors
        for h in range(sqrt_n, 0, -1):
            if num_patches % h == 0:
                w = num_patches // h
                return h, w
        
        # Fallback: use square root (will truncate)
        return sqrt_n, (num_patches + sqrt_n - 1) // sqrt_n
    H , W = find_grid_dims(original_num_patches)
    if H * W != original_num_patches:
        # If we can't reshape exactly, we need to pad or truncate
        # This shouldn't happen with the find_grid_dims function, but just in case
        actual_patches = H * W
        if actual_patches < original_num_patches:
            # Pad with zeros
            padding = torch.zeros(1, actual_patches - original_num_patches, dim, 
                                device=patch_pos_embed.device, dtype=patch_pos_embed.dtype)
            patch_pos_embed = torch.cat([patch_pos_embed, padding], dim=1)
        elif actual_patches > original_num_patches:
            patch_pos_embed = patch_pos_embed[:, :actual_patches, :]
    
    patch_pos_embed = patch_pos_embed.reshape(1, H, W, dim).permute(0, 3, 1, 2)  # [1, D, H, W]
    
    # Interpolate to match current number of patches
    new_H, new_W = find_grid_dims(num_patches)
    patch_pos_embed = F.interpolate(
        patch_pos_embed, 
        size=(new_H, new_W), 
        mode='bicubic', 
        align_corners=False
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, new_H * new_W, dim)
    
    if new_H * new_W != num_patches:
        if new_H * new_W < num_patches:
            # Pad with zeros
            padding = torch.zeros(1, num_patches - new_H * new_W, dim,
                                device=patch_pos_embed.device, dtype=patch_pos_embed.dtype)
            patch_pos_embed = torch.cat([patch_pos_embed, padding], dim=1)
        else:
            
            patch_pos_embed = patch_pos_embed[:, :num_patches, :]
    prompt_pos_embed = torch.zeros(
        1, num_prompts, dim, 
        device=pos_embed.device, 
        dtype=pos_embed.dtype
    )
    
    # Concatenate: [cls_pos, prompt_pos, patch_pos]
    return torch.cat([cls_pos_embed, prompt_pos_embed, patch_pos_embed], dim=1)

def add_custom_adapters(model, adapter_dim):
    adapter_count = 0
    
    # Handle standard ViT models with blocks
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block, 'norm1'):
                dim = block.norm1.normalized_shape[0]
                adapter_bottleneck = adapter_dim if adapter_dim else dim * 4
                block.adapter = nn.Sequential(
                    nn.Linear(dim, adapter_bottleneck),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(adapter_bottleneck, dim)
                )
                # Freeze original block parameters
                for param in block.parameters():
                    param.requires_grad = False
                # Only adapter is trainable
                for param in block.adapter.parameters():
                    param.requires_grad = True
                adapter_count += 1
    
    # Handle TinyViT models with stages
    elif hasattr(model, 'stages'):
        for stage_idx, stage in enumerate(model.stages):
            for block_idx, block in enumerate(stage):
                # Check if it's a TinyVitBlock (has attn and mlp)
                if hasattr(block, 'attn') and hasattr(block, 'mlp'):
                    # Get dimension from attention or MLP
                    if hasattr(block.attn, 'qkv'):
                        dim = block.attn.qkv.in_features
                    elif hasattr(block.mlp, 'fc1'):
                        dim = block.mlp.fc1.in_features
                    else:
                        continue 
                    
                    adapter_bottleneck = adapter_dim if adapter_dim else dim * 4
                    
                    # Add adapter after MLP (standard placement)
                    block.adapter = nn.Sequential(
                        nn.Linear(dim, adapter_bottleneck),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(adapter_bottleneck, dim)
                    )
                    
                    # Freeze original block parameters
                    for param in block.parameters():
                        param.requires_grad = False
                    # Only adapter is trainable
                    for param in block.adapter.parameters():
                        param.requires_grad = True
                    adapter_count += 1
    elif hasattr(model, 'layers'):  # Swin Transformer
        # Swin has layers containing SwinTransformerStage objects
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, 'blocks'):  # SwinTransformerStage has blocks
                for block_idx, block in enumerate(layer.blocks):
                    # Check if it's a SwinTransformerBlock (has attn and mlp)
                    if hasattr(block, 'attn') and hasattr(block, 'mlp'):
                        # Get dimension from attention or MLP
                        if hasattr(block.attn, 'qkv'):
                            dim = block.attn.qkv.in_features
                        elif hasattr(block.mlp, 'fc1'):
                            dim = block.mlp.fc1.in_features
                        else:
                            continue 
                        
                        adapter_bottleneck = adapter_dim if adapter_dim else dim * 4
                        
                        # Add adapter after MLP (standard placement)
                        block.adapter = nn.Sequential(
                            nn.Linear(dim, adapter_bottleneck),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(adapter_bottleneck, dim)
                        )
                        
                        # Freeze original block parameters
                        for param in block.parameters():
                            param.requires_grad = False
                        # Only adapter is trainable
                        for param in block.adapter.parameters():
                            param.requires_grad = True
                        adapter_count += 1
    else:
        raise ValueError("Model does not have 'blocks' or 'stages' attribute. Cannot add adapters.")
    
    print(f"Added {adapter_count} adapters to model")
    return model


class AdaptedVisionModel(nn.Module):
    def __init__(self, model, prompt_module=None, prefix_module=None):
        super().__init__()
        self.model = model
        self.prompt_module = prompt_module
        self.is_tinyvit = hasattr(model, 'stages')
        self.is_swin = hasattr(model, 'layers')  # Swin Transformer
        
        
    def forward(self, x):
        if self.prompt_module is not None and not self.is_tinyvit and not self.is_swin:
            if hasattr(self.model, 'patch_embed'):
                patch_tokens = self.model.patch_embed(x)  # [B, num_patches, D]
            else:
                # Fallback: try to find patch embedding
                for name, module in self.model.named_children():
                    if 'embed' in name.lower() or 'patch' in name.lower():
                        patch_tokens = module(x)
                        break
                else:
                    raise ValueError("Could not find patch embedding module")
            
            # Get CLS token
            B = patch_tokens.shape[0]
            if hasattr(self.model, 'cls_token'):
                
                cls_token = self.model.cls_token.expand(B, -1, -1)  # [B, 1, D]
            else:
                # Some models might not have explicit cls_token, create one
                dim = patch_tokens.shape[-1]
                cls_token = torch.zeros(B, 1, dim, device=patch_tokens.device, dtype=patch_tokens.dtype)
            
            # Concatenate: [cls_token, prompts, patch_tokens]
            x = self.prompt_module(cls_token, patch_tokens)
            
            # Handle positional embeddings
            if hasattr(self.model, 'pos_embed'):
                num_patches = patch_tokens.shape[1]
                num_prompts = self.prompt_module.num_prompts
                # Interpolate and extend positional embeddings
                pos_embed = interpolate_pos_embed(
                    self.model.pos_embed, 
                    num_patches, 
                    num_prompts
                )
                x = x + pos_embed.to(x.device)
            elif hasattr(self.model, '_pos_embed'):
                # If model has _pos_embed method, it might handle interpolation
                x = self.model._pos_embed(x)
            
            # Apply positional dropout if exists
            if hasattr(self.model, 'pos_drop'):
                x = self.model.pos_drop(x)
            
            # Forward through transformer blocks
            if hasattr(self.model, 'blocks'):
                x = self.model.blocks(x)
            
            # Apply norm
            if hasattr(self.model, 'norm'):
                x = self.model.norm(x)
            
            # Extract CLS token (still at position 0)
            cls_token_output = x[:, 0, :]  # [B, D]
            # Apply head
            if hasattr(self.model, 'head'):
                x = self.model.head(cls_token_output)
            elif hasattr(self.model, 'forward_head'):
                x = self.model.forward_head(cls_token_output)
            
            else:
                x = cls_token_output
            
            return x
            
        elif self.prompt_module is not None and self.is_tinyvit:
            if hasattr(self.model, 'stem'):  
                x = self.model.stem(x)
            else:
                for name, module in self.model.named_children():
                    if 'embed' in name.lower() or 'patch' in name.lower():
                        x = module(x)
                        break
            
            # Apply prompts (TinyViT handles 4D tensors differently)
            x = self.prompt_module(x)
            
            # Forward through stages
            if hasattr(self.model, 'stages'):
                for stage in self.model.stages:
                    x = stage(x)
            
            # Apply norm and head
            if hasattr(self.model, 'norm'):
                x = self.model.norm(x)
            elif hasattr(self.model, 'head_norm'):
                x = self.model.head_norm(x)
            
            if hasattr(self.model, 'head'):
                x = self.model.head(x)
            elif hasattr(self.model, 'forward_head'):
                x = self.model.forward_head(x)
            
            return x
        elif self.is_swin:
            # Swin Transformer - forward through layers
            # Swin doesn't use CLS tokens, so prompt tuning would need special handling
            # For now, just do standard forward pass
            if hasattr(self.model, 'patch_embed'):
                x = self.model.patch_embed(x)
            x = self.prompt_module(x)
            # Forward through layers (SwinTransformerStage objects)
            if hasattr(self.model, 'layers'):
                for layer in self.model.layers:
                    x = layer(x)
            
            # Apply norm
            if hasattr(self.model, 'norm'):
                x = self.model.norm(x)
            
            # Apply head
            if hasattr(self.model, 'head'):
                x = self.model.head(x)
            elif hasattr(self.model, 'forward_head'):
                x = self.model.forward_head(x)
            
            return x
        else:
            return self.model(x)


def fine_tune_setup(model, adaptation_method='full_finetuning', **adaptation_kwargs):
 
    # Prepare model for adaptation
    model, adaptation_module = prepare_model_for_adaptation(
        model, method=adaptation_method, **adaptation_kwargs
    )
    
    # Wrap if needed (for custom prompt tuning)
    if adaptation_module is not None and adaptation_method in ['prompt_tuning', 'prefix_tuning']:
        print("Using adaptation module")
        if adaptation_method == 'prompt_tuning':
            model = AdaptedVisionModel(model, prompt_module=adaptation_module)
    
    return model