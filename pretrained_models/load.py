import os
import torch
import timm
# print(timm.__version__)
# print(timm.list_models(pretrained=True))
# raise Exception('stop here')
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

swin_base_patch4_window7_224 = load_pretrained_model('swin_base_patch4_window7_224', pretrained=True)
# 3. DeiT III (Base) - supervised baseline
deit3_base = load_pretrained_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True)

# 4. DINOv2 ViT-S - self-supervised baseline
dinov2_vits = load_pretrained_model('vit_small_patch16_224.dino', pretrained=True)

# 5. CaiT (Class-Attention Transformer)
cait_xxs24_224 = load_pretrained_model('cait_xxs24_224', pretrained=True)
# Or other variants: cait_xs24_224, cait_s24_224, cait_s36_384

# 6. TinyViT
tinyvit_5m_224 = load_pretrained_model('tiny_vit_21m_224.dist_in22k_ft_in1k', pretrained=True)
# Or other variants: tiny_vit_11m_224, tiny_vit_21m_224, tiny_vit_21m_384

convnextv2_tiny = load_pretrained_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True)
mobilenetv2_100 = load_pretrained_model('mobilenetv2_100.ra_in1k', pretrained=True)
efficientvit_b2 = load_pretrained_model('efficientvit_b2.r224_in1k', pretrained=True)
inception_next_small = load_pretrained_model('inception_next_small.sail_in1k', pretrained=True)
def save_pretrained_models(save_dir='pretrained_models'):
    """
    Save all pretrained models to disk
    
    Args:
        save_dir: Directory to save the models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Model configurations: (model_name, model_object, filename)
    models_to_save = [
        ('vit_base_patch16_224', vit_b16, 'vit_b16_imagenet1k.pth'),
        ('deit3_base_patch16_224', deit3_base, 'deit3_base_imagenet1k.pth'),
        ('vit_small_patch14_dinov2.lvd142m', dinov2_vits, 'dinov2_vits.pth'),
        ('cait_xxs24_224', cait_xxs24_224, 'cait_xxs24_224_imagenet1k.pth'),
        ('tiny_vit_5m_224', tinyvit_5m_224, 'tinyvit_5m_224_imagenet1k.pth'),
        ('swin_base_patch4_window7_224', swin_base_patch4_window7_224, 'swin_base_patch4_window7_224_imagenet1k.pth'),
        ('convnextv2_tiny.fcmae_ft_in1k', convnextv2_tiny, 'convnextv2_tiny.fcmae_ft_in1k.pth'),
        ('mobilenetv2_100.ra_in1k', mobilenetv2_100, 'mobilenetv2_100.ra_in1k.pth'),
        ('efficientvit_b2.r224_in1k', efficientvit_b2, 'efficientvit_b2.r224_in1k.pth'),
        ('inception_next_small.sail_in1k', inception_next_small, 'inception_next_small.sail_in1k.pth'),
    ]
    
    for model_name, model, filename in models_to_save:
        save_path = os.path.join(save_dir, filename)
        # Save only the state_dict (recommended for pretrained models)
        torch.save({
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'num_classes': model.num_classes if hasattr(model, 'num_classes') else 1000,
        }, save_path)
        print(f"Saved {model_name} to {save_path}")
    
    print(f"\nAll models saved to {save_dir}/")

# Call this function to save all models
if __name__ == "__main__":
    save_pretrained_models(save_dir='pretrained_models')