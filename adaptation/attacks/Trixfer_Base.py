"""
Base of the Trixfer
"""
from abc import abstractmethod
from torch import nn 
import torch
import torch.nn.functional as F

def register_hidden_hook(model, model_name, layer_level=0.33):
    actual_model = model.base_model if hasattr(model, 'base_model') else model
    hidden = {}

    def hook_fn(module, input, output):
        hidden["h"] = output

    if hasattr(actual_model, 'stages'):
        target_module = actual_model.stages[int(len(actual_model.stages) * layer_level)]
    elif hasattr(actual_model, 'blocks'):
        target_module = actual_model.blocks[int(len(actual_model.blocks) * layer_level)]
    else:
        raise ValueError(f"Model {model_name} doesn't have 'stages' or 'blocks' attribute")

    handle = target_module.register_forward_hook(hook_fn)
    return hidden, handle

class Trixfer_Base:
    def __init__(self, eps=8/255, alpha=2/255, max_value=1., min_value=0., threshold=0., beta=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 fine_model=None, coarse_model=None, 
                 lambda_fine=1.0, lambda_coarse=1.0, lambda_sim=1.0,
                 model_name=None,layer_level=0.33):
        
        self.device = device
        
        # attack parameter
        self.eps = eps
        self.threshold = threshold
        self.max_value = max_value
        self.min_value = min_value
        self.beta = beta
        self.alpha = alpha
        
        # Hybrid loss parameters
        self.fine_model = fine_model
        self.coarse_model = coarse_model
        if self.fine_model is not None:
            self.fine_model.eval()
        if self.coarse_model is not None:
            self.coarse_model.eval()
        self.lambda_fine = lambda_fine
        self.lambda_coarse = lambda_coarse
        self.lambda_sim = lambda_sim
        self.model_name = model_name
        self.layer_level = layer_level
    def get_adv_example(self, ori_data, adv_data, grad, attack_step=None):
        """
        :param ori_data: original image
        :param adv_data: adversarial image in the last iteration
        :param grad: gradient in this iteration
        :return: adversarial example in this iteration
        """
        if attack_step is None:
            adv_example = adv_data.detach() + grad.sign() * self.alpha #FGSM
        else:
            adv_example = adv_data.detach() + grad.sign() * attack_step
        delta = torch.clamp(adv_example - ori_data.detach(), -self.eps, self.eps)
        return torch.clamp(ori_data.detach() + delta, max=self.max_value, min=self.min_value)

    def compute_hybrid_loss(self, ori_data, adv_data, fine_labels, coarse_labels = None):
        """
        Compute hybrid loss:
        L = λ_fine * CE(fine_model(x_adv), fine_labels) 
          + λ_coarse * CE(coarse_model(x_adv), coarse_labels) 
          - λ_sim * cosine_sim(features(x_orig), features(x_adv))
        
        Args:
            ori_data: Original images [B, C, H, W]
            adv_data: Adversarial images [B, C, H, W]
            fine_labels: Fine-grained labels [B]
            coarse_labels: Coarse-grained labels [B]
        
        Returns:
            Total loss and individual components
        """
        loss_func = torch.nn.CrossEntropyLoss() # graph-connected zero
        zero = adv_data.sum() * 0.0

        task_loss = zero
        if coarse_labels is not None:
            concept_loss = zero
        hidden_loss = zero
        # Fine-grained cross entropy loss
        if self.fine_model is not None:
            hidden, handle = register_hidden_hook(self.fine_model, self.model_name, self.layer_level)
            fine_logits = self.fine_model(adv_data) 
            task_loss = loss_func(fine_logits, fine_labels)
            
            adv_features = hidden["h"]
            handle.remove()
            ori_data_grad = ori_data.clone().requires_grad_(True)
            # Register hook for ORIGINAL features
            hidden_orig, handle_orig = register_hidden_hook(self.fine_model, self.model_name, self.layer_level)
            logits_orig = self.fine_model(ori_data_grad)
            fake = loss_func(logits_orig, fine_labels)
            orig_features = hidden_orig["h"]
            handle_orig.remove()
        # Coarse-grained cross entropy loss
        if self.coarse_model is not None and coarse_labels is not None:
            coarse_logits = self.coarse_model(adv_data)
            concept_loss = loss_func(coarse_logits, coarse_labels)
        
        
        if self.lambda_sim > 0:
          
            if task_loss.requires_grad:
                # Get gradient of task_loss w.r.t. adv_features
                feature_grad = torch.autograd.grad(
                    outputs=fake,
                    inputs=orig_features,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True
                )[0]
                if len(adv_features.shape) == 4:  # [B, C, H, W]
                    adv_features = F.adaptive_avg_pool2d(adv_features, (1, 1))
                    adv_features = adv_features.flatten(1)  # [B, C]
                elif len(adv_features.shape) == 3:  # [B, N, D]
                    adv_features = adv_features[:, 0] if adv_features.shape[1] > 0 else adv_features.mean(dim=1)
                elif len(adv_features.shape) == 2:  # [B, D]
                    adv_features = adv_features
                else:
                    raise ValueError(f"Unexpected adv_features shape: {adv_features.shape}")
                
                # Process orig_features
                if len(orig_features.shape) == 4:  # [B, C, H, W]
                    orig_features = F.adaptive_avg_pool2d(orig_features, (1, 1))
                    orig_features = orig_features.flatten(1)  # [B, C]
                elif len(orig_features.shape) == 3:  # [B, N, D]
                    orig_features = orig_features[:, 0] if orig_features.shape[1] > 0 else orig_features.mean(dim=1)
                elif len(orig_features.shape) == 2:  # [B, D]
                    orig_features = orig_features
                else:
                    raise ValueError(f"Unexpected orig_features shape: {orig_features.shape}")
                
                # Process feature_grad (same processing as features)
                if len(feature_grad.shape) == 4:  # [B, C, H, W]
                    feature_grad = F.adaptive_avg_pool2d(feature_grad, (1, 1))
                    feature_grad = feature_grad.flatten(1)  # [B, C]
                elif len(feature_grad.shape) == 3:  # [B, N, D]
                    feature_grad = feature_grad[:, 0] if feature_grad.shape[1] > 0 else feature_grad.mean(dim=1)
                elif len(feature_grad.shape) == 2:  # [B, D]
                    feature_grad = feature_grad
                else:
                    raise ValueError(f"Unexpected feature_grad shape: {feature_grad.shape}")
                
                # Detach to free memory (after processing)
                orig_features = orig_features.detach()
                feature_grad = feature_grad.detach()
                # Normalize gradient direction
                feature_grad_norm = F.normalize(feature_grad, p=2, dim=1, eps=1e-10)
                feature_diff = adv_features - orig_features
                
                projection = torch.sum(feature_diff * feature_grad_norm, dim=1)
                
                # We want to maximize the projection (align with gradient direction)
                # So minimize negative projection
                hidden_loss = -torch.mean(projection)
            else:
                hidden_loss = F.cosine_similarity(adv_features, orig_features, dim=1)
                hidden_loss = -torch.mean(hidden_loss)
        if coarse_labels is not None:
            # Total loss
            total_loss = (
                self.lambda_fine * task_loss +
                self.lambda_coarse * concept_loss +
                self.lambda_sim * hidden_loss
            )
        
            return total_loss, task_loss, concept_loss, hidden_loss
        else:
            total_loss = (
                self.lambda_fine * task_loss +
                self.lambda_sim * hidden_loss
            )
            return total_loss, task_loss, None, hidden_loss
    

    @abstractmethod
    def attack(self,
               data: torch.Tensor,
               fine_label: torch.Tensor,
               coarse_label: torch.Tensor = None,
               idx: int = -1) -> torch.Tensor:
        """
        Attack method
        :param data: input images
        :param fine_label: fine-grained labels
        :param coarse_label: coarse-grained labels (if None, will be computed from fine_label)
        :param idx: model index (for compatibility)
        :return: adversarial examples
        """
        ...

    def __call__(self, data, fine_label, coarse_label=None, idx=-1):
        return self.attack(data, fine_label, coarse_label, idx)