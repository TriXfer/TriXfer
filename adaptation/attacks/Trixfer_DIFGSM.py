"""
Trixfer base on DI-FGSM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptation.attacks.Trixfer_Base import Trixfer_Base


class Trixfer_DIFGSM(Trixfer_Base):
    def __init__(self, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), beta=10, momentum=0.9,
                 resize_rate=0.9, diversity_prob=0.5,fine_model=None, coarse_model=None, 
                 lambda_fine=1.0, lambda_coarse=1.0, lambda_sim=1.0,
                 model_name=None,layer_level=0.33):
        super().__init__(eps=eps, alpha=alpha, max_value=max_value, min_value=min_value,
                         threshold=threshold, device=device, beta=beta,
                         fine_model=fine_model, coarse_model=coarse_model,
                         lambda_fine=lambda_fine, lambda_coarse=lambda_coarse, lambda_sim=lambda_sim,
                         model_name=model_name,layer_level=layer_level)
        self.alpha = alpha
        self.iters = iters
        self.momentum = momentum
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def attack(self, data, fine_label, coarse_label=None):
        B, C, H, W = data.size()
        data, fine_label = data.clone().detach().to(self.device), fine_label.clone().detach().to(self.device)
        if coarse_label is None:
            pass
        else:
            coarse_label = coarse_label.clone().detach().to(self.device)
        

        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()

        grad_mom = torch.zeros_like(data, device=self.device)
        
        for i in range(self.iters):
            adv_data.requires_grad = True
            loss_self,_,_,_ = self.compute_hybrid_loss(ori_data=data, adv_data=adv_data, fine_labels=fine_label, coarse_labels=coarse_label)
            grad = torch.autograd.grad(loss_self, adv_data, retain_graph=True, create_graph=False)[0]

            # Momentum
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + self.momentum * grad_mom
            grad_mom = grad

            # Add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()

        return adv_data

