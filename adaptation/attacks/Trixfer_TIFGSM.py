"""
Trixfer base on TI-FGSM
"""
import torch
import torch.nn.functional as F
from adaptation.attacks.Trixfer_Base import Trixfer_Base

import numpy as np
from scipy import stats as st


class Trixfer_TIFGSM(Trixfer_Base):
    def __init__(self, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), beta=10,
                 kernel_name='gaussian', len_kernel=15, nsig=3, resize_rate=0.9, diversity_prob=0.5,
                 decay=0.0,
                 fine_model=None, coarse_model=None, 
                 lambda_fine=1.0, lambda_coarse=1.0, lambda_sim=1.0,
                 model_name=None,layer_level=0.33):
        super().__init__(eps=eps, alpha=alpha, max_value=max_value, min_value=min_value,
                         threshold=threshold, device=device, beta=beta,
                         fine_model=fine_model, coarse_model=coarse_model,
                         lambda_fine=lambda_fine, lambda_coarse=lambda_coarse, lambda_sim=lambda_sim,
                         model_name=model_name,layer_level=layer_level)
        self.iters = iters
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

    def attack(self, data, fine_label, coarse_label=None, idx=-1):
        """
        Attack with hybrid loss (Translation-Invariant FGSM)
        """
        data = data.clone().detach().to(self.device)
        fine_label = fine_label.clone().detach().to(self.device)
        
        # Get coarse labels if not provided
        if coarse_label is None:
            pass
        else:
            coarse_label = coarse_label.clone().detach().to(self.device)
        
        # init pert
        adv_data = data.clone().detach() + 0.001 * torch.randn(data.shape, device=self.device)
        adv_data = adv_data.detach()
        
        momentum = torch.zeros_like(data, device=self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)
        
        for i in range(self.iters):
            adv_data.requires_grad = True
            
            # Apply input diversity
            adv_data_diverse = self.input_diversity(adv_data)
            
            # Compute hybrid loss on diverse input
            total_loss, _, _, _ = self.compute_hybrid_loss(
                ori_data=data, 
                adv_data=adv_data_diverse, 
                fine_labels=fine_label, 
                coarse_labels=coarse_label
            )
            
            # Compute gradient
            grad = torch.autograd.grad(total_loss, adv_data, retain_graph=False)[0]
            
            # Translation-Invariant: convolve gradient with kernel
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding='same', groups=3)
            
            # Normalize gradient
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            
            # Momentum (if decay > 0)
            if self.decay > 0:
                grad = grad + momentum * self.decay
                momentum = grad
            
            # Add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()
        
        return adv_data

    def kernel_generation(self):
        """Generate translation-invariant kernel"""
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError(f"Kernel {self.kernel_name} not implemented")

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        """Returns a 2D uniform kernel array."""
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        """Returns a 2D linear kernel array."""
        kern1d = 1 - np.abs(np.linspace((-kernlen + 1) / 2,
                                        (kernlen - 1) / 2, kernlen) / (kernlen + 1) * 2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        """Apply input diversity transformation"""
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