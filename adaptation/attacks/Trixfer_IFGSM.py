"""
Trixfer base on I-FGSM
"""
import torch
from adaptation.attacks.Trixfer_Base import Trixfer_Base


class Trixfer_IFGSM(Trixfer_Base):
    def __init__(self, eps=8/255, alpha=2/255, iters=20, max_value=1., min_value=0., threshold=0.,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), beta=10,
                 fine_model=None, coarse_model=None, 
                 lambda_fine=1.0, lambda_coarse=1.0, lambda_sim=1.0,
                 model_name=None,layer_level=0.33):
        super().__init__(eps=eps, alpha=alpha, max_value=max_value, min_value=min_value,
                         threshold=threshold, device=device, beta=beta,
                         fine_model=fine_model, coarse_model=coarse_model,
                         lambda_fine=lambda_fine, lambda_coarse=lambda_coarse, lambda_sim=lambda_sim,
                         model_name=model_name,layer_level=layer_level)
        self.iters = iters

    def attack(self, data, fine_label, coarse_label=None, idx=-1):
        """
        Attack with hybrid loss (iterative FGSM)
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
        
        for i in range(self.iters):
            adv_data.requires_grad = True
            
            # Compute hybrid loss
            total_loss, _, _, _ = self.compute_hybrid_loss(
                ori_data=data, 
                adv_data=adv_data, 
                fine_labels=fine_label, 
                coarse_labels=coarse_label
            )
            
            # Compute gradient
            grad = torch.autograd.grad(total_loss, adv_data, retain_graph=False)[0]
            
            # Add perturbation
            adv_data = self.get_adv_example(ori_data=data, adv_data=adv_data, grad=grad)
            adv_data.detach_()
        
        return adv_data