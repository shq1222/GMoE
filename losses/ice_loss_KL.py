import torch
import torch.nn as nn
import torch.nn.functional as F


def ds_combination(logits: torch.Tensor, batch_size: int, num_classes: int, device: torch.device):
    """Compute belief masses and uncertainty using Dempster-Shafer theory.
    
    Args:
        logits: Raw output from model (batch_size, num_classes)
        batch_size: Number of samples in batch
        num_classes: Total number of classes
        device: Device to perform computations on
        
    Returns:
        Tuple containing:
        - b: Belief masses (batch_size, num_classes)
        - u: Uncertainty values (batch_size,)
        - a: Evidence values (batch_size, num_classes)
    """
    softplus = nn.Softplus()
    a = softplus(logits) + 1.0  # Compute evidence
    s = torch.sum(a, dim=1)  # Total evidence per sample
    
    # Normalize evidence to get belief masses
    b = logits / s.view(batch_size, 1).expand_as(logits)
    
    # Compute uncertainty
    u = num_classes / s
    
    return b.to(device), u.to(device), a.to(device)


def kl_divergence(alpha: torch.Tensor, num_classes: int, device: torch.device):
    """Compute KL divergence between Dirichlet distribution and uniform distribution.
    
    Args:
        alpha: Concentration parameters (batch_size, num_classes)
        num_classes: Total number of classes
        device: Device to perform computations on
        
    Returns:
        KL divergence values (batch_size, 1)
    """
    beta = torch.ones((1, num_classes), device=device)  # Uniform Dirichlet
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)
    
    # Compute log normalization terms
    lnB = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(sum_beta)
    
    # Compute digamma terms
    dg0 = torch.digamma(sum_alpha)
    dg1 = torch.digamma(alpha)
    
    # Final KL divergence
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    
    return kl


def ice_loss_KL(
    x: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    annealing_coef: float = 0.01
) -> torch.Tensor:
    """Compute ICE (Integrated Classification and Evidence) loss with KL regularization.
    
    Args:
        x: Input tensor (unused in current implementation)
        logits: Model predictions (batch_size, num_classes)
        y: Ground truth labels (batch_size,)
        num_classes: Total number of classes
        device: Device to perform computations on
        annealing_coef: Weight for KL regularization term
        
    Returns:
        Computed loss value
    """
    batch_size = x.size(0)
    
    # Get belief masses, uncertainty, and evidence
    b, u, a = ds_combination(logits, batch_size, num_classes, device)
    
    # Compute classification term
    sum_a = torch.sum(a, dim=1, keepdim=True)
    one_hot_labels = F.one_hot(y.long(), num_classes=num_classes)
    classification_term = torch.sum(
        one_hot_labels * (torch.digamma(sum_a) - torch.digamma(a)),
        dim=1, keepdim=True
    )
    
    # Compute KL regularization term
    kl_term = annealing_coef * kl_divergence(a, num_classes, device)
    
    # Combine terms and return mean loss
    loss = classification_term + kl_term
    return torch.mean(loss)


