"""
Learnable archetype embeddings with label dropout for CFG-ready conditioning.

This module implements the conditioning strategy described in the design:
- Learnable archetype embeddings E ∈ R^{K×d_c} for K archetypes
- Label dropout with probability p (e.g., 10-20%) to enable CFG
- Learned null token e_∅ for unconditional training
- CFG sampling: ε̂ = ε_θ(x_t, t, ∅) + w(ε_θ(x_t, t, e) - ε_θ(x_t, t, ∅))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class ArchetypeConditioning(nn.Module):
    """
    Learnable archetype embeddings with label dropout for CFG-ready conditioning.
    
    Args:
        num_archetypes (int): Number of archetypes K
        condition_dim (int): Dimension of condition embeddings d_c
        dropout_prob (float): Probability of dropping labels during training (default: 0.15)
        device (str): Device to use for computations
    """
    
    def __init__(
        self, 
        num_archetypes: int, 
        condition_dim: int, 
        dropout_prob: float = 0.15,
        device: str = "cuda"
    ):
        super().__init__()
        self.num_archetypes = num_archetypes
        self.condition_dim = condition_dim
        self.dropout_prob = dropout_prob
        self.device = device
        
        # Learnable archetype embeddings E ∈ R^{K×d_c}
        self.archetype_embeddings = nn.Embedding(
            num_archetypes, 
            condition_dim,
            device=device
        )
        
        # Learned null token e_∅ for unconditional training
        self.null_token = nn.Parameter(
            torch.randn(condition_dim, device=device) * 0.02
        )
        
        # Initialize archetype embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize archetype embeddings with small random values."""
        nn.init.normal_(self.archetype_embeddings.weight, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)
    
    def forward(
        self, 
        archetype_labels: torch.Tensor, 
        training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass for archetype conditioning.
        
        Args:
            archetype_labels: Archetype labels y ∈ {1, ..., K} of shape (batch_size,)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Condition embeddings of shape (batch_size, condition_dim)
        """
        batch_size = archetype_labels.shape[0]
        
        if training and self.dropout_prob > 0:
            # Apply label dropout during training
            dropout_mask = torch.rand(batch_size, device=self.device) > self.dropout_prob
            
            # Create labels with null tokens where dropout occurs
            effective_labels = archetype_labels.clone()
            effective_labels[~dropout_mask] = -1  # Use -1 to indicate null token
            
            # Get embeddings
            condition_embeddings = self.archetype_embeddings(effective_labels)
            
            # Replace -1 embeddings with null token
            null_mask = (effective_labels == -1)
            condition_embeddings[null_mask] = self.null_token.unsqueeze(0).expand(
                null_mask.sum().item(), -1
            )
        else:
            # No dropout during inference
            condition_embeddings = self.archetype_embeddings(archetype_labels)
        
        return condition_embeddings
    
    def get_unconditional_condition(self, batch_size: int) -> torch.Tensor:
        """
        Get unconditional condition (null token) for CFG.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            Null token embeddings of shape (batch_size, condition_dim)
        """
        return self.null_token.unsqueeze(0).expand(batch_size, -1)
    
    def get_archetype_condition(self, archetype_labels: torch.Tensor) -> torch.Tensor:
        """
        Get archetype condition without dropout (for CFG).
        
        Args:
            archetype_labels: Archetype labels y ∈ {1, ..., K}
            
        Returns:
            Archetype embeddings of shape (batch_size, condition_dim)
        """
        return self.archetype_embeddings(archetype_labels)
    
    def apply_cfg(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        model: nn.Module,
        archetype_labels: torch.Tensor,
        cfg_scale: float = 1.5
    ) -> torch.Tensor:
        """
        Apply Classifier-Free Guidance (CFG) during sampling.
        
        Args:
            x: Noisy input x_t
            t: Timestep t
            model: The denoising model
            archetype_labels: Target archetype labels
            cfg_scale: CFG scale w (typically 1.5-3.0)
            
        Returns:
            CFG-guided prediction ε̂
        """
        # Get unconditional prediction
        null_condition = self.get_unconditional_condition(x.shape[0])
        eps_uncond = model(x, t, null_condition)
        
        # Get conditional prediction
        archetype_condition = self.get_archetype_condition(archetype_labels)
        eps_cond = model(x, t, archetype_condition)
        
        # Apply CFG: ε̂ = ε_θ(x_t, t, ∅) + w(ε_θ(x_t, t, e) - ε_θ(x_t, t, ∅))
        eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        
        return eps_cfg
    
    def get_archetype_embeddings(self) -> torch.Tensor:
        """Get all archetype embeddings for analysis."""
        return self.archetype_embeddings.weight
    
    def get_null_token(self) -> torch.Tensor:
        """Get the learned null token."""
        return self.null_token


class ArchetypeConditioningConfig:
    """Configuration class for archetype conditioning."""
    
    def __init__(
        self,
        num_archetypes: int,
        condition_dim: int = 64,
        dropout_prob: float = 0.15,
        cfg_scale: float = 1.5,
        device: str = "cuda"
    ):
        self.num_archetypes = num_archetypes
        self.condition_dim = condition_dim
        self.dropout_prob = dropout_prob
        self.cfg_scale = cfg_scale
        self.device = device
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "num_archetypes": self.num_archetypes,
            "condition_dim": self.condition_dim,
            "dropout_prob": self.dropout_prob,
            "cfg_scale": self.cfg_scale,
            "device": self.device
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ArchetypeConditioningConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


def create_archetype_conditioning(
    config: Union[ArchetypeConditioningConfig, dict]
) -> ArchetypeConditioning:
    """
    Factory function to create archetype conditioning.
    
    Args:
        config: Configuration for archetype conditioning
        
    Returns:
        ArchetypeConditioning instance
    """
    if isinstance(config, dict):
        config = ArchetypeConditioningConfig.from_dict(config)
    
    return ArchetypeConditioning(
        num_archetypes=config.num_archetypes,
        condition_dim=config.condition_dim,
        dropout_prob=config.dropout_prob,
        device=config.device
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the archetype conditioning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create conditioning
    config = ArchetypeConditioningConfig(
        num_archetypes=5,
        condition_dim=64,
        dropout_prob=0.15,
        device=device
    )
    
    conditioning = create_archetype_conditioning(config)
    
    # Test forward pass
    batch_size = 8
    archetype_labels = torch.randint(0, 5, (batch_size,), device=device)
    
    # Training mode
    conditioning.train()
    condition_embeddings = conditioning(archetype_labels, training=True)
    print(f"Training mode - condition embeddings shape: {condition_embeddings.shape}")
    
    # Inference mode
    conditioning.eval()
    condition_embeddings = conditioning(archetype_labels, training=False)
    print(f"Inference mode - condition embeddings shape: {condition_embeddings.shape}")
    
    # Test CFG components
    null_condition = conditioning.get_unconditional_condition(batch_size)
    archetype_condition = conditioning.get_archetype_condition(archetype_labels)
    
    print(f"Null condition shape: {null_condition.shape}")
    print(f"Archetype condition shape: {archetype_condition.shape}")
    
    print("Archetype conditioning test completed successfully!")
