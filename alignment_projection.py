import math
import torch
from torch import nn
import torch.nn.functional as F

# Alignment projection from DiT model to DINOv2 features
# Align noise of layers to features in DINOv2 
# 
# Record the foward of the model and listen for activation outputs.
# Use those activation outputs as input in the AlignmentProjection

def register_dit_feature_hooks(
    dit_model_blocks: list[nn.Module], layers
) -> tuple[list[torch.utils.hooks.RemovableHandle], dict[str, torch.Tensor]]:
    """
    Register hooks to extract features from specific DiT layers

    Args:
        dit_model_blocks: The DiT model blocks
        layers: List of layer indices to extract features from

    Returns:
        hooks: List of hook handles
        features_dict: Dictionary to store features
    """
    features_dict = {}
    hooks = []

    def get_features(name):
        def hook(module, input, output):
            features_dict[name] = output

        return hook

    # Register hooks for each specified layer
    for layer_idx in layers:
        hook = dit_model_blocks[layer_idx].register_forward_hook(
            get_features(f"layer_{layer_idx}")
        )
        hooks.append(hook)

    return hooks, features_dict


class AlignmentProjection(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        hidden_dim=2048,
        input_dim=1152,
        output_dim=1024,
        max_layers=36,
    ):
        super().__init__()

        # Dimensionality
        self.input_dim = input_dim  # Example for DiT-XL/2
        self.output_dim = output_dim  # Example for DINOv2

        # Add layer embedding
        self.layer_embed = nn.Embedding(max_layers, embedding_dim)

        # Combined embedding processing
        self.combined_embed = nn.Sequential(
            nn.Linear(
                embedding_dim * 2, hidden_dim
            ),  # *2 because we combine timestep and layer
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Sinusoidal position embedding for timestep
        self.embedding_dim = embedding_dim

        # Projection MLP with timestep conditioning
        self.proj_1 = nn.Linear(self.input_dim, hidden_dim)
        self.proj_2 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_3 = nn.Linear(hidden_dim, self.output_dim)

        # Time projection layers for adaptive feature mapping
        self.time_proj_1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj_2 = nn.Linear(hidden_dim, hidden_dim)

    def get_timestep_embedding(self, timesteps):
        """Create sinusoidal timestep embeddings"""
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, dit_features, t, layer_idx):
        """
        Forward pass to get projected features

        Args:
            dit_features: Features extracted from DiT
            t: Timestep values (B,)

        Returns:
            projected_features: DiT features projected into DINO space
        """
        # Get timestep embeddings
        t_emb = self.get_timestep_embedding(t)

        # Get layer embeddings
        l_emb = self.layer_embed(layer_idx)

        # Combine embeddings
        combined_emb = torch.cat([t_emb, l_emb], dim=1)
        combined_emb = self.combined_embed(combined_emb)

        # First projection with timestep modulation
        h = self.proj_1(dit_features)
        time_scale_1 = self.time_proj_1(combined_emb)[:, None, :]  # [B, 1, D]
        time_shift_1 = combined_emb[:, None, :]  # [B, 1, D]
        h = h * (1 + time_scale_1) + time_shift_1
        h = F.silu(h)

        # Second projection with timestep modulation
        h = self.proj_2(h)
        time_scale_2 = self.time_proj_2(combined_emb)[:, None, :]
        h = h * (1 + time_scale_2) + time_shift_1
        h = F.silu(h)

        # Final projection
        projected_features = self.proj_3(h)

        return projected_features


def extract_dino_features(dino_model, dino_processor, real_image):
    image = dino_processor(image=real_image)
    return dino_model(image)


def extract_siglip_features(siglip_model, siglip_processor, real_image):
    image = siglip_processor(image=real_image)
    return siglip_model(image)


# Loss computation function
def compute_alignment_loss(projected_features, features):
    """
    Compute alignment loss between projected DiT features and DINO features

    Args:
        projected_features: DiT features projected to DINO space
        features: Features from DINO model

    Returns:
        loss: Alignment loss
    """
    # Normalize features
    proj_feat = F.normalize(projected_features, dim=-1)
    dino_feat = F.normalize(features, dim=-1)

    # Compute similarity
    similarity = (proj_feat * dino_feat).sum(dim=-1).mean()

    # Loss is negative similarity (we want to maximize similarity)
    loss = -similarity

    return loss
