"""
Training package for neural network router models.

Provides:
- nn_router: Core neural network router implementation
- model_router: Production-ready ModelRouter class
- generate_hf_embeddings: HuggingFace embedding generation
"""

from .nn_router import load_router, route_embeddings, RouterNetwork
from .model_router import ModelRouter

__all__ = [
    'load_router',
    'route_embeddings',
    'RouterNetwork',
    'ModelRouter',
]

