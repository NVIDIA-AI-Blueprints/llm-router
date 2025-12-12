"""
ModelRouter: Production-ready class for routing requests to the best model

This class provides a simple interface for routing text/image queries to the
optimal model (GPT-5, Nemotron, etc) based on a trained neural network router.

Uses CLIP embeddings (512D text + 512D image = 1024D combined) for multimodal routing.

Usage:
    router = ModelRouter()
    model = router.route("What is the capital of France?")
    
    # With images
    model = router.route(
        "What's in this image?",
        images=["data:image/png;base64,iVBORw0KG..."]
    )
    
    # With custom thresholds
    router = ModelRouter(model_thresholds={'gpt-5-chat': 0.65, 'nemotron-nano-12b-v2-vl': 0.55})
    
    # With custom CLIP server
    router = ModelRouter(clip_server="grpc://your-clip-server:51000")
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from clip_client import Client
import warnings
import asyncio
import os
# Import from nn_router
from .nn_router import load_router, route_embeddings


def _blocking_generate_embedding(generate_embedding_fn):
    """
    Wrapper to run blocking code in a worker thread with standard asyncio policy.
    This allows Jina/CLIP to use asyncio.run() without uvloop interference.
    """
    # Set asyncio policy to standard in this thread (not uvloop)
    # This is crucial - uvloop's global policy blocks event loop creation in threads
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    # Now call the blocking function - Jina's asyncio.run() will work
    return generate_embedding_fn()

# CLIP embedding dimensions
CLIP_TEXT_DIM = 512
CLIP_IMAGE_DIM = 512
COMBINED_DIM = CLIP_TEXT_DIM + CLIP_IMAGE_DIM  # 1024

def _resolve_router_path(router_path: str = "router_artifacts/nn_router.pth") -> Path:
    """
    Resolve the router model path supporting both development and packaged environments.
    
    Tries to find the model in the following order:
    1. If absolute path and exists: use as-is
    2. Relative to current working directory
    3. Relative to package installation directory (training subdirectory)
    4. Using Python package resources (for installed packages)
    
    Args:
        router_path: Path to router model (default: "router_artifacts/nn_router.pth")
    
    Returns:
        Resolved Path object to the router model
        
    Raises:
        FileNotFoundError: If model cannot be found in any location
    """
    # Try as absolute path first
    abs_path = Path(router_path)
    if abs_path.is_absolute() and abs_path.exists():
        return abs_path
    
    # Try relative to current working directory
    cwd_path = Path.cwd() / router_path
    if cwd_path.exists():
        return cwd_path
    
    # Try relative to this file's directory (training package)
    package_path = Path(__file__).parent / router_path
    if package_path.exists():
        return package_path
    
    # Try relative to nat_sfc_router package root
    package_root = Path(__file__).parent.parent / router_path
    if package_root.exists():
        return package_root
    
    # Try using importlib.resources (for installed packages)
    try:
        # Python 3.9+
        from importlib.resources import files
        pkg_files = files("nat_sfc_router").joinpath("training", router_path)
        # Check if resource exists
        if pkg_files.is_file():
            # For packaged resources, we need to extract or read through importlib
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                extracted_path = tmpdir_path / Path(router_path).name
                # Copy the resource to temp location
                with pkg_files.open('rb') as f:
                    extracted_path.write_bytes(f.read())
                return extracted_path
    except (ImportError, AttributeError, FileNotFoundError):
        pass
    
    # If we get here, we couldn't find the file
    search_paths = [
        abs_path,
        cwd_path,
        package_path,
        package_root,
    ]
    
    raise FileNotFoundError(
        f"Router model not found at '{router_path}'.\n\n"
        f"Searched locations:\n"
        + "\n".join(f"  - {p}" for p in search_paths) +
        f"\n\nPlease ensure the model file exists or train it first using: python nn_router.py"
    )



class ModelRouter:
    """
    Production-ready router for selecting the best model for a given query.
    
    This class combines:
    1. CLIP for embedding generation (512D text + 512D image = 1024D combined)
    2. Trained neural network router for model selection
    3. Configurable thresholds for cost optimization
    
    Attributes:
        clip_client: The CLIP client for embedding generation
        router_model: The trained neural network router
        model_names: List of available models
        model_thresholds: Dict of model -> minimum confidence threshold
        device: PyTorch device (cuda/cpu)
    """
    
    # Recommended threshold configurations
    THRESHOLD_CONFIGS = {
        'conservative': {
            'description': 'Minimal cost savings, maximum quality',
            'thresholds': {'gpt-5-chat': 0.60},
            'estimated_savings': '~7%'
        },
        'balanced': {
            'description': 'Balanced cost savings and quality',
            'thresholds': {'gpt-5-chat': 0.65, 'nemotron-nano-12b-v2-vl': 0.55},
            'estimated_savings': '~12%'
        },
        'aggressive': {
            'description': 'Maximum cost savings, monitor quality',
            'thresholds': {'gpt-5-chat': 0.70, 'nemotron-nano-12b-v2-vl': 0.55},
            'estimated_savings': '~16%'
        },
        'none': {
            'description': 'No thresholds, pure router predictions',
            'thresholds': {},
            'estimated_savings': 'baseline'
        }
    }
    
    def __init__(
        self,
        router_path: str = "router_artifacts/nn_router.pth",
        clip_server: str = "grpc://0.0.0.0:51000",
        model_thresholds: Optional[Dict[str, float]] = None,
        threshold_config: str = 'balanced',
        device: Optional[str] = None,
        verbose: bool = True,
        cost_aware: bool = True,
        high_confidence_threshold: float = 0.80
    ):
        """
        Initialize the ModelRouter.
        
        Args:
            router_path: Path to the trained router model (.pth file)
            clip_server: CLIP server address (gRPC endpoint)
            model_thresholds: Custom threshold dict (overrides threshold_config if provided)
            threshold_config: Preset configuration name ('conservative', 'balanced', 'aggressive', 'none')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            verbose: Print initialization messages
            cost_aware: If True, prefer cheaper models when they have high confidence (default True)
            high_confidence_threshold: Confidence level above which cheaper models are preferred (default 0.80)
        """
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cost_aware = cost_aware
        self.high_confidence_threshold = high_confidence_threshold
        
        if self.verbose:
            print("="*80)
            print("Initializing ModelRouter")
            print("="*80)
            print(f"Device: {self.device}")
        
        
        resolved_path = _resolve_router_path(router_path)
        # Load router model
        if self.verbose:
            print(f"\n1. Loading neural network router from {str(resolved_path)}...")

        self.router_model, self.model_names, self.router_config = load_router(str(resolved_path))
        
        if self.verbose:
            print(f"   ✓ Loaded router (models: {', '.join(self.model_names)})")
        
        clip_server = os.getenv("CLIP_SERVER", clip_server)
        # Connect to CLIP server
        if self.verbose:
            print(f"\n2. Connecting to CLIP server ({clip_server})...")
        
        self.clip_client = Client(clip_server)
        
        if self.verbose:
            print(f"   ✓ Connected to CLIP server")
            print(f"   ✓ Embedding dimensions: text={CLIP_TEXT_DIM}, image={CLIP_IMAGE_DIM}, combined={COMBINED_DIM}")
        
        # Set thresholds
        if model_thresholds is not None:
            self.model_thresholds = model_thresholds
            self.threshold_config_name = 'custom'
        elif threshold_config in self.THRESHOLD_CONFIGS:
            self.model_thresholds = self.THRESHOLD_CONFIGS[threshold_config]['thresholds']
            self.threshold_config_name = threshold_config
        else:
            warnings.warn(f"Unknown threshold config '{threshold_config}', using 'balanced'")
            self.model_thresholds = self.THRESHOLD_CONFIGS['balanced']['thresholds']
            self.threshold_config_name = 'balanced'
        
        if self.verbose:
            print(f"\n3. Threshold configuration: {self.threshold_config_name}")
            if self.model_thresholds:
                for model, threshold in self.model_thresholds.items():
                    print(f"   - {model}: {threshold:.2f}")
                if self.threshold_config_name in self.THRESHOLD_CONFIGS:
                    config_info = self.THRESHOLD_CONFIGS[self.threshold_config_name]
                    print(f"   Expected savings: {config_info['estimated_savings']}")
            else:
                print(f"   - No thresholds (baseline)")
            
            print(f"\n4. Cost-aware routing: {'ENABLED' if self.cost_aware else 'DISABLED'}")
            if self.cost_aware:
                print(f"   - High confidence threshold: {self.high_confidence_threshold:.2f}")
                print(f"   - When cheaper models exceed {self.high_confidence_threshold:.2f}, prefer them")
            
            print("\n" + "="*80)
            print("ModelRouter ready!")
            print("="*80)
    
    def generate_embedding(
        self,
        text: str,
        images: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate embedding for text and optional images using CLIP.
        
        Returns a 1024D vector:
        - If text + image: [text_embedding(512) | image_embedding(512)]
        - If text only: [text_embedding(512) | zeros(512)]
        
        Args:
            text: The input text/prompt
            images: Optional list of base64-encoded images
                   Format: "data:image/png;base64,iVBORw0KG..." or just the base64 string
        
        Returns:
            numpy array of shape (1024,)
        """
        # Normalize images to proper format
        if images is None:
            images = []
        else:
            # Ensure proper format for images
            formatted_images = []
            for img in images:
                if img is None:
                    continue
                # Add data URI prefix if not present
                if not img.startswith('data:image'):
                    img = f"data:image/png;base64,{img}"
                formatted_images.append(img)
            images = formatted_images
        
        # Generate CLIP embeddings
        if len(images) > 0:
            # Has both text and image - concatenate text and image embeddings
            # Encode text and image separately
            text_embedding = self.clip_client.encode([text])  # Shape: (1, 512)
            image_embedding = self.clip_client.encode([images[0]])  # Shape: (1, 512), use first image
            
            # Concatenate text and image embeddings
            combined_embedding = np.concatenate([
                text_embedding[0],  # (512,)
                image_embedding[0]  # (512,)
            ])  # Result: (1024,)
        else:
            # Text only - pad with zeros
            text_embedding = self.clip_client.encode([text])  # Shape: (1, 512)
            
            # Pad with 512 zeros
            padding = np.zeros(CLIP_IMAGE_DIM)
            combined_embedding = np.concatenate([
                text_embedding[0],  # (512,)
                padding  # (512,)
            ])  # Result: (1024,)
        
        return combined_embedding
    
    async def generate_embedding_async(
        self,
        text: str,
        images: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Async version of generate_embedding that runs blocking CLIP calls in a dedicated thread.
        
        This method is safe to call from async contexts without event loop conflicts.
        Properly handles uvloop by resetting asyncio policy in the worker thread.
        
        Returns a 1024D vector (same as generate_embedding):
        - If text + image: [text_embedding(512) | image_embedding(512)]
        - If text only: [text_embedding(512) | zeros(512)]
        
        Args:
            text: The input text/prompt
            images: Optional list of base64-encoded images
        
        Returns:
            numpy array of shape (1024,)
        """
        # Use asyncio.to_thread() to run blocking operation in a separate thread
        # The wrapper function resets asyncio policy to standard (not uvloop)
        # This allows Jina to call asyncio.run() without interference
        embedding = await asyncio.to_thread(
            _blocking_generate_embedding,
            lambda: self.generate_embedding(text, images)
        )
        
        return embedding
    
    def route(
        self,
        text: str,
        images: Optional[List[str]] = None,
        return_probabilities: bool = False,
        return_all_info: bool = False
    ) -> str:
        """
        Route a query to the best model.
        
        Args:
            text: The input text/prompt
            images: Optional list of base64-encoded images
            return_probabilities: If True, return (model_name, probabilities_dict)
            return_all_info: If True, return detailed routing info
        
        Returns:
            model_name: Name of the selected model (str)
            OR (model_name, probabilities): if return_probabilities=True
            OR routing_info: if return_all_info=True (dict with all details)
        
        Examples:
            >>> router = ModelRouter()
            >>> model = router.route("What is 2+2?")
            >>> print(model)  # 'meta/llama-3.2-1b-instruct'
            
            >>> model, probs = router.route("Complex reasoning task", return_probabilities=True)
            >>> print(f"Chose {model}, confidence: {probs[model]:.3f}")
        """
        # Generate embedding
        embedding = self.generate_embedding(text, images)
        
        # Route using neural network
        embedding_2d = embedding.reshape(1, -1)  # Add batch dimension
        
        if return_probabilities or return_all_info:
            chosen_models, probabilities = route_embeddings(
                embedding_2d,
                self.router_model,
                self.model_names,
                return_probs=True,
                model_thresholds=self.model_thresholds
            )
            chosen_model = chosen_models[0]
            probs_dict = {
                model: float(probabilities[0, i])
                for i, model in enumerate(self.model_names)
            }
            
            if return_all_info:
                # Get original choice (without thresholds)
                original_chosen, _ = route_embeddings(
                    embedding_2d,
                    self.router_model,
                    self.model_names,
                    return_probs=True,
                    model_thresholds=None
                )
                
                routing_info = {
                    'chosen_model': chosen_model,
                    'probabilities': probs_dict,
                    'original_choice': original_chosen[0],
                    'threshold_applied': chosen_model != original_chosen[0],
                    'thresholds': self.model_thresholds,
                    'threshold_config': self.threshold_config_name,
                    'has_images': images is not None and len(images) > 0
                }
                return routing_info
            
            if return_probabilities:
                return chosen_model, probs_dict
        else:
            chosen_models = route_embeddings(
                embedding_2d,
                self.router_model,
                self.model_names,
                return_probs=False,
                model_thresholds=self.model_thresholds
            )
            chosen_model = chosen_models[0]
        
        return chosen_model
    
    def route_batch(
        self,
        texts: List[str],
        images_list: Optional[List[List[str]]] = None,
        return_probabilities: bool = False
    ) -> List[str]:
        """
        Route a batch of queries to models.
        
        Args:
            texts: List of input texts
            images_list: Optional list of image lists (one per text)
            return_probabilities: If True, return list of (model, probs) tuples
        
        Returns:
            List of model names (or list of (model, probs) if return_probabilities=True)
        
        Examples:
            >>> router = ModelRouter()
            >>> texts = ["What is 2+2?", "Explain quantum physics"]
            >>> models = router.route_batch(texts)
            >>> print(models)  # ['meta/llama-3.2-1b-instruct', 'gpt-5-chat']
        """
        if images_list is None:
            images_list = [None] * len(texts)
        
        results = []
        for text, images in zip(texts, images_list):
            if return_probabilities:
                model, probs = self.route(text, images, return_probabilities=True)
                results.append((model, probs))
            else:
                model = self.route(text, images)
                results.append(model)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about available models and current configuration.
        
        Returns:
            Dict with model information
        """
        return {
            'available_models': self.model_names,
            'threshold_config': self.threshold_config_name,
            'thresholds': self.model_thresholds,
            'device': self.device,
            'router_config': self.router_config,
            'available_threshold_configs': list(self.THRESHOLD_CONFIGS.keys())
        }
    
    def print_routing_stats(self, routing_results: List[str]):
        """
        Print statistics about routing results.
        
        Args:
            routing_results: List of model names from routing
        """
        from collections import Counter
        
        counts = Counter(routing_results)
        total = len(routing_results)
        
        print("\nRouting Statistics:")
        print("="*60)
        print(f"{'Model':<45} {'Count':>7} {'%':>6}")
        print("-"*60)
        
        for model in self.model_names:
            count = counts.get(model, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"{model:<45} {count:>7} {pct:>5.1f}%")
        
        print("-"*60)
        print(f"{'TOTAL':<45} {total:>7} {100.0:>5.1f}%")
        print("="*60)


# Convenience function for quick usage
def route_query(
    text: str,
    images: Optional[List[str]] = None,
    threshold_config: str = 'balanced',
    clip_server: str = "grpc://0.0.0.0:51000",
    cost_aware: bool = True,
    high_confidence_threshold: float = 0.80
) -> str:
    """
    Quick utility function to route a single query without creating a router instance.
    
    Note: This creates a new router each time, so for multiple queries use the
    ModelRouter class directly.
    
    Args:
        text: Input text
        images: Optional list of base64-encoded images
        threshold_config: Threshold configuration ('conservative', 'balanced', 'aggressive', 'none')
        clip_server: CLIP server address (gRPC endpoint)
        cost_aware: If True, prefer cheaper models when they have high confidence (default True)
        high_confidence_threshold: Confidence level above which cheaper models are preferred (default 0.80)
    
    Returns:
        Name of the selected model
    
    Example:
        >>> model = route_query("What is machine learning?")
        >>> print(model)  # 'gpt-5-chat'
    """
    router = ModelRouter(
        threshold_config=threshold_config, 
        clip_server=os.getenv("CLIP_SERVER"), 
        verbose=False,
        cost_aware=cost_aware,
        high_confidence_threshold=high_confidence_threshold
    )
    return router.route(text, images)


if __name__ == "__main__":
    # Example usage and demonstration
    print("ModelRouter Demo")
    print("="*80)
    
    # Initialize router with balanced configuration
    router = ModelRouter(threshold_config='balanced')
    
    # Example 1: Simple text query
    print("\n" + "="*80)
    print("Example 1: Simple Math Question")
    print("="*80)
    query1 = "What is 2 + 2?"
    model1, probs1 = router.route(query1, return_probabilities=True)
    print(f"Query: {query1}")
    print(f"Chosen model: {model1}")
    print(f"Probabilities:")
    for m, p in sorted(probs1.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m}: {p:.3f}")
    
    # Example 2: Complex reasoning query
    print("\n" + "="*80)
    print("Example 2: Complex Reasoning")
    print("="*80)
    query2 = "Explain the implications of quantum entanglement for faster-than-light communication"
    model2, probs2 = router.route(query2, return_probabilities=True)
    print(f"Query: {query2}")
    print(f"Chosen model: {model2}")
    print(f"Probabilities:")
    for m, p in sorted(probs2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m}: {p:.3f}")
    
    # Example 3: Batch routing
    print("\n" + "="*80)
    print("Example 3: Batch Routing")
    print("="*80)
    queries = [
        "What's the weather like?",
        "Solve this differential equation: dy/dx = x^2",
        "Hello!",
        "Explain general relativity",
        "What is 5 * 7?"
    ]
    models = router.route_batch(queries)
    print(f"Processed {len(queries)} queries:")
    for q, m, p in zip(queries, models, probs):
        print(f"  '{q[:50]}...' -> {m} (confidence: {p:.3f})")
    
    # Print routing statistics
    router.print_routing_stats(models)
    
    # Example 4: Detailed routing info
    print("\n" + "="*80)
    print("Example 4: Detailed Routing Info")
    print("="*80)
    info = router.route("Write a haiku about programming", return_all_info=True)
    print(f"Chosen model: {info['chosen_model']}")
    print(f"Original choice: {info['original_choice']}")
    print(f"Threshold applied: {info['threshold_applied']}")
    print(f"Configuration: {info['threshold_config']}")
    print(f"Probabilities: {info['probabilities']}")
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)

