"""
Neural Network Objective Function for Intelligent Model Routing

This module provides a production-ready objective function that uses a pre-trained
neural network router to intelligently route requests to the best model based on:
1. Embedding generation from text and multimodal content
2. Neural network predictions with configurable confidence thresholds
3. Multi-turn conversation context understanding

The router loads on service startup (before request handling) for optimal performance.
"""

import json
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging
from functools import lru_cache
import time

from pydantic import Field
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# Import ModelRouter from training package
from nat_sfc_router.training import ModelRouter
from nat_sfc_router.training.model_router import _resolve_router_path

# Import OpenAI schema for type hints
from nat_sfc_router.schema.openai_chat_request import OpenAIChatRequest

logger = logging.getLogger(__name__)

# ========== MODEL ROUTING CONFIGURATION ==========

# Map router output model names to actual pipeline model names
# The router predicts one of these model names based on embeddings
MODEL_ROUTER_TO_TARGET = {
    # GPT models -> openai/gpt-oss-120b
    'gpt-5-chat': 'gpt-5-chat',
    'gpt-5': 'gpt-5-chat',
    
    # Nemotron VL models -> nvidia/nemotron-nano-12b-v2-vl
    'nemotron-vl': 'nvidia/nemotron-nano-12b-v2-vl',
    'nemotron-nano-12b-v2-vl': 'nvidia/nemotron-nano-12b-v2-vl',
    'nvidia/nemotron-nano-12b-v2-vl': 'nvidia/nemotron-nano-12b-v2-vl',
    
    # Nemotron models -> nvidia/nvidia-nemotron-nano-9b-v2
    'nemotron-nano-12b-v2-vl': 'nvidia/nvidia-nemotron-nano-9b-v2',
    'nemotron-nano': 'nvidia/nvidia-nemotron-nano-9b-v2',
    'nvidia/nvidia-nemotron-nano-9b-v2': 'nvidia/nvidia-nemotron-nano-9b-v2',
}

# Custom confidence thresholds for model selection
# These thresholds ensure the router only selects a model if confidence is above the threshold
# If the top choice doesn't meet its threshold, the router falls back to the next best model
CUSTOM_THRESHOLDS = {
    'gpt-5-chat': 0.70,
    'nvidia/nemotron-nano-12b-v2-vl': 0.55,
    'nvidia/nvidia-nemotron-nano-9b-v2': 0.50,
}

# Default fallback model (used on errors or unknown routing)
DEFAULT_FALLBACK_MODEL = 'nvidia/nvidia-nemotron-nano-9b-v2'

# ========== GLOBAL ROUTER (loaded on startup) ==========

_router = None


def _load_router():
    """
    Load the ModelRouter on service startup (before _response_fn).
    This ensures the router is initialized once when the service starts,
    not on every request.
    """
    global _router
    if _router is None:
        logger.info("Loading ModelRouter on service startup...")
        try:
            _router = ModelRouter(
                router_path="router_artifacts/nn_router.pth",
                model_thresholds=CUSTOM_THRESHOLDS,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=True
            )
            logger.info("✓ ModelRouter loaded successfully on startup")
        except Exception as e:
            logger.error(f"Failed to load ModelRouter: {e}", exc_info=True)
            raise
    return _router


# ========== MESSAGE PARSING UTILITIES ==========

def extract_text_and_images_from_messages(
    messages: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Extract and format text and images from multi-turn chat messages.
    
    Handles OpenAI API format messages with support for:
    - Multiple turns (user, assistant, system)
    - Text content (strings and lists)
    - Images (data URIs and base64 encoded)
    - Both dict and Pydantic object types
    
    Args:
        messages: List of message dicts or objects with 'role' and 'content'
                 Example: [
                     {"role": "user", "content": "What is this?"},
                     {"role": "assistant", "content": "This is..."},
                     {"role": "user", "content": [
                         {"type": "text", "text": "More details?"},
                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                     ]}
                 ]
    
    Returns:
        Tuple of (combined_text, images_list)
        - combined_text: Full conversation context as a single string
        - images_list: List of image URIs/base64 strings suitable for embedding model
    """
    full_text_parts = []
    images = []
    
    # Convert messages to list in case it's a ValidatorIterator or other iterable
    messages_list = list(messages) if not isinstance(messages, list) else messages
    
    for msg_idx, message in enumerate(messages_list):
        # Handle both dict and object types
        if isinstance(message, dict):
            role = message.get("role", "").upper()
            content = message.get("content", "")
        else:
            # Try to access as object attributes (Pydantic models, etc.)
            role = getattr(message, "role", "").upper()
            content = getattr(message, "content", "")
        
        # Add role prefix for conversation context
        if role:
            full_text_parts.append(f"[{role}]:")
        
        # Handle string content
        if isinstance(content, str):
            if content.strip():
                full_text_parts.append(content)
        
        # Handle list content (multimodal)
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if isinstance(part, dict):
                    # Text items (dict)
                    if part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text:
                            full_text_parts.append(text)
                    
                    # Image items (dict)
                    elif part.get("type") == "image_url":
                        img_url_obj = part.get("image_url", {})
                        
                        # Handle both dict and string formats
                        if isinstance(img_url_obj, dict):
                            url = img_url_obj.get("url", "").strip()
                        else:
                            url = str(img_url_obj).strip()
                        
                        # Only include valid image URLs
                        if url and isinstance(url, str) and len(url) > 0:
                            images.append(url)
                else:
                    # Handle object types (from Pydantic models)
                    part_type = getattr(part, "type", None)
                    
                    if part_type == "text":
                        text_content = getattr(part, "text", "")
                        if text_content:
                            full_text_parts.append(text_content)
                    
                    elif part_type == "image_url":
                        img_url_obj = getattr(part, "image_url", None)
                        if img_url_obj:
                            # Handle as dict or object
                            if isinstance(img_url_obj, dict):
                                url = img_url_obj.get("url", "").strip()
                            else:
                                url = getattr(img_url_obj, "url", "")
                                if url:
                                    url = url.strip()
                            
                            # Only include valid image URLs
                            if url and isinstance(url, str) and len(url) > 0:
                                images.append(url)
        else:
            # Try to iterate over content in case it's a ValidatorIterator
            try:
                content_list = list(content)
                for part_idx, part in enumerate(content_list):
                    if isinstance(part, dict):
                        # Text items (dict)
                        if part.get("type") == "text":
                            text = part.get("text", "").strip()
                            if text:
                                full_text_parts.append(text)
                        # Image items (dict)
                        elif part.get("type") == "image_url":
                            img_url_obj = part.get("image_url", {})
                            if isinstance(img_url_obj, dict):
                                url = img_url_obj.get("url", "").strip()
                            else:
                                url = str(img_url_obj).strip()
                            if url and isinstance(url, str) and len(url) > 0:
                                images.append(url)
                    else:
                        # Handle object types
                        part_type = getattr(part, "type", None)
                        if part_type == "text":
                            text_content = getattr(part, "text", "")
                            if text_content:
                                full_text_parts.append(text_content)
                        elif part_type == "image_url":
                            img_url_obj = getattr(part, "image_url", None)
                            if img_url_obj:
                                if isinstance(img_url_obj, dict):
                                    url = img_url_obj.get("url", "").strip()
                                else:
                                    url = getattr(img_url_obj, "url", "")
                                    if url:
                                        url = url.strip()
                                if url and isinstance(url, str) and len(url) > 0:
                                    images.append(url)
            except TypeError:
                # Not iterable, log and skip
                logger.debug(
                    f"extract_text_and_images_from_messages - Message {msg_idx}: "
                    f"content is not iterable, skipping"
                )
    
    # Combine all text parts
    full_text = " ".join(full_text_parts).strip()
    
    # Fallback text if empty
    if not full_text:
        logger.warning("No text extracted from messages, using default prompt")
        full_text = "routing query"
    
    logger.info(f"extract_text_and_images_from_messages - Final: text_len={len(full_text)}, images={len(images)}")
    
    return full_text, images


def map_router_model_to_target(router_model: str) -> str:
    """
    Map router output model name to target pipeline model name.
    
    Args:
        router_model: Model name predicted by router (e.g., 'gpt-5-chat')
    
    Returns:
        Target model name for routing (e.g., 'openai/gpt-oss-120b')
    """
    # Direct lookup
    if router_model in MODEL_ROUTER_TO_TARGET:
        return MODEL_ROUTER_TO_TARGET[router_model]
    
    # Case-insensitive lookup
    lower_model = router_model.lower()
    for key, value in MODEL_ROUTER_TO_TARGET.items():
        if key.lower() == lower_model:
            return value
    
    # Partial match (e.g., 'gpt' in the name maps to GPT model)
    if 'gpt' in lower_model:
        logger.warning(f"Router model '{router_model}' not in mapping, assuming GPT model")
        return 'openai/gpt-oss-120b'
    elif 'nemotron' in lower_model and 'vl' in lower_model:
        logger.warning(f"Router model '{router_model}' not in mapping, assuming Nemotron VL model")
        return 'nvidia/nemotron-nano-12b-v2-vl'
    
    # Default fallback
    logger.warning(
        f"Unknown router model '{router_model}', using default fallback: {DEFAULT_FALLBACK_MODEL}"
    )
    return DEFAULT_FALLBACK_MODEL


# ========== COST-BASED ROUTING LOGIC ==========

def select_best_model_by_cost(
    probabilities: Dict[str, float],
    model_thresholds: Optional[Dict[str, float]] = None,
    model_costs: Optional[Dict[str, float]] = None
) -> Tuple[str, str]:
    """
    Select the best model using cost-based routing.
    
    Strategy:
    1. Filter models that meet their confidence threshold
    2. Among qualified models, select the one with lowest cost factor
    3. Falls back to highest probability if no thresholds specified
    
    Args:
        probabilities: Dict of model_name -> confidence score
        model_thresholds: Dict of model_name -> minimum confidence threshold
        model_costs: Dict of model_name -> cost factor (lower is better)
    
    Returns:
        Tuple of (selected_model, selection_reason)
        - selected_model: Name of the model to use
        - selection_reason: Human-readable explanation of selection logic
    """
    if not probabilities:
        raise ValueError("No probabilities provided")
    
    # Filter models that meet their thresholds
    qualified_models = []
    
    if model_thresholds:
        for model_name, prob in probabilities.items():
            threshold = model_thresholds.get(model_name, 0.0)
            if prob >= threshold:
                qualified_models.append((model_name, prob, threshold))
        
        if qualified_models:
            logger.debug(
                f"Models meeting thresholds: "
                f"{[(m, f'{p:.3f}') for m, p, t in qualified_models]}"
            )
        else:
            logger.warning(
                "No models met their thresholds, falling back to highest probability"
            )
            # Fallback: use highest probability if nothing meets threshold
            best_model = max(probabilities.items(), key=lambda x: x[1])
            return best_model[0], "threshold_fallback"
    else:
        # No thresholds - all models are qualified
        qualified_models = [(m, p, 0.0) for m, p in probabilities.items()]
    
    # If cost factors provided, select by lowest cost among qualified models
    if model_costs:
        # Find the qualified model with lowest cost
        best_model_name = None
        best_cost = float('inf')
        best_prob = 0.0
        
        for model_name, prob, threshold in qualified_models:
            cost = model_costs.get(model_name, float('inf'))
            
            # Prefer lower cost, but use probability as tiebreaker
            if cost < best_cost or (cost == best_cost and prob > best_prob):
                best_model_name = model_name
                best_cost = cost
                best_prob = prob
        
        if best_model_name:
            logger.debug(
                f"Cost-based selection: {best_model_name} "
                f"(cost={best_cost:.3f}, prob={best_prob:.3f})"
            )
            return best_model_name, "cost_optimized"
    
    # Default: select highest probability among qualified models
    best_model = max(qualified_models, key=lambda x: x[1])
    return best_model[0], "highest_probability"


# ========== OBJECTIVE FUNCTION REGISTRATION ==========

class NNObjectiveConfig(FunctionBaseConfig, name="nn_objective_fn"):
    """Neural network objective function configuration for model routing.
    
    Attributes:
        model_thresholds: Dict of model_name -> minimum confidence threshold
                         Only routes to models meeting their threshold
        model_costs: Dict of model_name -> cost factor for cost-based selection
                    Among models meeting thresholds, selects lowest cost model
    """
    model_thresholds: Optional[Dict[str, float]] = None
    model_costs: Optional[Dict[str, float]] = None


@register_function(config_type=NNObjectiveConfig)
async def nn_objective_fn(config: NNObjectiveConfig, _builder: Builder):
    """
    Neural network objective function for intelligent cost-optimized model routing.
    
    Uses a pre-trained neural network router with:
    - CLIP embedding model for multimodal content encoding
    - Trained router network to predict best model
    - Confidence thresholds (from config) for quality gates
    - Cost-based selection among qualified models
    - Multi-turn conversation context support
    
    Configuration (from config.yml):
    - model_thresholds: Dict of model -> minimum confidence threshold
    - model_costs: Dict of model -> cost factor (lower = cheaper)
    
    Routing Strategy:
    1. Generate embedding for the query
    2. Get confidence scores from neural router
    3. Filter models meeting their confidence threshold
    4. Select lowest-cost model among qualified options
    
    This function loads the router on service startup for optimal performance.
    """
    
    # Extract configuration
    model_thresholds = config.model_thresholds or {}
    model_costs = config.model_costs or {}
    
    logger.info("nn_objective_fn: Configuration loaded")
    logger.info(f"  Thresholds: {model_thresholds}")
    logger.info(f"  Costs: {model_costs}")
    
    # Load router on startup - this happens BEFORE _response_fn
    # so it only happens once when the service starts
    logger.info("nn_objective_fn: Initializing router...")
    router = _load_router()
    logger.info("nn_objective_fn: Router ready")
    
    async def _response_fn(chat_request: OpenAIChatRequest) -> Tuple[str, Dict[str, float]]:
        """
        Route a chat request to the best model using the neural network router.
        
        Args:
            chat_request: OpenAI-format chat request
        
        Returns:
            Target model name to route the request to
        """
        response_start = time.perf_counter()
        
        try:
            # ===== EXTRACT MESSAGES =====
            extract_start = time.perf_counter()
            messages = chat_request.messages

            logger.info(f"Routing with messages: {messages}")
            
            if not messages:
                logger.warning("No messages received, using default fallback model")
                return DEFAULT_FALLBACK_MODEL
            
            # Convert messages to dicts if needed (handle Pydantic models)
            messages_dict = []
            for msg in messages:
                if hasattr(msg, 'model_dump'):
                    messages_dict.append(msg.model_dump())
                elif hasattr(msg, '__dict__'):
                    messages_dict.append(vars(msg))
                elif isinstance(msg, dict):
                    messages_dict.append(msg)
                else:
                    messages_dict.append(dict(msg))
            
            #logger.info(f"Routing with messages_dict: {messages_dict}")
            
            extract_time = time.perf_counter() - extract_start
            logger.debug(f"Extracted {len(messages_dict)} messages in {extract_time*1000:.2f}ms")
            
            # ===== PARSE TEXT AND IMAGES =====
            parse_start = time.perf_counter()
            full_text, images = extract_text_and_images_from_messages(messages_dict)
            parse_time = time.perf_counter() - parse_start
            
            logger.debug(
                f"Parsed {len(images)} images, "
                f"text: {len(full_text)} chars, "
                f"time: {parse_time*1000:.2f}ms"
            )

            logger.info(f"Routing with full text: {full_text} and number of images: {len(images)}")
            
            # ===== ROUTE USING NEURAL NETWORK =====
            route_start = time.perf_counter()
            
            # Get probabilities from router (without thresholds - raw scores)
            # Use async version to avoid event loop conflicts with CLIP client
            embedding = await router.generate_embedding_async(full_text, images)
            embedding_2d = embedding.reshape(1, -1)
            
            # Get raw probabilities from router
            router.router_model.eval()
            with torch.no_grad():
                proba = router.router_model(torch.FloatTensor(embedding_2d).to(router.device))
                proba = proba.cpu().numpy()
            
            # Build probability dict
            probabilities = {
                model: float(proba[0, i])
                for i, model in enumerate(router.model_names)
            }
            
            route_time = time.perf_counter() - route_start
            
            # ===== COST-BASED SELECTION =====
            cost_select_start = time.perf_counter()
            
            # Use cost-based routing to select model
            router_model, selection_reason = select_best_model_by_cost(
                probabilities=probabilities,
                model_thresholds=model_thresholds,
                model_costs=model_costs
            )
            
            cost_select_time = time.perf_counter() - cost_select_start
            confidence = probabilities.get(router_model, 0.0)
            
            logger.info(
                f"Routing decision | "
                f"Model: {router_model} | "
                f"Confidence: {confidence:.3f} | "
                f"Selection: {selection_reason} | "
                f"Probabilities: {{{', '.join(f'{m}: {p:.3f}' for m, p in probabilities.items())}}} | "
                f"Route+Select time: {route_time*1000 + cost_select_time*1000:.2f}ms"
            )
            
            # ===== MAP TO TARGET MODEL =====
            map_start = time.perf_counter()
            target_model = map_router_model_to_target(router_model)
            map_time = time.perf_counter() - map_start
            
            total_time = time.perf_counter() - response_start
            
            logger.info(
                f"Final routing | "
                f"Router model: {router_model} -> Target: {target_model} | "
                f"Total time: {total_time*1000:.2f}ms"
            )
            
            return target_model, probabilities
        
        except Exception as e:
            logger.error(f"Error in nn_objective_fn routing: {e}", exc_info=True)
            logger.warning(f"Using default fallback model: {DEFAULT_FALLBACK_MODEL}")
            return DEFAULT_FALLBACK_MODEL
    
    yield FunctionInfo.from_fn(
        _response_fn,
        description="Neural network objective function for intelligent model routing using embeddings and trained router."
    )

