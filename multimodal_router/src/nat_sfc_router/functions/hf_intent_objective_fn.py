import json
from typing import Any, Dict, List
from transformers import AutoTokenizer
import logging
from typing import Tuple, List, Optional
import asyncio
from functools import lru_cache
import time
import requests
import os 
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# Remote model configuration
REMOTE_MODEL_URL = os.getenv("ARCH_ROUTER_ENDPOINT")
MODEL_NAME = "katanemo/Arch-Router-1.5B"

# Only need tokenizer for prompt encoding (model is remote)
tokenizer = None

def _load_tokenizer():
    """Lazy load the tokenizer on first use."""
    global tokenizer
    if tokenizer is None:
        logger.info(f"Loading tokenizer for {MODEL_NAME}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    return tokenizer

def _check_remote_model():
    """Check if remote model is available."""
    try:
        response = requests.get(f"{REMOTE_MODEL_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Remote model at {REMOTE_MODEL_URL} is available")
            return True
    except Exception as e:
        logger.error(f"Remote model at {REMOTE_MODEL_URL} is not available: {e}")
        return False

# Please use our provided prompt for best performance
TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent. 
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"} 
"""

# Custom JSON encoder for Pydantic models and non-serializable objects
class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # Handle dict-like objects
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Handle iterables (except strings)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return list(obj)
            except TypeError:
                pass
        return super().default(obj)

# Define route config
route_config = [
    {
        "name": "hard_question",
        "description": "A question that requires deep reasoning, or complex problem solving, or if the user asks for careful thinking or careful consideration",
    },
    {
        "name": "chit_chat",
        "description": "Any social chit chat, small talk, or casual conversation.",
    },
    {
        "name": "try_again",
        "description": "Only if the user explicitly says the previous answer was incorrect or incomplete.",
    },
    {
        "name": "image_understanding",
        "description": "A question that requires understanding an image.",
    },
    {
        "name": "image_question",
        "description": "A question that requires the assistant to see the user eg a question about their appearance, environment, scene or surroundings.",
    },
]

# Pre-compute routes JSON once to avoid repeated serialization
_ROUTES_JSON_CACHED = json.dumps(route_config, cls=PydanticEncoder)

MAP_INTENT_TO_PIPELINE = {
    "other": "nvidia/nvidia-nemotron-nano-9b-v2",
    "chit_chat": "nvidia/nvidia-nemotron-nano-9b-v2",
    "hard_question": "gpt-5-chat",
    "image_understanding": "Qwen/Qwen3-VL-8B-Instruct",
    "image_question": "Qwen/Qwen3-VL-8B-Instruct",
    "try_again": "gpt-5-chat",
}

# Helper function to redact images while preserving text context
def redact_images_from_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove image data for router."""
    redacted = []
    for i, msg in enumerate(conversation):
        msg_copy = msg.copy()
        content = msg_copy.get("content")
    
        
        # If content is a list (multimodal), process it
        if isinstance(content, list):
            text_parts = []
            
            for item in content:
                logger.info(f"  Item: {type(item)}, {item if not isinstance(item, dict) else list(item.keys())}")
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        item_text = item.get("text", "")
                        text = f"<new msg>{item_text} </msg>"
                        text_parts.append(text)
                    elif item.get("type") == "image_url":
                        continue
                        
            
            # Combine text parts and add image indicator if present
            combined_text = " ".join(text_parts)
            
            msg_copy["content"] = combined_text
        
        redacted.append(msg_copy)
    
    return redacted

# Helper function to create the system prompt for our model
def format_prompt(conversation: List[Dict[str, Any]]):
    """Create the system prompt - uses pre-computed routes JSON for efficiency."""
    return (
        TASK_INSTRUCTION.format(
            routes=_ROUTES_JSON_CACHED,  # Use pre-computed JSON
            conversation=json.dumps(conversation, cls=PydanticEncoder)
        )
        + FORMAT_PROMPT
    )

# Cached JSON response parsing
@lru_cache(maxsize=128)
def _parse_route_response(response: str) -> str:
    """Parse and cache route responses to avoid repeated JSON parsing."""
    try:
        return json.loads(response)["route"]
    except json.JSONDecodeError:
        # Handle single quote format
        import ast
        return ast.literal_eval(response)["route"]



class HFIntentObjectiveConfig(FunctionBaseConfig, name="hf_intent_objective_fn"):
    """HF intent objective function for best route."""
    pass


def materialize_iterator(obj):
    """Recursively convert ValidatorIterator and other iterables to lists."""
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
        try:
            return [materialize_iterator(item) for item in obj]
        except TypeError:
            pass
    elif isinstance(obj, dict):
        return {k: materialize_iterator(v) for k, v in obj.items()}
    return obj

@register_function(config_type=HFIntentObjectiveConfig)
async def hf_intent_objective_fn(config: HFIntentObjectiveConfig,
                                 _builder: Builder):
    """HF intent objective function for best route."""

    from nat_sfc_router.schema.openai_chat_request import OpenAIChatRequest

    # Check if remote model is available
    _check_remote_model()
    
    # Load tokenizer (model is remote)
    loaded_tokenizer = _load_tokenizer()

    def get_route_from_conversation(conversation: List[Dict[str, Any]]) -> str:
        """Determine the best route for the conversation (using remote model)."""
        inference_start = time.perf_counter()
        
        # Redact images from messages because the router does not support them
        # But it can still determine if the text intent requires image understanding
        redacted_conversation = redact_images_from_conversation(conversation)
        
        # ===== FORMAT PROMPT =====
        prompt_start = time.perf_counter()
        route_prompt = format_prompt(redacted_conversation)
        prompt_time = time.perf_counter() - prompt_start
        
        # ===== CONSTRUCT MESSAGES =====
        construct_start = time.perf_counter()
        messages = [
            {"role": "user", "content": route_prompt},
        ]
        construct_time = time.perf_counter() - construct_start

        # ===== ENCODE (TOKENIZE) =====
        # Not needed for remote API, but keeping for timing consistency
        encode_start = time.perf_counter()
        encode_time = time.perf_counter() - encode_start

        # ===== GENERATION (REMOTE API CALL) =====
        generation_start = time.perf_counter()
        try:
            # Call remote vLLM OpenAI-compatible API
            response = requests.post(
                f"{REMOTE_MODEL_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "max_tokens": 32,
                    "temperature": 0.3,
                    "top_p": 0.9,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to call remote model: {e}")
            raise
        
        generation_time = time.perf_counter() - generation_start

        # ===== DECODING =====
        decode_start = time.perf_counter()
        # Response is already decoded text from remote API
        decode_time = time.perf_counter() - decode_start
        
        # Use cached parser
        route = _parse_route_response(response_text)
        
        total_time = time.perf_counter() - inference_start
        
        # Log timing breakdown
        logger.info(
            f"Route inference timing breakdown | "
            f"Format: {prompt_time*1000:.2f}ms | "
            f"Construct: {construct_time*1000:.2f}ms | "
            f"Encode: {encode_time*1000:.2f}ms | "
            f"Generate: {generation_time*1000:.2f}ms | "
            f"Decode: {decode_time*1000:.2f}ms | "
            f"Total: {total_time*1000:.2f}ms"
        )
        logger.debug(f"Route: {route}, Response: {response_text[:100]}")
        
        return route

    async def _response_fn(chat_request: OpenAIChatRequest) -> Tuple[str, str]:  # pyright: ignore[reportUnusedParameter]
        """HF intent objective function for best route."""
        response_start = time.perf_counter()

        # ===== EXTRACT MESSAGES =====
        extract_start = time.perf_counter()
        messages = chat_request.messages
        extract_time = time.perf_counter() - extract_start

        if messages:
            # ===== CONVERT TO DICT =====
            dict_convert_start = time.perf_counter()
            last_msg = messages[-1]
            last_msg_dict = last_msg.model_dump() if hasattr(last_msg, 'model_dump') else dict(last_msg)
            dict_convert_time = time.perf_counter() - dict_convert_start

            # ===== MATERIALIZE ITERATORS =====
            materialize_start = time.perf_counter()
            last_msg_dict = materialize_iterator(last_msg_dict)
            materialize_time = time.perf_counter() - materialize_start

            # Assign a list containing only the last message's dictionary
            messages_dict = [last_msg_dict]
            
            logger.debug(
                f"Message preparation timing | "
                f"Extract: {extract_time*1000:.2f}ms | "
                f"Dict convert: {dict_convert_time*1000:.2f}ms | "
                f"Materialize: {materialize_time*1000:.2f}ms"
            )
        else:
            # Handle the case where the list of messages is empty
            messages_dict = []
            logger.warning("No messages received in chat request")

        # Run model inference (blocking call in event loop)
        user_intent = get_route_from_conversation(messages_dict)
        
        total_response_time = time.perf_counter() - response_start

        logger.info(f"User intent: {user_intent} (total response time: {total_response_time*1000:.2f}ms)")
        return MAP_INTENT_TO_PIPELINE[user_intent], ""
    

    yield FunctionInfo.from_fn(
        _response_fn,
        description="Demonstrative objective function for best model.")
