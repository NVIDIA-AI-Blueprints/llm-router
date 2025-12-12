import json
import random
import os
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
from PIL import Image
import base64
import io
from openai import AzureOpenAI, OpenAI
import time
from functools import wraps

MODEL_IDS = {
    "gpt-5-chat": 0,
    "nvidia/nemotron-nano-12b-v2-vl": 1,
    "nvidia/nvidia-nemotron-nano-9b-v2": 2
}

def retry_with_exponential_backoff(
    max_retries=50,
    initial_delay=1,
    exponential_base=2,
    max_delay=3000,
    jitter=True
):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        print(f"Max retries ({max_retries}) reached for {func.__name__}. Last error: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    current_delay = min(delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter if enabled
                    if jitter:
                        current_delay = current_delay * (0.5 + random.random())
                    
                    print(f"Error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {current_delay:.2f} seconds...")
                    time.sleep(current_delay)
            
            raise last_exception
        
        return wrapper
    return decorator


# Login using e.g. `huggingface-cli login` to access this dataset
print("Loading dataset...")
dataset = load_from_disk("finevision_combined_images_text")


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def prepare_prompt_with_images(sample):
    """
    Prepare the prompt by replacing <imageN> tags with actual image references
    and collecting the images
    """

    return {
        "prompt": sample["text"][0]["content"],
        "images": sample["images"],
        "answer": sample["answer"]
    }


@retry_with_exponential_backoff(max_retries=5, initial_delay=1)
def generate_response_with_openai(prompt_data):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2025-01-01-preview"
    )
    # Prepare content with text and images
    content = [{"type": "text", "text": prompt_data['prompt']}]
    
    # Add images if available
    for image in prompt_data.get('images', []):
        if image is not None:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
    
    response = client.chat.completions.create(
        model="gpt-5-chat",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content

@retry_with_exponential_backoff(max_retries=5, initial_delay=1)
def generate_response_with_nemotron_vlm(prompt_data):

    client = OpenAI(
        base_url=os.getenv("VLM_MODEL_BASE_URL"),
        api_key="not-needed"
    )
    content = [{"type": "text", "text": prompt_data['prompt']}]
    
    # Add images if available
    for image in prompt_data.get('images', []):
        if image is not None:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
    
    response = client.chat.completions.create(
        model="nvidia/nemotron-nano-12b-v2-vl",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content


def generate_response_with_text(prompt_data):
    client = OpenAI(
        base_url=os.getenv("NANO_TEXT_MODEL_BASE_URL"),
        api_key="not-needed"
    )
    # Build messages from conversation
    messages = []
    conversation = prompt_data['prompt']
    first_user_turn = True
    
    # Process each turn in the conversation
    for turn in conversation:
        role = turn.get("role", "user")
        content_text = turn.get("content", "")
        
        # For the first user message, include images
        if role == "user" and first_user_turn:
            first_user_turn = False
            content = [{"type": "text", "text": content_text}]
            
            # Add images if available
            for image_dict in prompt_data.get('images', []):
                if image_dict is not None and 'bytes' in image_dict:
                    # Get image bytes and convert to base64
                    img_bytes = image_dict['bytes']
                    if img_bytes is not None:
                        # Convert bytes to PIL Image then to base64
                        image = Image.open(io.BytesIO(img_bytes))
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        })
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": content_text})
    
    response = client.chat.completions.create(
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        messages=messages
    )
    return response.choices[0].message.content


def generate_model_responses(prompt_data, model_ids):
    """
    Generate responses from different models for the given prompt
    
    Args:
        prompt_data: Dict with 'prompt', 'images', etc.
        model_ids: List of model IDs to generate responses from
    
    Returns:
        Dict mapping model_id to response
    """
    responses = {}
    for model_id in model_ids:
        try:
            if model_id == "gpt-5-chat":
                responses[model_id] = generate_response_with_openai(prompt_data)
            elif model_id == "nvidia/nemotron-nano-12b-v2-vl":
                responses[model_id] = generate_response_with_nemotron_vlm(prompt_data)
            elif model_id == "nvidia/nvidia-nemotron-nano-9b-v2":
                responses[model_id] = "I cannot answer that question because I am a model that can only answer questions without images."
        except Exception as e:
            print(f"Failed to generate response for {model_id} after all retries. Skipping. Error: {e}")
            continue
    return responses


@retry_with_exponential_backoff(max_retries=5, initial_delay=1)
def judge_single_response(prompt_data, model_response, ground_truth_label):
    """
    Use Azure OpenAI as a judge to evaluate if a single model response correctly answers the question.
    Returns True if the model response is correct, False otherwise.
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2025-01-01-preview"
    )

    # Compose the system prompt for the judge
    sys_prompt = (
        "You are an expert AI tasked with evaluating whether a model's response correctly answers a question. "
        "You will be given the original prompt (with images), the ground truth answer, and the model's response. "
        "Your task is to determine whether the model's response correctly answers the question and matches "
        "or logically aligns with the ground truth answer.\n\n"
        "Reply strictly with 'yes' if the response is correct, or 'no' if it is incorrect."
    )

    content = [{"type": "text", "text": "PROMPT: " + prompt_data['prompt']}]
    
    # Add images if available
    for image in prompt_data.get('images', []):
        if image is not None:
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })

    content.append({"type": "text", "text": f"\n\nGROUND TRUTH ANSWER: {ground_truth_label}"})
    content.append({"type": "text", "text": f"\n\nMODEL RESPONSE: {model_response}"})

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-5-chat",
            messages=messages
        )
        judge_answer = response.choices[0].message.content.strip().lower()
        return judge_answer.startswith("yes")
    except Exception as e:
        print(f"Error during Azure judge evaluation: {e}")
        return False


def create_evaluation_data(dataset_split, num_samples=None):
    """
    Evaluate all models on the dataset and record success (1) or failure (0) for each model.
    
    Args:
        dataset_split: The dataset split to process
        num_samples: Number of samples to process (None = all)
    
    Returns:
        List of evaluation samples with model success indicators
    """
    evaluation_data = []
    model_list = list(MODEL_IDS.keys())
    
    # Limit number of samples if specified
    num_to_process = len(dataset_split) if num_samples is None else min(num_samples, len(dataset_split))
    
    print(f"\nProcessing {num_to_process} samples...")
    
    skipped_samples = 0
    
    for idx in tqdm(range(num_to_process)):
        sample = dataset_split[idx]
        
        # Prepare prompt with images
        prompt_data = prepare_prompt_with_images(sample)
        
        # Skip if prompt is empty
        if not prompt_data["prompt"] or len(prompt_data["prompt"]) == 0:
            print(f"Skipping sample {idx}: Empty prompt")
            skipped_samples += 1
            continue
        
        responses = generate_model_responses(prompt_data, model_list)
        
        # Judge each model's response independently
        model_scores = {}
        all_failed = True
        
        for model_id in model_list:
            if model_id not in responses:
                print(f"Skipping model {model_id} for sample {idx}: No response generated")
                model_scores[model_id] = 0
                continue
            
            try:
                # Judge if the response is correct
                is_correct = judge_single_response(prompt_data, responses[model_id], prompt_data["answer"])
                model_scores[model_id] = 1 if is_correct else 0
                if is_correct:
                    all_failed = False
                print(f"Sample {idx} - {model_id}: {'✓' if is_correct else '✗'}")
            except Exception as e:
                print(f"Failed to judge {model_id} for sample {idx}: {e}")
                model_scores[model_id] = 0
        
        # Skip if all models failed (optional - remove this if you want to keep all samples)
        # if all_failed:
        #     print(f"Skipping sample {idx}: All models failed")
        #     skipped_samples += 1
        #     continue
        
        # Convert images to base64 for serialization
        images_base64 = []
        for image in prompt_data["images"]:
            if image is not None:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                images_base64.append(img_base64)
        
        evaluation_data.append({
            "idx": idx,
            "prompt": prompt_data["prompt"],
            "answer": prompt_data["answer"],
            "images": images_base64,
            "model_scores": model_scores
        })
    
    print(f"\nSkipped {skipped_samples} samples due to failures")
    
    return evaluation_data


if __name__ == "__main__":
    # Evaluate all models on the dataset
    # For demonstration, process only a small subset
    # Set num_samples=None to process all data
    all_evaluations = create_evaluation_data(dataset, num_samples=2000)
    
    # Split into training and testing sets
    train_evaluations = all_evaluations[:1600]  # 80% for training
    test_evaluations = all_evaluations[1600:]   # 20% for testing
    
    print(f"\nGenerated {len(train_evaluations)} train samples")
    print(f"Generated {len(test_evaluations)} test samples")
    
    # Calculate and print success rates for each model
    print("\nModel Success Rates (Train):")
    for model_id in MODEL_IDS.keys():
        success_count = sum(1 for item in train_evaluations if item["model_scores"].get(model_id, 0) == 1)
        total_count = len(train_evaluations)
        print(f"  {model_id}: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    
    # Save the evaluation data
    output_train_path = "hf_evaluations_train.json"
    output_test_path = "hf_evaluations_test.json"
    
    with open(output_train_path, "w") as f:
        json.dump(train_evaluations, f, indent=2)
    
    with open(output_test_path, "w") as f:
        json.dump(test_evaluations, f, indent=2)
    
    print(f"\nSaved to {output_train_path} and {output_test_path}")
