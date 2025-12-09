import json
import numpy as np
from clip_client import Client
from tqdm import tqdm
import os

# CLIP embedding dimensions
CLIP_TEXT_DIM = 512
CLIP_IMAGE_DIM = 512
COMBINED_DIM = CLIP_TEXT_DIM + CLIP_IMAGE_DIM  # 1024


def load_embedding_model():
    """Load the CLIP client"""
    print("Connecting to CLIP server...")
    clip_server = os.getenv("CLIP_SERVER")
    client = Client(f'grpc://{clip_server}')
    print("Connected to CLIP server")
    return client


def load_json_data(json_path):
    """
    Load data from JSON file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_prompt_with_images(sample):
    """
    Prepare the prompt by collecting images from JSON format
    """
    prompt = sample["prompt"]
    images = sample.get("images", [])
    
    # Filter out None values if any
    images = [img for img in images if img is not None]
    
    return {
        "prompt": prompt,
        "images": images,
        "task": sample.get("task", ""),
        "label": sample.get("label", "")
    }


def generate_embeddings_for_dataset(clip_client, dataset_split, max_samples=None):
    """
    Generate embeddings for all prompts in the dataset using CLIP
    
    Args:
        clip_client: The CLIP client
        dataset_split: Dataset split to process
        max_samples: Maximum number of samples to process (None = all)
    
    Returns:
        numpy array of embeddings [num_samples, 1024]
        - If text + image: [text_embedding(512) | image_embedding(512)]
        - If text only: [text_embedding(512) | zeros(512)]
    """
    embeddings_list = []
    
    num_to_process = len(dataset_split) if max_samples is None else min(max_samples, len(dataset_split))
    
    print(f"Generating embeddings for {num_to_process} samples...")
    print(f"Output embedding dimension: {COMBINED_DIM}")
    
    for idx in tqdm(range(num_to_process)):
        sample = dataset_split[idx]
        prompt_data = prepare_prompt_with_images(sample)
        
        text = prompt_data['prompt']
        images = prompt_data['images']
        
        # Generate embedding based on whether images are present
        if len(images) > 0:
            # Has both text and image - concatenate text and image embeddings
            # Encode text and image separately
            text_embedding = clip_client.encode([text])  # Shape: (1, 512)
            image_data_uri = f"data:image/png;base64,{images[0]}"
            image_embedding = clip_client.encode([image_data_uri])  # Shape: (1, 512)
            
            # Concatenate text and image embeddings
            combined_embedding = np.concatenate([
                text_embedding[0],  # (512,)
                image_embedding[0]  # (512,)
            ])  # Result: (1024,)
            
            embeddings_list.append(combined_embedding)
        else:
            # Text only - pad with zeros
            text_embedding = clip_client.encode([text])  # Shape: (1, 512)
            
            # Pad with 512 zeros
            padding = np.zeros(CLIP_IMAGE_DIM)
            combined_embedding = np.concatenate([
                text_embedding[0],  # (512,)
                padding  # (512,)
            ])  # Result: (1024,)
            
            embeddings_list.append(combined_embedding)
    
    # Stack all embeddings
    embeddings = np.stack(embeddings_list, axis=0)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings


def main():
    # Load the HF pairwise data from JSON files
    print("Loading HF pairwise data from JSON files...")
    train_data = load_json_data("hf_evaluations_train.json")
    test_data = load_json_data("hf_evaluations_test.json")
    
    print(f"Loaded {len(train_data)} train samples")
    print(f"Loaded {len(test_data)} test samples")
    
    # Load CLIP client
    clip_client = load_embedding_model()
    
    # Generate embeddings for train set
    print("\n" + "="*80)
    print("Generating TRAIN embeddings...")
    print("="*80)
    train_embeddings = generate_embeddings_for_dataset(clip_client, train_data, max_samples=None)
    
    # Generate embeddings for test set
    print("\n" + "="*80)
    print("Generating TEST embeddings...")
    print("="*80)
    test_embeddings = generate_embeddings_for_dataset(clip_client, test_data, max_samples=None)
    
    # Save train and test embeddings separately
    train_output_path = "hf_train_embeddings.npy"
    test_output_path = "hf_test_embeddings.npy"
    
    np.save(train_output_path, train_embeddings)
    np.save(test_output_path, test_embeddings)
    
    print(f"\n✓ Saved train embeddings to: {train_output_path}")
    print(f"  Shape: {train_embeddings.shape}")
    print(f"✓ Saved test embeddings to: {test_output_path}")
    print(f"  Shape: {test_embeddings.shape}")
    
    # Save metadata
    metadata = {
        "train_samples": len(train_embeddings),
        "test_samples": len(test_embeddings),
        "embedding_dim": train_embeddings.shape[1],
        "model": "CLIP (grpc://10.185.119.147:51000)",
        "text_dim": CLIP_TEXT_DIM,
        "image_dim": CLIP_IMAGE_DIM,
        "combined_dim": COMBINED_DIM,
        "embedding_structure": "text_embedding(512) | image_embedding_or_zeros(512)"
    }
    
    with open("hf_embeddings_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: hf_embeddings_metadata.json")
    
    print("\n" + "="*80)
    print("Embeddings generated successfully!")
    print("You can now use these embeddings for training the matrix factorization model.")
    print("="*80)


if __name__ == "__main__":
    main()