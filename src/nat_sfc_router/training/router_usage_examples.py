#!/usr/bin/env python3
"""
Comprehensive examples of using the ModelRouter class

This file demonstrates various ways to use the ModelRouter for production deployments.
"""

from model_router import ModelRouter, route_query
import time
from typing import List


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize with balanced configuration (recommended)
    router = ModelRouter(threshold_config='balanced', verbose=True)
    
    # Route a simple query
    query = "What is the capital of France?"
    model = router.route(query)
    
    print(f"\nQuery: {query}")
    print(f"Selected model: {model}")


def example_2_with_probabilities():
    """Example 2: Getting probability scores"""
    print("\n" + "="*80)
    print("EXAMPLE 2: With Probability Scores")
    print("="*80)
    
    router = ModelRouter(threshold_config='balanced', verbose=False)
    
    queries = [
        "What is 2+2?",
        "Explain quantum entanglement in detail",
        "Hello, how are you?",
    ]
    
    for query in queries:
        model, probs = router.route(query, return_probabilities=True)
        print(f"\nQuery: {query}")
        print(f"Selected: {model}")
        print("Probabilities:")
        for m, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {m:40s}: {p:.3f}")


def example_3_threshold_configurations():
    """Example 3: Comparing different threshold configurations"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Threshold Configurations")
    print("="*80)
    
    query = "Explain the theory of relativity"
    
    configs = ['none', 'conservative', 'balanced', 'aggressive']
    
    print(f"\nQuery: {query}\n")
    print(f"{'Configuration':<15} {'Selected Model':<45} {'Cost Savings':<15}")
    print("-" * 80)
    
    for config in configs:
        router = ModelRouter(threshold_config=config, verbose=False)
        model = router.route(query)
        savings = ModelRouter.THRESHOLD_CONFIGS[config]['estimated_savings']
        print(f"{config:<15} {model:<45} {savings:<15}")


def example_4_custom_thresholds():
    """Example 4: Using custom thresholds"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Thresholds")
    print("="*80)
    
    # Define custom thresholds
    custom_thresholds = {
        'gpt-5-chat': 0.75,  # Very high threshold for GPT-5
        'nemotron-nano-12b-v2-vl': 0.60  # Moderate threshold for Nemotron
    }
    
    router = ModelRouter(
        model_thresholds=custom_thresholds,
        verbose=False
    )
    
    queries = [
        "Simple math: 5 + 3",
        "Complex analysis of macroeconomic trends",
        "What time is it?"
    ]
    
    print(f"\nCustom thresholds: {custom_thresholds}\n")
    
    for query in queries:
        info = router.route(query, return_all_info=True)
        print(f"Query: {query}")
        print(f"  Original choice: {info['original_choice']}")
        print(f"  Final choice: {info['chosen_model']}")
        print(f"  Threshold applied: {info['threshold_applied']}")
        print()


def example_5_batch_processing():
    """Example 5: Batch processing for efficiency"""
    print("="*80)
    print("EXAMPLE 5: Batch Processing")
    print("="*80)
    
    router = ModelRouter(threshold_config='balanced', verbose=False)
    
    # Simulate a batch of user queries
    queries = [
        "What is machine learning?",
        "2 + 2 = ?",
        "Explain blockchain technology",
        "Hello!",
        "What's the weather?",
        "Solve the traveling salesman problem",
        "Good morning",
        "Describe photosynthesis",
        "Calculate 15% of 200",
        "What is love?"
    ]
    
    print(f"\nProcessing {len(queries)} queries in batch...\n")
    
    start_time = time.time()
    models = router.route_batch(queries)
    elapsed_time = time.time() - start_time
    
    print(f"{'Query':<50} {'Model':<40}")
    print("-" * 95)
    for query, model in zip(queries, models):
        short_query = query[:47] + "..." if len(query) > 50 else query
        short_model = model.split('/')[-1] if '/' in model else model
        print(f"{short_query:<50} {short_model:<40}")
    
    print(f"\nProcessed {len(queries)} queries in {elapsed_time:.2f} seconds")
    print(f"Average time per query: {elapsed_time/len(queries):.3f} seconds")
    
    # Print statistics
    router.print_routing_stats(models)


def example_6_with_images():
    """Example 6: Routing with image inputs"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Routing with Images")
    print("="*80)
    
    router = ModelRouter(threshold_config='balanced', verbose=False)
    
    # Example with dummy base64 image (in production, use real images)
    # For demonstration, we'll use a placeholder
    dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # Text only query
    text_only_query = "What is in this image?"
    model_text_only = router.route(text_only_query)
    
    # Same query with image
    model_with_image = router.route(
        text_only_query,
        images=[dummy_image_base64]
    )
    
    print(f"Query: {text_only_query}")
    print(f"  Text only → {model_text_only}")
    print(f"  With image → {model_with_image}")
    print("\nNote: With actual images, routing may differ based on visual content")


def example_7_production_integration():
    """Example 7: Production integration pattern"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Production Integration Pattern")
    print("="*80)
    
    # Initialize router once at application startup
    router = ModelRouter(threshold_config='balanced', verbose=False)
    
    print("\nProduction usage pattern:")
    print("""
    # In your application initialization
    global_router = ModelRouter(threshold_config='balanced')
    
    # In your request handler
    def handle_user_query(user_query, user_images=None):
        # Route to best model
        chosen_model = global_router.route(user_query, user_images)
        
        # Call the appropriate model API
        if chosen_model == 'gpt-5-chat':
            response = call_gpt5_api(user_query, user_images)
        elif chosen_model == 'nemotron-nano-12b-v2-vl':
            response = call_nemotron_api(user_query, user_images)
        else:  # llama
            response = call_llama_api(user_query, user_images)
        
        return response
    """)
    
    # Demonstrate
    example_queries = [
        "Quick question: What's 5+5?",
        "Detailed explanation needed: How does nuclear fusion work?",
    ]
    
    print("\nExample routing decisions:")
    for query in example_queries:
        info = router.route(query, return_all_info=True)
        print(f"\nQuery: {query}")
        print(f"  Route to: {info['chosen_model']}")
        print(f"  Confidence: {info['probabilities'][info['chosen_model']]:.3f}")


def example_8_monitoring_and_logging():
    """Example 8: Monitoring and logging for production"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Monitoring and Logging")
    print("="*80)
    
    router = ModelRouter(threshold_config='balanced', verbose=False)
    
    # Simulate a day's worth of queries
    queries = [
        "What is AI?",
        "2+2",
        "Explain quantum physics",
        "Hello",
        "Weather forecast",
    ] * 10  # Repeat to simulate more traffic
    
    # Track routing decisions
    routing_decisions = []
    
    for query in queries:
        info = router.route(query, return_all_info=True)
        
        # Log routing decision
        routing_decisions.append({
            'query': query,
            'model': info['chosen_model'],
            'confidence': info['probabilities'][info['chosen_model']],
            'threshold_applied': info['threshold_applied'],
        })
    
    # Analyze routing patterns
    print(f"\nAnalyzed {len(routing_decisions)} routing decisions:")
    
    # Count threshold applications
    threshold_applied_count = sum(1 for d in routing_decisions if d['threshold_applied'])
    print(f"  Threshold applied: {threshold_applied_count}/{len(routing_decisions)} "
          f"({threshold_applied_count/len(routing_decisions)*100:.1f}%)")
    
    # Model distribution
    models_used = [d['model'] for d in routing_decisions]
    router.print_routing_stats(models_used)
    
    # Average confidence by model
    print("\nAverage confidence when chosen:")
    for model in router.model_names:
        model_decisions = [d for d in routing_decisions if d['model'] == model]
        if model_decisions:
            avg_conf = sum(d['confidence'] for d in model_decisions) / len(model_decisions)
            print(f"  {model:40s}: {avg_conf:.3f}")


def example_9_quick_utility_function():
    """Example 9: Quick utility function for single queries"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Quick Utility Function")
    print("="*80)
    
    print("\nFor single queries without creating a router instance:")
    print("(Note: This is less efficient for multiple queries)\n")
    
    # One-liner routing
    model = route_query("What is the meaning of life?", threshold_config='balanced')
    print(f"Quick route: 'What is the meaning of life?' → {model}")


def main():
    """Run all examples"""
    print("\n" + "="*100)
    print(" "*30 + "MODEL ROUTER EXAMPLES")
    print("="*100)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("With Probabilities", example_2_with_probabilities),
        ("Threshold Configurations", example_3_threshold_configurations),
        ("Custom Thresholds", example_4_custom_thresholds),
        ("Batch Processing", example_5_batch_processing),
        ("With Images", example_6_with_images),
        ("Production Integration", example_7_production_integration),
        ("Monitoring and Logging", example_8_monitoring_and_logging),
        ("Quick Utility Function", example_9_quick_utility_function),
    ]
    
    for name, example_func in examples:
        print(f"\n\n")
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*100)
    print(" "*35 + "ALL EXAMPLES COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()

