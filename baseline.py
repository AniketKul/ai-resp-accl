# baseline.py
from vllm import LLM, SamplingParams
import time

class BaselineGenerator:
    def __init__(self, model_path="./models/target"):
        print("🚀 Loading Qwen2.5-72B-AWQ (Baseline Mode)...")
        print("This may take 1-2 minutes...")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,  # Reduced to prevent OOM
            max_model_len=2048,  # Reduced context length for better memory efficiency
            trust_remote_code=True,  # Required for Qwen
            quantization="awq",  # Use AWQ quantization to fit in 80GB
            dtype="auto"
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["<|endoftext|>", "<|im_end|>"]  # Qwen-specific stop tokens
        )
        
        print("✅ Qwen2.5-72B-AWQ loaded successfully!")
    
    def generate(self, prompt):
        """Generate response with timing"""
        # Format prompt for Qwen chat template
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        start_time = time.time()
        outputs = self.llm.generate([formatted_prompt], self.sampling_params)
        latency = time.time() - start_time
        
        output_text = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        return {
            'text': output_text.strip(),
            'latency': latency,
            'tokens': num_tokens,
            'tokens_per_sec': num_tokens / latency if latency > 0 else 0
        }

# Test
if __name__ == "__main__":
    gen = BaselineGenerator()
    
    test_prompt = "Explain quantum computing in 2 sentences."
    print(f"\n{'='*70}")
    print(f"Test Prompt: {test_prompt}")
    print(f"{'='*70}")
    
    result = gen.generate(test_prompt)
    
    print(f"\n📝 Output:\n{result['text']}")
    print(f"\n⏱️  Latency: {result['latency']:.2f}s")
    print(f"🚀 Speed: {result['tokens_per_sec']:.1f} tokens/sec")
    print(f"📊 Total Tokens: {result['tokens']}")
