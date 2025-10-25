# speculative.py
from vllm import LLM, SamplingParams
import time

class QwenSpeculativeGenerator:
    def __init__(self, 
                 draft_model_path="./models/draft",
                 target_model_path="./models/target",
                 num_speculative_tokens=5):
        
        print("🚀 Loading Qwen2.5-72B-AWQ with Speculative Decoding...")
        print(f"   Draft: Qwen2.5-7B")
        print(f"   Target: Qwen2.5-72B-AWQ")
        print(f"   Speculative tokens: {num_speculative_tokens}")
        print("This may take 2-3 minutes...")
        
        self.llm = LLM(
            model=target_model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,  # Reduced to prevent OOM with both models
            max_model_len=2048,  # Reduced context length for better memory efficiency
            trust_remote_code=True,
            quantization="awq",  # Use AWQ quantization for target model
            dtype="auto",
            # Speculative decoding configuration
            speculative_model=draft_model_path,
            num_speculative_tokens=num_speculative_tokens,
            use_v2_block_manager=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["<|endoftext|>", "<|im_end|>"]
        )
        
        print("✅ Speculative decoding enabled!")
        print(f"   Expected speedup: 2.5-3.5x")
    
    def generate(self, prompt):
        """Generate response with speculative decoding"""
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
    gen = QwenSpeculativeGenerator(num_speculative_tokens=5)
    
    test_prompt = "Explain quantum computing in 2 sentences."
    print(f"\n{'='*70}")
    print(f"Test Prompt: {test_prompt}")
    print(f"{'='*70}")
    
    result = gen.generate(test_prompt)
    
    print(f"\n📝 Output:\n{result['text']}")
    print(f"\n⏱️  Latency: {result['latency']:.2f}s")
    print(f"🚀 Speed: {result['tokens_per_sec']:.1f} tokens/sec")
    print(f"📊 Total Tokens: {result['tokens']}")
