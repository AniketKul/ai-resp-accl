# benchmark.py
from baseline import BaselineGenerator
from speculative import QwenSpeculativeGenerator
import pandas as pd
import json
import time

# Enterprise-relevant prompts for CEO demo
DEMO_PROMPTS = [
    "Summarize the key benefits of cloud migration for enterprise companies in 3 bullet points.",
    "Write a professional email declining a meeting request due to scheduling conflicts.",
    "Explain our data privacy policy regarding customer information in simple terms that anyone can understand.",
    "What are the main differences between agile and waterfall project management methodologies?",
    "Draft a 2-sentence product announcement for our new AI-powered analytics platform.",
    "List 3 key performance indicators for measuring customer satisfaction in SaaS businesses.",
    "Describe the advantages of microservices architecture over monolithic architecture.",
    "Write a brief response to a customer asking about our return policy for enterprise software licenses."
]

def run_comprehensive_benchmark():
    print("\n" + "="*80)
    print("🎯 QWEN 2.5 SPECULATIVE DECODING BENCHMARK")
    print("="*80)
    print("\n⚠️  This benchmark loads both models simultaneously (~60GB GPU memory required)")
    print("    Ensure sufficient GPU memory is available before proceeding.\n")
    
    # Initialize both generators
    print("\n[1/3] Loading Baseline Generator (Qwen2.5-72B-AWQ)...")
    baseline = BaselineGenerator()
    
    print("\n[2/3] Loading Speculative Generator (Qwen2.5-7B → Qwen2.5-72B-AWQ)...")
    speculative = QwenSpeculativeGenerator(num_speculative_tokens=5)
    
    print("\n[3/3] Running Benchmarks on {} prompts...".format(len(DEMO_PROMPTS)))
    print("="*80)
    
    results = []
    
    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        print(f"\n{'─'*80}")
        print(f"PROMPT {i}/{len(DEMO_PROMPTS)}")
        print(f"{'─'*80}")
        print(f"📝 {prompt[:75]}...")
        
        # Baseline inference
        print("\n[BASELINE - Sequential Decoding]")
        base_result = baseline.generate(prompt)
        print(f"  ⏱️  Latency: {base_result['latency']:.2f}s")
        print(f"  🚀 Speed: {base_result['tokens_per_sec']:.1f} tok/s")
        print(f"  📊 Tokens: {base_result['tokens']}")
        
        time.sleep(0.5)  # Brief pause
        
        # Speculative inference
        print("\n[SPECULATIVE - Parallel Verification]")
        spec_result = speculative.generate(prompt)
        print(f"  ⏱️  Latency: {spec_result['latency']:.2f}s")
        print(f"  🚀 Speed: {spec_result['tokens_per_sec']:.1f} tok/s")
        print(f"  📊 Tokens: {spec_result['tokens']}")
        
        # Calculate metrics
        speedup = base_result['latency'] / spec_result['latency']
        latency_reduction = ((base_result['latency'] - spec_result['latency']) / base_result['latency']) * 100
        throughput_increase = ((spec_result['tokens_per_sec'] - base_result['tokens_per_sec']) / base_result['tokens_per_sec']) * 100
        
        print(f"\n  🎯 SPEEDUP: {speedup:.2f}x faster")
        print(f"  📉 Latency Reduction: {latency_reduction:.1f}%")
        print(f"  📈 Throughput Increase: {throughput_increase:.1f}%")
        
        # Store results
        results.append({
            'prompt_id': i,
            'prompt': prompt,
            'prompt_preview': prompt[:60] + "...",
            'baseline_latency': round(base_result['latency'], 3),
            'baseline_tokens_per_sec': round(base_result['tokens_per_sec'], 1),
            'baseline_tokens': base_result['tokens'],
            'speculative_latency': round(spec_result['latency'], 3),
            'speculative_tokens_per_sec': round(spec_result['tokens_per_sec'], 1),
            'speculative_tokens': spec_result['tokens'],
            'speedup': round(speedup, 2),
            'latency_reduction_pct': round(latency_reduction, 1),
            'throughput_increase_pct': round(throughput_increase, 1),
            'baseline_output': base_result['text'],
            'speculative_output': spec_result['text']
        })
        
        time.sleep(1)  # Cool down between runs
    
    # Calculate summary statistics
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY - QWEN 2.5 SPECULATIVE DECODING")
    print("="*80)
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Average Speedup: {df['speedup'].mean():.2f}x (range: {df['speedup'].min():.2f}x - {df['speedup'].max():.2f}x)")
    print(f"   Median Speedup: {df['speedup'].median():.2f}x")
    
    print(f"\n⏱️  Latency:")
    print(f"   Baseline Avg: {df['baseline_latency'].mean():.2f}s")
    print(f"   Speculative Avg: {df['speculative_latency'].mean():.2f}s")
    print(f"   Avg Reduction: {df['latency_reduction_pct'].mean():.1f}%")
    
    print(f"\n🚀 Throughput:")
    print(f"   Baseline Avg: {df['baseline_tokens_per_sec'].mean():.1f} tok/s")
    print(f"   Speculative Avg: {df['speculative_tokens_per_sec'].mean():.1f} tok/s")
    print(f"   Avg Increase: {df['throughput_increase_pct'].mean():.1f}%")
    
    print(f"\n💰 Business Impact (assuming 1M queries/day):")
    time_saved_per_query = df['baseline_latency'].mean() - df['speculative_latency'].mean()
    total_time_saved_hours = (time_saved_per_query * 1_000_000) / 3600
    print(f"   Time saved per query: {time_saved_per_query:.2f}s")
    print(f"   Total time saved daily: {total_time_saved_hours:.1f} hours")
    print(f"   GPU capacity increase: {df['speedup'].mean():.1f}x (serve {df['speedup'].mean():.1f}x more users)")
    
    # Save results
    output_files = {
        'csv': 'qwen_benchmark_results.csv',
        'json': 'qwen_benchmark_results.json',
        'summary': 'qwen_benchmark_summary.txt'
    }
    
    df.to_csv(output_files['csv'], index=False)
    with open(output_files['json'], 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    with open(output_files['summary'], 'w') as f:
        f.write("QWEN 2.5 SPECULATIVE DECODING BENCHMARK SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Average Speedup: {df['speedup'].mean():.2f}x\n")
        f.write(f"Latency Reduction: {df['latency_reduction_pct'].mean():.1f}%\n")
        f.write(f"Throughput Increase: {df['throughput_increase_pct'].mean():.1f}%\n")
    
    print(f"\n📁 Results saved:")
    for file_type, filename in output_files.items():
        print(f"   {file_type.upper()}: {filename}")
    
    print("\n✅ Benchmark complete!")
    
    return df

if __name__ == "__main__":
    results_df = run_comprehensive_benchmark()
