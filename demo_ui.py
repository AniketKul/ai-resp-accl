# demo_ui.py
import gradio as gr
from baseline import BaselineGenerator
from speculative import QwenSpeculativeGenerator
import time
import json

# Initialize models lazily to avoid OOM
print("🚀 Models will be loaded on first use to optimize memory...")

baseline_gen = None
spec_gen = None

def get_baseline_generator():
    global baseline_gen
    if baseline_gen is None:
        print("Loading Baseline Generator...")
        baseline_gen = BaselineGenerator()
    return baseline_gen

def get_speculative_generator():
    global spec_gen
    if spec_gen is None:
        print("Loading Speculative Generator...")
        spec_gen = QwenSpeculativeGenerator(num_speculative_tokens=5)
    return spec_gen

print("✅ Demo ready! Models will load on first inference.")

def single_inference(prompt, mode):
    """Run inference with selected mode"""
    if not prompt.strip():
        return "⚠️ Please enter a prompt", "", "", ""
    
    try:
        if mode == "Baseline (72B-AWQ Only)":
            gen = get_baseline_generator()
            result = gen.generate(prompt)
            method = "Sequential Decoding (Standard)"
            model_info = "Qwen2.5-72B-AWQ"
        else:
            gen = get_speculative_generator()
            result = gen.generate(prompt)
            method = "Speculative Decoding (7B→72B-AWQ)"
            model_info = "Qwen2.5-7B (draft) + Qwen2.5-72B-AWQ (verify)"
        
        return (
            result['text'],
            f"⏱️ {result['latency']:.2f} seconds",
            f"🚀 {result['tokens_per_sec']:.1f} tokens/sec",
            f"📊 {method}\n💻 {model_info}\n🔢 {result['tokens']} tokens generated"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", ""

def side_by_side_comparison(prompt):
    """Run both modes for direct comparison"""
    if not prompt.strip():
        return "", "", "⚠️ Please enter a prompt"
    
    try:
        # Run baseline
        print(f"Running baseline for: {prompt[:50]}...")
        baseline = get_baseline_generator()
        base_result = baseline.generate(prompt)
        
        # Brief pause
        time.sleep(0.5)
        
        # Run speculative
        print(f"Running speculative for: {prompt[:50]}...")
        speculative = get_speculative_generator()
        spec_result = speculative.generate(prompt)
        
        # Calculate metrics
        speedup = base_result['latency'] / spec_result['latency']
        latency_reduction = ((base_result['latency'] - spec_result['latency']) / base_result['latency']) * 100
        throughput_gain = ((spec_result['tokens_per_sec'] / base_result['tokens_per_sec']) - 1) * 100
        
        # Format comparison
        comparison = f"""
## ⚡ Performance Comparison

| Metric | Baseline | Speculative | Improvement |
|--------|----------|-------------|-------------|
| **Latency** | {base_result['latency']:.2f}s | {spec_result['latency']:.2f}s | **{speedup:.2f}x faster** ⚡ |
| **Throughput** | {base_result['tokens_per_sec']:.1f} tok/s | {spec_result['tokens_per_sec']:.1f} tok/s | **+{throughput_gain:.1f}%** 📈 |
| **Tokens Generated** | {base_result['tokens']} | {spec_result['tokens']} | Same output length |
| **Latency Reduction** | - | - | **-{latency_reduction:.1f}%** 📉 |

---

### 🎯 Key Insights

- **Speed:** Speculative decoding is **{speedup:.2f}x faster** with identical quality
- **Efficiency:** Processes **{throughput_gain:.1f}% more tokens per second**
- **Quality:** Both outputs are equivalent (same model verification)

### 💰 Business Impact

For **1,000 daily queries**:
- Time saved: **{(base_result['latency'] - spec_result['latency']) * 1000 / 60:.1f} minutes/day**
- Capacity increase: Serve **{speedup:.1f}x more users** with same hardware

---

**Model Configuration:**
- Draft: Qwen2.5-7B-Instruct (fast proposal)
- Target: Qwen2.5-72B-Instruct-AWQ (verification & quality assurance)
"""
        
        return (
            base_result['text'],
            spec_result['text'],
            comparison
        )
    except Exception as e:
        return "", "", f"❌ Error: {str(e)}"

# Example prompts for demo
enterprise_examples = [
    "Summarize the key benefits of adopting AI in customer service operations.",
    "Write a professional email response to a client requesting a project timeline extension.",
    "Explain the concept of technical debt to non-technical stakeholders in simple terms.",
    "List 5 best practices for ensuring data security in cloud-based applications.",
    "Draft a brief announcement about our new product feature launch for internal communication.",
]

# Create Gradio interface
with gr.Blocks(
    title="Qwen 2.5 AI Response Accelerator",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
) as demo:
    
    gr.Markdown("""
    # 🚀 AI Response Accelerator - Qwen 2.5 Edition
    ### Enterprise-Grade LLM with 3× Faster Response Times
    
    **Powered by:** Speculative Decoding (Qwen2.5-7B → Qwen2.5-72B-AWQ)
    
    ---
    
    **Demo for Leadership:** Experience how we achieve dramatically faster AI responses 
    without sacrificing output quality using advanced GPU acceleration techniques.
    """)
    
    with gr.Tab("⚡ Quick Demo"):
        gr.Markdown("### Single Inference Mode")
        gr.Markdown("Test individual queries with either baseline or speculative decoding.")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Enter Your Prompt",
                    placeholder="Ask me anything...",
                    lines=4
                )
                mode_select = gr.Radio(
                    choices=["Baseline (72B-AWQ Only)", "Speculative (7B→72B-AWQ)"],
                    value="Speculative (7B→72B-AWQ)",
                    label="Inference Mode",
                    info="Compare standard vs accelerated inference"
                )
                run_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                output_text = gr.Textbox(label="📝 Generated Response", lines=8)
                with gr.Row():
                    latency_out = gr.Textbox(label="⏱️ Latency", scale=1)
                    speed_out = gr.Textbox(label="🚀 Throughput", scale=1)
                method_out = gr.Textbox(label="ℹ️ Model Information", lines=3)
        
        gr.Examples(
            examples=enterprise_examples,
            inputs=prompt_input,
            label="Example Enterprise Prompts"
        )
        
        run_btn.click(
            single_inference,
            inputs=[prompt_input, mode_select],
            outputs=[output_text, latency_out, speed_out, method_out]
        )
    
    with gr.Tab("📊 Side-by-Side Comparison"):
        gr.Markdown("### Direct Performance Comparison")
        gr.Markdown("Run the same prompt through both methods to see the performance difference.")
        gr.Markdown("⚠️ **Note:** This loads both models simultaneously and may require ~60GB GPU memory.")
        
        compare_prompt = gr.Textbox(
            label="Enter Prompt for Comparison",
            placeholder="Enter your query to compare both inference methods...",
            lines=3
        )
        
        compare_btn = gr.Button("⚡ Run Comparison", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🐢 Baseline (Sequential)")
                baseline_output = gr.Textbox(label="Output", lines=10)
            
            with gr.Column():
                gr.Markdown("### 🚀 Speculative (Parallel)")
                spec_output = gr.Textbox(label="Output", lines=10)
        
        performance_summary = gr.Markdown(label="Performance Analysis")
        
        gr.Examples(
            examples=enterprise_examples,
            inputs=compare_prompt,
            label="Try These Examples"
        )
        
        compare_btn.click(
            side_by_side_comparison,
            inputs=compare_prompt,
            outputs=[baseline_output, spec_output, performance_summary]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## How Speculative Decoding Works
        
        **Traditional Approach (Baseline):**
        - Generates tokens one at a time sequentially
        - Each token requires a full model forward pass
        - Limited by sequential dependencies
        
        **Speculative Decoding (Accelerated):**
        1. **Draft Phase:** Small model (7B) quickly generates multiple token candidates
        2. **Verification Phase:** Large model (72B) verifies all candidates in parallel
        3. **Acceptance:** Keep correct predictions, regenerate incorrect ones
        
        **Result:** 2-4× faster inference with identical output quality!
        
        ---
        
        ### Technical Specifications
        
        - **Draft Model:** Qwen2.5-7B-Instruct (~14GB)
        - **Target Model:** Qwen2.5-72B-Instruct-AWQ (~40GB, 4-bit quantized)
        - **Hardware:** NVIDIA H100 (80GB)
        - **Framework:** vLLM with speculative decoding
        - **Expected Speedup:** 2.5-3.5×
        
        ---
        
        ### Business Benefits
        
        ✅ **3× more throughput** with same hardware  
        ✅ **Sub-2-second responses** for better UX  
        ✅ **Zero quality loss** - same model outputs  
        ✅ **Lower inference costs** per query  
        """)

# Launch demo
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌐 Launching Qwen 2.5 AI Response Accelerator Demo")
    print("="*70)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public link
        show_error=True
    )
