# Memories-S0: An Efficient and Accurate Framework for Security Video Understanding

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Memories--ai%2Fsecurity__model-blue)](https://huggingface.co/Memories-ai/security_model)
[![SmartHomeBench](https://img.shields.io/badge/Benchmark-SmartHome--Bench-yellow)](https://github.com/Xinyi-0724/SmartHome-Bench-LLM)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**Memories-S0** is a novel, end-to-end framework designed to address the critical challenges of data scarcity and edge deployment in security video understanding. At its core lies **Memories-S0**, a highly efficient 3-billion-parameter (3B) Vision-Language Model (VLM) optimized for real-time surveillance analysis.

By leveraging synthetic data generation (Veo 3), extreme token compression, and post-training reinforcement learning, Memories-S0 achieves State-of-the-Art (SOTA) performance on anomaly detection tasks while maintaining a minimal computational footprint suitable for mobile and edge devices.

---

## 🚀 Key Innovations

### 1. Data-Centric Generation Pipeline

To overcome the scarcity of high-quality surveillance data , we utilize advanced video generation models (e.g., Veo 3) to synthesize a massive, diverse set of domain-specific videos.

**Scenario Diversity:** Prompts cover various lighting conditions and security scenarios (e.g., "unattended package," "dimly lit hallway").

**Pixel-Perfect Annotation:** The generative process yields automatic, accurate ground truth labels for object detection and tracking.

### 2. Extreme Model Efficiency (3B Parameters)

We introduce specific optimization strategies to deploy complex video understanding on resource-constrained edge devices:

**Input Token Compression:** An innovative algorithm analyzes spatiotemporal redundancy, pruning background tokens to focus attention solely on foreground objects and motion.

**Model Compression:** Utilization of structured pruning and low-rank factorization to reduce memory footprint without compromising accuracy.

### 3. Efficient Post-Training Strategy

Instead of costly full-scale fine-tuning, we employ a two-step post-training recipe:

**Event-Based Temporal Shuffling:** Combined with Reinforcement Learning (RL) to enhance the model's understanding of sequential relationships.

**Supervised Fine-Tuning (SFT) & RL:** A specialized recipe that boosts capabilities with minimal computational overhead.

---

## 📊 Performance (SmartHomeBench)

We evaluated Memories-S0(3B) on the **SmartHomeBench** dataset, a recognized benchmark for smart home video anomaly detection.

Despite having only **3B parameters**, our model achieves an **F1-score of 79.21** using a simple **Zero-shot** prompt, surpassing larger models like VILA-13b and performing competitively against GPT-4o and Claude-3.5-Sonnet (which require complex Chain-of-Thought prompting).

| Model | Params | Prompting Method | Accuracy | Precision | Recall | **F1-score** |
| --- | --- | --- | --- | --- | --- | --- |
| **Memories-S0 (Ours)** | **3B** | **Zero-shot** | **71.33** | **73.04** | **86.51** | **79.21** |
| VILA-13b | 13B | Few-shot CoT | 67.17 | 69.18 | 70.57 | 69.87 |
| GPT-4o | Closed | Zero-shot | 68.41 | 80.09 | 55.16 | 65.33 |
| Gemini-1.5-Pro | Closed | Zero-shot | 57.36 | 84.34 | 25.73 | 39.43 |

---

## 🛠️ Installation

```bash
conda create -n memories-s0 python=3.10 -y
conda activate memories-s0

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu121>

# Install dependencies for Qwen2.5-VL architecture and Flash Attention
pip install transformers>=4.37.0 accelerate qwen_vl_utils
pip install flash-attn --no-build-isolation

```

---

## 💻 Inference

The following script demonstrates how to run the **Memories-S0** model. It automatically handles the loading of weights from the official Hugging Face repository.

```python
import torch
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Official Model Repository
MODEL_ID = "Memories-ai/security_model"

def run_inference(video_path, model_id=MODEL_ID):
    # Load Model with Flash Attention 2 for efficiency
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Define Security Analysis Prompt
    prompt_text = """YOUR_PROMPT"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Preprocessing
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=768)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    args = parser.parse_args()
    run_inference(args.video_path)

```

---

## 📜 Citation

If you use this model or framework in your research, please cite the technical report:

```
@techreport{memories_s0_2025,
  title       = {{Memories-S0}: An Efficient and Accurate Framework for Security Video Understanding},
  author      = {{Memories.ai Research}},
  institution = {Memories.ai},
  year        = {2025},
  month       = oct,
  url         = {https://huggingface.co/Memories-ai/security_model},
  note        = {Accessed: 2025-11-20}
}

```

---

## ⚖️ Disclaimer

This model is intended for research and positive safety applications. Users are responsible for complying with all local laws and regulations regarding surveillance and privacy. The model's outputs are probabilistic and should be verified by human oversight in critical scenarios.
