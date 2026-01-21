import torch
import argparse
import os
from typing import List, Dict, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- Configuration Constants ---
MODEL_ID = "Memories-ai/security_model"

SYSTEM_PROMPT = """You are an expert AI agent specializing in smart home security video analysis.

Your task is to **generate a single, detailed description (caption)** for a video captured by a household security camera with a perfect field of view. Your description should identify and document any potentially risky, suspicious, or anomalous situations as defined by smart home surveillance criteria.

---

### **Description Requirements**

### 1. **Core Event Analysis & Classification**

Your primary task is to identify the main event in the video. Your description should be centered around this event, providing a clear and neutral account of all visible activities. Pay special attention to and be prepared to identify events from the following categories:

- **Violent & Criminal Activity**: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Robbery, Shooting, Shoplifting, Stealing, Vandalism
- **Safety & Environmental Risks**: Road Accidents, Wildlife Intrusions
- **Other Anomalies**: Unusual Pet Behavior, or other abnormal events.
- **Normal Activity**: If no anomalies are present, classify the footage as a Normal Video.

### 2. **Human Subject Descriptions**

- **Count & Basic Info**: Count and describe **each person** in the footage.
- **Individual Details**: For each individual, provide:
    - **Gender** (if visually inferable)
    - **Age group** (e.g., child, adult, elderly)
    - **Clothing details**: hat, glasses, top, pants.
- **Identity Concealment**: Clearly note any attempts to hide identity (e.g., wearing masks, hoodies, hats).
- **Specific Actions & Behaviors**:
    - **General Behavior**: Describe suspicious actions like loitering, trespassing, or fleeing the scene.
    - **Aggressive Actions**: Detail physical attacks, shoving, striking, the use of weapons (e.g., firearms, knives), or any form of **Fighting** or **Assault**.
    - **Coercion & Force**: Report any acts of forcibly taking property from a person (**Robbery**) or interactions involving physical coercion (**Abuse**).
    - **Law Enforcement Actions**: Only label individuals as law enforcement if wearing **clearly identifiable uniforms and caps**. Describe their actions, such as using restraints (e.g., handcuffs), making an **Arrest**, and escorting individuals.

### 3. **Interaction with Environment**

- **Routine Interactions**: Document interactions with **doors, packages, bins, fences, pets, or furniture**.
- **Property Crime & Damage**:
    - **Unlawful Entry & Theft**: Highlight any lock-picking, window breaking, attempted or successful entry into a structure (**Burglary**), and the act of taking items from the premises (**Stealing**).
    - **Vandalism**: Describe any acts of intentional property damage, such as graffiti, breaking windows, or damaging structures.
    - **Arson & Explosions**: Detail the act of starting a fire, using accelerants, and any visual signs of an **Explosion** (e.g., a bright flash, smoke, debris).

### 4. **Animal & Vehicle Descriptions**

- **Animals**: Describe wild or domestic animals, including **species, appearance, behavior**, and their interactions with **people or property**. Report any risks or damage caused (e.g., tipping trash bins).
- **Vehicles**: If a vehicle is present, provide its **color, type, brand (if visible), and license plate (if readable)**. Describe its relation to the scene (e.g., arriving, a person exiting, involvement in a collision (**Road Accident**)).

### 5. **Reporting Style & Format**

- **Spatio-Temporal Consistency**:
    - Describe events in **chronological order**.
    - Use **landmarks and camera orientation** to describe locations (e.g., “enters from the left side of the frame,” “approaches the trash bin near the driveway”).
    - Include **time context** if inferable (e.g., “at night,” “in daylight”).
- **Objective Surveillance Tone**:
    - Use a **neutral, concise, and factual tone**.
    - **Avoid speculation or storytelling**. Do not infer intent or emotions beyond visible behavior.
"""


class SecurityVideoAnalyzer:
    """
    A class to handle video inference using the Qwen2.5-VL security model.
    """

    def __init__(self, model_id: str = MODEL_ID, use_flash_attn: bool = True):
        """
        Initialize the analyzer by loading the model and processor.
        
        Args:
            model_id (str): The Hugging Face model ID.
            use_flash_attn (bool): Whether to use Flash Attention 2 if available.
        """
        print(f"Loading model from: {model_id}...")
        
        # Configure Attention implementation
        attn_impl = "flash_attention_2" if use_flash_attn and torch.cuda.is_available() else "eager"
        torch_dtype = torch.bfloat16 if use_flash_attn and torch.cuda.is_available() else "auto"

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print(f"Model loaded successfully. Attention mechanism: {attn_impl}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def analyze(self, video_path: str, max_new_tokens: int = 768) -> str:
        """
        Run inference on a specific video file.

        Args:
            video_path (str): Path to the input video file.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The generated video description/caption.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # Construct messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path
                    },
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ]

        # Preprocess inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        # Move inputs to device (handles 'auto' device map implicitly via tensor movement)
        inputs = inputs.to(self.model.device)

        # Inference
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Post-processing: Trim input tokens from output to get only the generation
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL Security Video Analysis")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file to analyze")
    parser.add_argument("--no_flash_attn", action="store_true", help="Disable Flash Attention 2")
    
    args = parser.parse_args()

    # Determine if Flash Attention should be used
    use_flash = not args.no_flash_attn

    try:
        # Initialize Analyzer
        analyzer = SecurityVideoAnalyzer(use_flash_attn=use_flash)
        
        # Run Analysis
        print(f"\nAnalyzing video: {args.video_path}\n" + "-"*30)
        result = analyzer.analyze(args.video_path)
        
        print("\n[Analysis Result]:\n")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
