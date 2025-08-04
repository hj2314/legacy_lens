"""
Generates a 'vault.txt' file containing descriptions of media files.

This script processes a collection of local image and video files. For each file, 
it combines a manual, user-provided description with an AI-generated description 
from a local Ollama model (gemma3n). The script extracts the first frame of videos 
for AI analysis.

The final output is 'vault.txt', where each line represents a media item (or a 
group of related items) and contains the combined descriptions along with a 
reference to the source file(s).

Key functionalities:
- Encodes images and video frames into base64 strings for model input.
- Uses the Ollama library to get AI-generated descriptions for visuals.
- Combines manual and AI descriptions into a structured text format.
- Writes the consolidated information into a 'vault.txt' file.
"""

import ollama
import os
from pathlib import Path
import cv2
import base64
import numpy as np

# --- CONFIGURATION ---
# Sets the model name and paths for input media.
MODEL_NAME = "gemma3n"
input_dir = "./data/" 
grandpa_jpg = os.path.join(input_dir, "grandpa.jpg")
grandpa_mp4 = os.path.join(input_dir, "grandpa.mp4") 
babywalk_mp4 = os.path.join(input_dir, "babyWalk.mp4")

# --- MANUAL DESCRIPTIONS ---
# User-provided context for each media file. This context is passed to the AI 
# to generate a more relevant and detailed description.
manual_grandpa_photo = "this is a photo from 1950 where grandpa is 8 years old in osaka japan he is on the far right with his friends."
manual_grandpa_audio = "when grandpa was young he played catchball and caught fish with his friends."
manual_babywalk = "this is the first time naoto walked on sep 4 2024"

def image_to_base64(image_path):
    """
    Converts an image file to a base64 encoded string.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: The base64 encoded string of the image, or None if an error occurs.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def frame_to_base64(frame):
    """
    Converts an OpenCV video frame to a base64 encoded string.

    The frame is first encoded as a JPEG image in memory before being converted.

    Args:
        frame (np.ndarray): The video frame captured by OpenCV.

    Returns:
        str: The base64 encoded string of the frame, or None if an error occurs.
    """
    try:
        # Encode frame as JPEG. This is a standard, efficient format for image data.
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert the in-memory buffer to a base64 string.
        encoded_string = base64.b64encode(buffer).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error converting frame to base64: {e}")
        return None

def get_image_description(image_path, context=""):
    """
    Generates a description for an image using a multimodal AI model.

    It sends the image (as a base64 string) and optional context to the model.
    Note: gemma3n via Ollama has a >75% tendency to hallucinate on certain hardware so results should be reviewed / 
    one should use Gemma3 with Ollama. Gemma3n image processing on kagglehub or huggingface yields good results. 

    Args:
        image_path (str): The file path to the image.
        context (str, optional): Manual description to provide context to the AI. 
                                 Defaults to "".

    Returns:
        str: The AI-generated description, or an error message.
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return ""
        
        base64_image = image_to_base64(image_path)
        if not base64_image:
            return "image conversion to base64 failed"
        
        # The prompt is engineered to ask the model to add new visual details,
        # avoiding repetition of information already provided in the context.
        if context:
            prompt = f"Given that {context}, describe what you see in this image succinctly in one sentence, focusing on visual details not already mentioned in the context."
        else:
            prompt = "Describe this image succinctly in one sentence"
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [base64_image]
                }
            ]
        )
        
        return response['message']['content'].strip()
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return "image description unavailable"

def extract_frame_and_describe(video_path, context=""):
    """
    Extracts the first frame of a video and generates a description for it.

    This function captures the very first frame, converts it to base64, and sends
    it to the AI model for description, similar to the image function.

    Args:
        video_path (str): The file path to the video.
        context (str, optional): Manual description to provide context to the AI. 
                                 Defaults to "".

    Returns:
        str: The AI-generated description of the first frame, or an error message.
    """
    try:
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return ""
            
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read() # ret is a boolean, True if frame was read successfully
        cap.release()
        
        if ret:
            base64_frame = frame_to_base64(frame)
            if not base64_frame:
                return "frame conversion to base64 failed"
            
            if context:
                prompt = f"Given that {context}, describe what you see in this image succinctly in one sentence, focusing on visual details not already mentioned in the context."
            else:
                prompt = "Describe this image succinctly in one sentence"
            
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64_frame]
                    }
                ]
            )
            
            return response['message']['content'].strip()
        
        return "video frame extraction failed"
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return "video description unavailable"

# --- MAIN EXECUTION ---
# The main block of the script that orchestrates the processing.

print("="*60)
print("🔄 PROCESSING MEDIA FILES")
print("="*60)

# Generate AI descriptions for the specified media files.
# For the grandpa photo, it uses the manual photo description as context.
grandpa_photo_ai_desc = get_image_description(grandpa_jpg, manual_grandpa_photo)
print(f"AI description for grandpa.jpg: {grandpa_photo_ai_desc}")

# For the baby walking video, it uses the manual walk description as context.
babywalk_ai_desc = extract_frame_and_describe(babywalk_mp4, manual_babywalk)
print(f"AI description for babywalk.mp4: {babywalk_ai_desc}")

print("\n" + "="*60)
print("🔄 GENERATING VAULT ENTRIES")
print("="*60)

# Combine manual descriptions and the new AI description into a single string.
# The `[[...]]` syntax is a custom format to link the description to file names.
grandpa_vault_entry = f"{manual_grandpa_audio}, {manual_grandpa_photo}, {grandpa_photo_ai_desc}. [[grandpa.jpg,grandpa.mp4]]"
babywalk_vault_entry = f"{manual_babywalk}, {babywalk_ai_desc}. [[babywalk.mp4]]"
vault_content = [grandpa_vault_entry, babywalk_vault_entry]

# Write the combined entries into the vault.txt file.
with open('vault.txt', 'w', encoding='utf-8') as f:
    for entry in vault_content:
        f.write(entry + '\n')

print("\n" + "="*60)
print("🎉 VAULT.TXT GENERATED SUCCESSFULLY!")
print("="*60)

# Display the final content that was written to the file for verification.
print("\n📄 VAULT ENTRIES PREVIEW:")
print("-" * 60)
for i, entry in enumerate(vault_content, 1):
    print(f"Row {i}:")
    print(f"  {entry}")
    print()