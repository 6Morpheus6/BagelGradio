# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Copyright 2025 [Your Name/Entity]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# THIS FILE HAS BEEN MODIFIED FROM ITS ORIGINAL VERSION OR IS A NEW FILE ADDED TO THE PROJECT.
# Modifications by ansmom, May 2025:
# - Added Gradio UI elements for image saving functionality:
#   - Checkbox to enable/disable saving.
#   - Textbox for specifying output directory.
#   - Text display for save status.
# - Updated Gradio button click handlers (`run_inference_step_by_step`) to:
#   - Accept `save_images` and `output_dir` parameters from the UI.
#   - Pass these parameters to the `inferencer.interleave_inference` method.
#   - Return and display the `saved_image_path` or error message.
# - Modified `reset_context` to clear the save status display.
import gradio as gr
import torch
from PIL import Image
import sys
import os
import argparse

parser = argparse.ArgumentParser(description="BAGEL Interleaved Inference UI")
parser.add_argument(
    "--model_folder", "-m",
    type=str,
    default=os.environ.get("BAGEL_MODEL_PATH", os.path.join(os.path.dirname(__file__), "ckpt")),
    help="Pfad zum Modell-Ordner (default: ./ckpt oder ENV VAR BAGEL_MODEL_PATH)"
)
parser.add_argument(
    "--offload_folder", "-o",
    type=str,
    default=os.environ.get("BAGEL_OFFLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "offload_data")),
    help="Pfad zum Offload-Ordner (default: ./offload_data oder ENV VAR BAGEL_OFFLOAD_FOLDER)"
)
args = parser.parse_args()

# Add the parent directory to the sys.path to import app and inferencer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs(args.offload_folder, exist_ok=True)

os.environ["BAGEL_MODEL_PATH"] = args.model_folder

from app import load_model

# Load the model and inferencer
inferencer = load_model(offload_folder=args.offload_folder)

def run_inference(text_input, image_input, current_context, inference_type, temperature, max_length, cfg_text_scale, cfg_img_scale, cfg_interval_start, cfg_interval_end, num_timesteps, timestep_shift, cfg_renorm_min, cfg_renorm_type):
    # Initialize context if it's the first turn or reset
    if current_context is None:
        current_context = inferencer.init_gen_context()

    input_list = []
    if image_input is not None:
        input_list.append(image_input)
    if text_input:
        input_list.append(text_input)

    if not input_list:
        return current_context, "Please provide text or image input.", None

    understanding_output = False
    if inference_type == "Generate Text":
        understanding_output = True
    elif inference_type == "Generate Image":
        understanding_output = False
    elif inference_type == "Interleaved Inference":
        # Determine output type based on the last input
        if isinstance(input_list[-1], str):
            understanding_output = True
        elif isinstance(input_list[-1], Image.Image):
            understanding_output = False

    # Update context with current inputs
    # The interleave_inference function handles context updates internally
    # We just need to pass the current_context and the new inputs

    output = inferencer.interleave_inference(
        input_lists=input_list,
        understanding_output=understanding_output,
        text_temperature=temperature,
        max_length=max_length, # This parameter is not directly in interleave_inference, need to check inferencer.py again
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval_start, cfg_interval_end],
        num_timesteps=num_timesteps,
        timestep_shift=timestep_shift,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
    )

    # The interleave_inference returns a list of outputs.
    # The last element is the final output (text or image).
    # The context is updated internally within interleave_inference, but we need to get the updated context back.
    # Looking at inferencer.py, interleave_inference doesn't return the updated context.
    # This means we need to update the context step-by-step using update_context_text and update_context_image
    # before calling gen_text or gen_image.

    # Let's refactor the inference logic to update context first, then generate.

    return current_context, output[0] if isinstance(output[0], str) else "", output[0] if isinstance(output[0], Image.Image) else None


def run_inference_step_by_step(text_input, image_input, current_context, inference_type, temperature, max_length, cfg_text_scale, cfg_img_scale, cfg_interval_start, cfg_interval_end, num_timesteps, timestep_shift, cfg_renorm_min, cfg_renorm_type, save_images, output_dir):
    # Initialize context if it's the first turn or reset
    if current_context is None:
        current_context = inferencer.init_gen_context()

    input_list = []
    if image_input is not None:
        input_list.append(image_input)
    if text_input:
        input_list.append(text_input)

    if not input_list:
        return current_context, "Please provide text or image input.", None

    understanding_output = False
    if inference_type == "Generate Text":
        understanding_output = True
    elif inference_type == "Generate Image":
        understanding_output = False
    elif inference_type == "Interleaved Inference":
        # Determine output type based on the last input
        if isinstance(input_list[-1], str):
            understanding_output = True
        elif isinstance(input_list[-1], Image.Image):
            understanding_output = False

    # Always call interleave_inference and handle the returned context
    output_list, updated_context, saved_image_path = inferencer.interleave_inference(
        input_lists=input_list,
        understanding_output=understanding_output,
        max_length=max_length,
        text_temperature=temperature,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=[cfg_interval_start, cfg_interval_end],
        num_timesteps=num_timesteps,
        timestep_shift=timestep_shift,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        save_image=save_images,
        output_dir=output_dir
    )

    text_output = ""
    image_output = None
    status_message = "Output generated."

    if saved_image_path:
        if "Error" in saved_image_path:
            status_message = saved_image_path
        else:
            status_message = f"Image saved to: {saved_image_path}"
    elif save_images and not understanding_output: # If save was intended for an image but no path returned
        status_message = "Image generation selected, but no image was saved (or an error occurred silently)."


    # Process the output list
    for output_item in output_list:
        if isinstance(output_item, str):
            text_output += output_item + "\n" # Append text outputs
        elif isinstance(output_item, Image.Image):
            image_output = output_item # Assume only one image output for simplicity in UI

    # Return the updated context and outputs
    return updated_context, text_output, image_output, status_message

def reset_context():
    return inferencer.init_gen_context(), "", None, "" # Reset context and clear outputs, including save status

with gr.Blocks() as demo:
    gr.Markdown("# BAGEL Interleaved Inference UI")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text Input", lines=5)
            image_input = gr.Image(type="pil", label="Image Input")
            
            with gr.Accordion("Parameters", open=False):
                temperature_slider = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature")
                max_length_slider = gr.Slider(minimum=50, maximum=2000, value=500, step=1, label="Max Length (for text generation)")
                cfg_text_scale_slider = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, label="CFG Text Scale")
                cfg_img_scale_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.5, label="CFG Image Scale")
                cfg_interval_start_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.4, label="CFG Interval Start")
                cfg_interval_end_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="CFG Interval End")
                num_timesteps_slider = gr.Slider(minimum=10, maximum=200, value=50, step=1, label="Number of Timesteps (for image generation)")
                timestep_shift_slider = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, label="Timestep Shift (for image generation)")
                cfg_renorm_min_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="CFG Renorm Min (for image generation)")
                cfg_renorm_type_radio = gr.Radio(["global", "local"], value="global", label="CFG Renorm Type (for image generation)")

            with gr.Accordion("Image Saving", open=False):
                save_images_checkbox = gr.Checkbox(label="Save generated images", value=False)
                output_dir_textbox = gr.Textbox(label="Output Directory", value="./generated_images")

            with gr.Row():
                generate_text_btn = gr.Button("Generate Text")
                generate_image_btn = gr.Button("Generate Image")
                interleaved_inference_btn = gr.Button("Interleaved Inference")

            reset_btn = gr.Button("Reset Context")

        with gr.Column():
            text_output = gr.Textbox(label="Text Output", lines=10)
            image_output = gr.Image(label="Image Output")
            save_status_text = gr.Textbox(label="Save Status", interactive=False)

    # State to maintain context
    gen_context_state = gr.State(None)

    generate_text_btn.click(
        fn=run_inference_step_by_step,
        inputs=[
            text_input,
            image_input,
            gen_context_state,
            gr.State("Generate Text"),
            temperature_slider,
            max_length_slider,
            cfg_text_scale_slider,
            cfg_img_scale_slider,
            cfg_interval_start_slider,
            cfg_interval_end_slider,
            num_timesteps_slider,
            timestep_shift_slider,
            cfg_renorm_min_slider,
            cfg_renorm_type_radio,
            save_images_checkbox,
            output_dir_textbox,
        ],
        outputs=[gen_context_state, text_output, image_output, save_status_text]
    )

    generate_image_btn.click(
        fn=run_inference_step_by_step,
        inputs=[
            text_input,
            image_input,
            gen_context_state,
            gr.State("Generate Image"),
            temperature_slider,
            max_length_slider,
            cfg_text_scale_slider,
            cfg_img_scale_slider,
            cfg_interval_start_slider,
            cfg_interval_end_slider,
            num_timesteps_slider,
            timestep_shift_slider,
            cfg_renorm_min_slider,
            cfg_renorm_type_radio,
            save_images_checkbox,
            output_dir_textbox,
        ],
        outputs=[gen_context_state, text_output, image_output, save_status_text]
    )

    interleaved_inference_btn.click(
        fn=run_inference_step_by_step,
        inputs=[
            text_input,
            image_input,
            gen_context_state,
            gr.State("Interleaved Inference"),
            temperature_slider,
            max_length_slider,
            cfg_text_scale_slider,
            cfg_img_scale_slider,
            cfg_interval_start_slider,
            cfg_interval_end_slider,
            num_timesteps_slider,
            timestep_shift_slider,
            cfg_renorm_min_slider,
            cfg_renorm_type_radio,
            save_images_checkbox,
            output_dir_textbox,
        ],
        outputs=[gen_context_state, text_output, image_output, save_status_text]
    )

    reset_btn.click(
        fn=reset_context,
        inputs=[],
        outputs=[gen_context_state, text_output, image_output, save_status_text]
    )


if __name__ == "__main__":
    demo.launch()
