# modal_app.py
import modal
import os
import sys

# Add the current directory to Python's path so imports work correctly
# This assumes modal_app.py is at the root of the ltx-video-distilled project
project_root = os.path.abspath(".") 
sys.path.insert(0, project_root)

stub = modal.Stub("ltx-video-distilled-app")

# Define the Modal Image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    .apt_install("ffmpeg")  # For imageio-ffmpeg
    .copy_local_dir(project_root, "/app") # Copy all project files
    .workdir("/app")
)

@stub.cls(image=image, gpu="A10G", timeout=1200, container_idle_timeout=300)
class Model:
    def __init__(self):
        import yaml
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from inference import create_ltx_video_pipeline, create_latent_upsampler

        self.config_file_path = "configs/ltxv-13b-0.9.7-distilled.yaml"
        with open(self.config_file_path, "r") as file:
            self.pipeline_config_yaml = yaml.safe_load(file)

        self.ltx_repo = "Lightricks/LTX-Video"
        self.models_dir = "/models_cache" # Use a path inside the container
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)

        print("Downloading models (if not present)...")
        distilled_model_actual_path = hf_hub_download(
            repo_id=self.ltx_repo,
            filename=self.pipeline_config_yaml["checkpoint_path"],
            local_dir=self.models_dir,
            local_dir_use_symlinks=False,
        )
        self.pipeline_config_yaml["checkpoint_path"] = distilled_model_actual_path
        print(f"Distilled model path: {distilled_model_actual_path}")

        spatial_upscaler_filename = self.pipeline_config_yaml["spatial_upscaler_model_path"]
        spatial_upscaler_actual_path = hf_hub_download(
            repo_id=self.ltx_repo,
            filename=spatial_upscaler_filename,
            local_dir=self.models_dir,
            local_dir_use_symlinks=False,
        )
        self.pipeline_config_yaml["spatial_upscaler_model_path"] = spatial_upscaler_actual_path
        print(f"Spatial upscaler model path: {spatial_upscaler_actual_path}")

        self.target_inference_device = "cuda" # Will be CUDA inside Modal GPU container

        print(f"Creating LTX Video pipeline on {self.target_inference_device}...")
        self.pipeline_instance = create_ltx_video_pipeline(
            ckpt_path=self.pipeline_config_yaml["checkpoint_path"],
            precision=self.pipeline_config_yaml["precision"],
            text_encoder_model_name_or_path=self.pipeline_config_yaml["text_encoder_model_name_or_path"],
            sampler=self.pipeline_config_yaml["sampler"],
            device=self.target_inference_device,
            enhance_prompt=False, # Set to False for simplicity, can be enabled
            prompt_enhancer_image_caption_model_name_or_path=self.pipeline_config_yaml.get("prompt_enhancer_image_caption_model_name_or_path"),
            prompt_enhancer_llm_model_name_or_path=self.pipeline_config_yaml.get("prompt_enhancer_llm_model_name_or_path"),
        )
        print("LTX Video pipeline created.")

        self.latent_upsampler_instance = None
        if self.pipeline_config_yaml.get("spatial_upscaler_model_path"):
            print(f"Creating latent upsampler on {self.target_inference_device}...")
            self.latent_upsampler_instance = create_latent_upsampler(
                self.pipeline_config_yaml["spatial_upscaler_model_path"],
                device=self.target_inference_device
            )
            print("Latent upsampler created.")

    @modal.method()
    def generate(self, 
                 prompt, negative_prompt, input_image_filepath, input_video_filepath,
                 height_ui, width_ui, mode,
                 duration_ui, 
                 ui_frames_to_use,
                 seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag):
        import torch
        import random
        import numpy as np
        import imageio
        import tempfile
        from inference import (
            load_image_to_tensor_with_resize_and_crop,
            seed_everething,
            calculate_padding,
            load_media_file
        )
        from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXMultiScalePipeline
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

        MAX_NUM_FRAMES = 257 # From app.py
        FPS = 30.0 # From app.py
        
        # --- Logic from app.py's generate function ---
        if randomize_seed:
            seed_ui = random.randint(0, 2**32 - 1)
        seed_everething(int(seed_ui))
        
        target_frames_ideal = duration_ui * FPS
        target_frames_rounded = round(target_frames_ideal)
        if target_frames_rounded < 1: 
            target_frames_rounded = 1
        
        n_val = round((float(target_frames_rounded) - 1.0) / 8.0)
        actual_num_frames = int(n_val * 8 + 1)

        actual_num_frames = max(9, actual_num_frames)
        actual_num_frames = min(MAX_NUM_FRAMES, actual_num_frames)
        
        actual_height = int(height_ui)
        actual_width = int(width_ui)

        height_padded = ((actual_height - 1) // 32 + 1) * 32
        width_padded = ((actual_width - 1) // 32 + 1) * 32
        num_frames_padded = ((actual_num_frames - 2) // 8 + 1) * 8 + 1 
        if num_frames_padded != actual_num_frames:
            print(f"Warning: actual_num_frames ({actual_num_frames}) and num_frames_padded ({num_frames_padded}) differ. Using num_frames_padded for pipeline.")
        
        padding_values = calculate_padding(actual_height, actual_width, height_padded, width_padded)

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height_padded,
            "width": width_padded,
            "num_frames": num_frames_padded, 
            "frame_rate": int(FPS), 
            "generator": torch.Generator(device=self.target_inference_device).manual_seed(int(seed_ui)),
            "output_type": "pt", 
            "conditioning_items": None,
            "media_items": None,
            "decode_timestep": self.pipeline_config_yaml["decode_timestep"],
            "decode_noise_scale": self.pipeline_config_yaml["decode_noise_scale"],
            "stochastic_sampling": self.pipeline_config_yaml["stochastic_sampling"],
            "image_cond_noise_scale": 0.15,
            "is_video": True,
            "vae_per_channel_normalize": True,
            "mixed_precision": (self.pipeline_config_yaml["precision"] == "mixed_precision"),
            "offload_to_cpu": False, # Not relevant in Modal single function context
            "enhance_prompt": False, # Keep False or implement prompt enhancement logic here
        }

        stg_mode_str = self.pipeline_config_yaml.get("stg_mode", "attention_values")
        if stg_mode_str.lower() in ["stg_av", "attention_values"]:
            call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.AttentionValues
        elif stg_mode_str.lower() in ["stg_as", "attention_skip"]:
            call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.AttentionSkip
        elif stg_mode_str.lower() in ["stg_r", "residual"]:
            call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.Residual
        elif stg_mode_str.lower() in ["stg_t", "transformer_block"]:
            call_kwargs["skip_layer_strategy"] = SkipLayerStrategy.TransformerBlock
        else:
            raise ValueError(f"Invalid stg_mode: {stg_mode_str}")

        if mode == "image-to-video" and input_image_filepath:
            try:
                media_tensor = load_image_to_tensor_with_resize_and_crop(
                    input_image_filepath, actual_height, actual_width
                )
                media_tensor = torch.nn.functional.pad(media_tensor, padding_values)
                call_kwargs["conditioning_items"] = [ConditioningItem(media_tensor.to(self.target_inference_device), 0, 1.0)]
            except Exception as e:
                print(f"Error loading image {input_image_filepath}: {e}")
                raise RuntimeError(f"Could not load image: {e}") # Use RuntimeError for Modal
        elif mode == "video-to-video" and input_video_filepath:
            try:
                call_kwargs["media_items"] = load_media_file(
                    media_path=input_video_filepath,
                    height=actual_height, 
                    width=actual_width,
                    max_frames=int(ui_frames_to_use), 
                    padding=padding_values
                ).to(self.target_inference_device)
            except Exception as e:
                print(f"Error loading video {input_video_filepath}: {e}")
                raise RuntimeError(f"Could not load video: {e}")

        active_latent_upsampler = None
        if improve_texture_flag and self.latent_upsampler_instance:
            active_latent_upsampler = self.latent_upsampler_instance

        result_images_tensor = None
        if improve_texture_flag:
            if not active_latent_upsampler:
                raise RuntimeError("Spatial upscaler model not loaded or improve_texture not selected, cannot use multi-scale.")
            
            multi_scale_pipeline_obj = LTXMultiScalePipeline(self.pipeline_instance, active_latent_upsampler)
            
            first_pass_args = self.pipeline_config_yaml.get("first_pass", {}).copy()
            first_pass_args["guidance_scale"] = float(ui_guidance_scale) 
            first_pass_args.pop("num_inference_steps", None)

            second_pass_args = self.pipeline_config_yaml.get("second_pass", {}).copy()
            second_pass_args["guidance_scale"] = float(ui_guidance_scale)
            second_pass_args.pop("num_inference_steps", None)
            
            multi_scale_call_kwargs = call_kwargs.copy()
            multi_scale_call_kwargs.update({
                "downscale_factor": self.pipeline_config_yaml["downscale_factor"],
                "first_pass": first_pass_args,
                "second_pass": second_pass_args,
            })
            
            print(f"Calling multi-scale pipeline (eff. HxW: {actual_height}x{actual_width}, Frames: {actual_num_frames} -> Padded: {num_frames_padded}) on {self.target_inference_device}")
            result_images_tensor = multi_scale_pipeline_obj(**multi_scale_call_kwargs).images
        else:
            single_pass_call_kwargs = call_kwargs.copy()
            first_pass_config_from_yaml = self.pipeline_config_yaml.get("first_pass", {})

            single_pass_call_kwargs["timesteps"] = first_pass_config_from_yaml.get("timesteps")
            single_pass_call_kwargs["guidance_scale"] = float(ui_guidance_scale)
            single_pass_call_kwargs["stg_scale"] = first_pass_config_from_yaml.get("stg_scale")
            single_pass_call_kwargs["rescaling_scale"] = first_pass_config_from_yaml.get("rescaling_scale")
            single_pass_call_kwargs["skip_block_list"] = first_pass_config_from_yaml.get("skip_block_list")
            
            single_pass_call_kwargs.pop("num_inference_steps", None) 
            single_pass_call_kwargs.pop("first_pass", None) 
            single_pass_call_kwargs.pop("second_pass", None)
            single_pass_call_kwargs.pop("downscale_factor", None)
            
            print(f"Calling base pipeline (padded HxW: {height_padded}x{width_padded}, Frames: {actual_num_frames} -> Padded: {num_frames_padded}) on {self.target_inference_device}")
            result_images_tensor = self.pipeline_instance(**single_pass_call_kwargs).images

        if result_images_tensor is None:
            raise RuntimeError("Generation failed.")

        pad_left, pad_right, pad_top, pad_bottom = padding_values
        slice_h_end = -pad_bottom if pad_bottom > 0 else None
        slice_w_end = -pad_right if pad_right > 0 else None
        
        result_images_tensor = result_images_tensor[
            :, :, :actual_num_frames, pad_top:slice_h_end, pad_left:slice_w_end
        ]

        video_np = result_images_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()
        
        video_np = np.clip(video_np, 0, 1) 
        video_np = (video_np * 255).astype(np.uint8)

        temp_dir = tempfile.mkdtemp()
        timestamp = random.randint(10000,99999)
        output_video_path = os.path.join(temp_dir, f"output_{timestamp}.mp4")
        
        try:
            with imageio.get_writer(output_video_path, fps=call_kwargs["frame_rate"], macro_block_size=1) as video_writer:
                for frame_idx in range(video_np.shape[0]):
                    # No progress bar in modal method directly, Gradio handles it
                    video_writer.append_data(video_np[frame_idx])
        except Exception as e:
            print(f"Error saving video with macro_block_size=1: {e}")
            try:
                with imageio.get_writer(output_video_path, fps=call_kwargs["frame_rate"], format='FFMPEG', codec='libx264', quality=8) as video_writer:
                     for frame_idx in range(video_np.shape[0]):
                        video_writer.append_data(video_np[frame_idx])
            except Exception as e2:
                print(f"Fallback video saving error: {e2}")
                raise RuntimeError(f"Failed to save video: {e2}")
                
        return output_video_path, seed_ui

# --- Gradio UI Served by Modal ---
# We need to load the YAML config here for UI defaults, outside the Model class
# to avoid instantiating Model just for UI setup.
import yaml
with open("configs/ltxv-13b-0.9.7-distilled.yaml", "r") as file:
    INITIAL_PIPELINE_CONFIG_YAML = yaml.safe_load(file)

MAX_IMAGE_SIZE = INITIAL_PIPELINE_CONFIG_YAML.get("max_resolution", 1280)
MAX_NUM_FRAMES = 257 # From app.py
MIN_DIM_SLIDER = 256
TARGET_FIXED_SIDE = 768

# Helper functions from app.py (can be kept outside the Modal class)
def calculate_new_dimensions(orig_w, orig_h):
    if orig_w == 0 or orig_h == 0:
        return int(TARGET_FIXED_SIDE), int(TARGET_FIXED_SIDE)
    if orig_w >= orig_h:
        new_h = TARGET_FIXED_SIDE
        aspect_ratio = orig_w / orig_h
        new_w_ideal = new_h * aspect_ratio
        new_w = round(new_w_ideal / 32) * 32
        new_w = max(MIN_DIM_SLIDER, min(new_w, MAX_IMAGE_SIZE))
        new_h = max(MIN_DIM_SLIDER, min(new_h, MAX_IMAGE_SIZE)) 
    else:
        new_w = TARGET_FIXED_SIDE
        aspect_ratio = orig_h / orig_w
        new_h_ideal = new_w * aspect_ratio
        new_h = round(new_h_ideal / 32) * 32
        new_h = max(MIN_DIM_SLIDER, min(new_h, MAX_IMAGE_SIZE))
        new_w = max(MIN_DIM_SLIDER, min(new_w, MAX_IMAGE_SIZE))
    return int(new_h), int(new_w)

@stub.asgi_app()
def web_app():
    import gradio as gr
    from PIL import Image
    import imageio # For video dim reading

    # --- Gradio UI Definition ---
    # Most of this is copied from app.py, with generate calls changed
    css="""
    #col-container {
        margin: 0 auto;
        max-width: 900px;
    }
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# LTX Video 0.9.7 Distilled (via Modal)")
        gr.Markdown("Fast high quality video generation. [Model](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors) [GitHub](https://github.com/Lightricks/LTX-Video) [Diffusers](#)")
        
        with gr.Row():
            with gr.Column():
                with gr.Tab("image-to-video") as image_tab:
                    video_i_hidden = gr.Textbox(label="video_i", visible=False, value=None)
                    image_i2v = gr.Image(label="Input Image", type="filepath", sources=["upload", "webcam", "clipboard"])
                    i2v_prompt = gr.Textbox(label="Prompt", value="The creature from the image starts to move", lines=3)
                    i2v_button = gr.Button("Generate Image-to-Video", variant="primary")
                with gr.Tab("text-to-video") as text_tab:
                    image_n_hidden = gr.Textbox(label="image_n", visible=False, value=None)
                    video_n_hidden = gr.Textbox(label="video_n", visible=False, value=None)
                    t2v_prompt = gr.Textbox(label="Prompt", value="A majestic dragon flying over a medieval castle", lines=3)
                    t2v_button = gr.Button("Generate Text-to-Video", variant="primary")
                with gr.Tab("video-to-video", visible=True) as video_tab: # Made visible for testing
                    image_v_hidden = gr.Textbox(label="image_v", visible=False, value=None)
                    video_v2v = gr.Video(label="Input Video", sources=["upload"]) # type defaults to filepath
                    frames_to_use = gr.Slider(label="Frames to use from input video", minimum=9, maximum=MAX_NUM_FRAMES, value=9, step=8, info="Number of initial frames to use for conditioning/transformation. Must be N*8+1.")
                    v2v_prompt = gr.Textbox(label="Prompt", value="Change the style to cinematic anime", lines=3)
                    v2v_button = gr.Button("Generate Video-to-Video", variant="primary")

                duration_input = gr.Slider(
                    label="Video Duration (seconds)", 
                    minimum=0.3, 
                    maximum=8.5, # Max from app.py
                    value=2,  
                    step=0.1, 
                    info=f"Target video duration (0.3s to 8.5s)"
                )
                improve_texture = gr.Checkbox(label="Improve Texture (multi-scale)", value=True, info="Uses a two-pass generation for better quality, but is slower. Recommended for final output.")

            with gr.Column():
                output_video = gr.Video(label="Generated Video", interactive=False)

        with gr.Accordion("Advanced settings", open=False):
            mode = gr.Dropdown(["text-to-video", "image-to-video", "video-to-video"], label="task", value="image-to-video", visible=False)
            negative_prompt_input = gr.Textbox(label="Negative Prompt", value="worst quality, inconsistent motion, blurry, jittery, distorted", lines=2)
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=2**32-1)
                randomize_seed_input = gr.Checkbox(label="Randomize Seed", value=True)
            with gr.Row():
                guidance_scale_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=10.0, value=INITIAL_PIPELINE_CONFIG_YAML.get("first_pass", {}).get("guidance_scale", 1.0), step=0.1, info="Controls how much the prompt influences the output. Higher values = stronger influence.")
            with gr.Row():
                height_input = gr.Slider(label="Height", value=512, step=32, minimum=MIN_DIM_SLIDER, maximum=MAX_IMAGE_SIZE, info="Must be divisible by 32.")
                width_input = gr.Slider(label="Width", value=704, step=32, minimum=MIN_DIM_SLIDER, maximum=MAX_IMAGE_SIZE, info="Must be divisible by 32.")

        # --- Event handlers for updating dimensions on upload (copied from app.py) ---
        def handle_image_upload_for_dims_gr(image_filepath, current_h, current_w):
            if not image_filepath:
                return gr.update(value=current_h), gr.update(value=current_w)
            try:
                img = Image.open(image_filepath)
                orig_w, orig_h = img.size
                new_h, new_w = calculate_new_dimensions(orig_w, orig_h)
                return gr.update(value=new_h), gr.update(value=new_w)
            except Exception as e:
                print(f"Error processing image for dimension update: {e}")
                return gr.update(value=current_h), gr.update(value=current_w)

        def handle_video_upload_for_dims_gr(video_filepath, current_h, current_w):
            if not video_filepath:
                return gr.update(value=current_h), gr.update(value=current_w)
            try:
                video_filepath_str = str(video_filepath) 
                if not os.path.exists(video_filepath_str):
                    print(f"Video file path does not exist for dimension update: {video_filepath_str}")
                    return gr.update(value=current_h), gr.update(value=current_w)

                orig_w, orig_h = -1, -1
                with imageio.get_reader(video_filepath_str) as reader:
                    meta = reader.get_meta_data()
                    if 'size' in meta:
                        orig_w, orig_h = meta['size']
                    else:
                        try:
                            first_frame = reader.get_data(0)
                            orig_h, orig_w = first_frame.shape[0], first_frame.shape[1]
                        except Exception as e_frame:
                            print(f"Could not get video size from metadata or first frame: {e_frame}")
                            return gr.update(value=current_h), gr.update(value=current_w)
                if orig_w == -1 or orig_h == -1:
                     print(f"Could not determine dimensions for video: {video_filepath_str}")
                     return gr.update(value=current_h), gr.update(value=current_w)
                new_h, new_w = calculate_new_dimensions(orig_w, orig_h)
                return gr.update(value=new_h), gr.update(value=new_w)
            except Exception as e:
                print(f"Error processing video for dimension update: {e} (Path: {video_filepath}, Type: {type(video_filepath)})")
                return gr.update(value=current_h), gr.update(value=current_w)

        image_i2v.upload(
            fn=handle_image_upload_for_dims_gr,
            inputs=[image_i2v, height_input, width_input],
            outputs=[height_input, width_input]
        )
        video_v2v.upload(
            fn=handle_video_upload_for_dims_gr,
            inputs=[video_v2v, height_input, width_input],
            outputs=[height_input, width_input]
        )

        # --- Mode updates ---
        def update_task_image(): return "image-to-video"
        def update_task_text(): return "text-to-video"
        def update_task_video(): return "video-to-video"

        image_tab.select(fn=update_task_image, outputs=[mode])
        text_tab.select(fn=update_task_text, outputs=[mode])
        video_tab.select(fn=update_task_video, outputs=[mode]) # Added for v2v

        # --- Main generate function for Gradio to call Modal ---
        _model = Model() # Instantiate Modal class once
        
        def gradio_generate_wrapper(prompt, negative_prompt, input_image_filepath, input_video_filepath,
                                    height_ui, width_ui, mode_val,
                                    duration_ui, ui_frames_to_use,
                                    seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag,
                                    progress=gr.Progress(track_tqdm=True)):
            # Gradio progress will show "Running..." as it calls the remote Modal function
            # Actual TQDM progress from imageio will appear in Modal container logs.
            try:
                # Handle cases where Gradio might pass None for hidden file inputs
                input_image_filepath = input_image_filepath if input_image_filepath else None
                input_video_filepath = input_video_filepath if input_video_filepath else None

                output_path, seed = _model.generate.remote(
                    prompt, negative_prompt, input_image_filepath, input_video_filepath,
                    height_ui, width_ui, mode_val,
                    duration_ui, ui_frames_to_use,
                    seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag
                )
                return output_path, seed
            except Exception as e:
                # Catch Modal's remote exceptions (which might be wrapped)
                # or any other error during the call.
                print(f"Error during Modal call: {e}")
                import traceback
                traceback.print_exc()
                # Raise gr.Error to display it nicely in the Gradio UI
                raise gr.Error(f"Generation failed: {str(e)}")


        # --- Button click handlers ---
        t2v_inputs = [t2v_prompt, negative_prompt_input, image_n_hidden, video_n_hidden,
                      height_input, width_input, mode,
                      duration_input, frames_to_use, 
                      seed_input, randomize_seed_input, guidance_scale_input, improve_texture]
        
        i2v_inputs = [i2v_prompt, negative_prompt_input, image_i2v, video_i_hidden,
                      height_input, width_input, mode,
                      duration_input, frames_to_use, 
                      seed_input, randomize_seed_input, guidance_scale_input, improve_texture]

        v2v_inputs = [v2v_prompt, negative_prompt_input, image_v_hidden, video_v2v,
                      height_input, width_input, mode,
                      duration_input, frames_to_use, 
                      seed_input, randomize_seed_input, guidance_scale_input, improve_texture]

        # Use the wrapper for Gradio button clicks
        t2v_button.click(fn=gradio_generate_wrapper, inputs=t2v_inputs, outputs=[output_video, seed_input], api_name="text_to_video")
        i2v_button.click(fn=gradio_generate_wrapper, inputs=i2v_inputs, outputs=[output_video, seed_input], api_name="image_to_video")
        v2v_button.click(fn=gradio_generate_wrapper, inputs=v2v_inputs, outputs=[output_video, seed_input], api_name="video_to_video")

    return demo.queue().launch(share=False, show_error=True) # Important: share=False for Modal
