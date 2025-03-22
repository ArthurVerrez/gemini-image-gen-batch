import os
import base64
import mimetypes
import tempfile
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv
import argparse
import asyncio
import concurrent.futures
import time
import datetime
import sys
import platform
import uuid
import shutil

# Load environment variables from .env file
load_dotenv()


# Define a utility function for logging with distinct markers
def log_marker(message, marker_type="INFO"):
    timestamp = datetime.datetime.now()
    if marker_type == "START":
        print("\n" + "=" * 80)
        print(f"[{timestamp}] 🟢 STARTING: {message}")
        print("=" * 80)
    elif marker_type == "END":
        print("\n" + "=" * 80)
        print(f"[{timestamp}] ✅ COMPLETED: {message}")
        print("=" * 80)
    elif marker_type == "ERROR":
        print("\n" + "=" * 80)
        print(f"[{timestamp}] 🔴 ERROR: {message}")
        print("=" * 80)
    elif marker_type == "WARNING":
        print("\n" + "-" * 80)
        print(f"[{timestamp}] ⚠️ WARNING: {message}")
        print("-" * 80)
    else:  # INFO
        print(f"[{timestamp}] ℹ️ {message}")


def log_system_info():
    """Log system information to help with debugging"""
    log_marker("System Information", "INFO")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")

    # Log relevant package versions
    try:
        import pkg_resources

        packages = ["gradio", "google-genai", "python-dotenv"]
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"Package {package}: v{version}")
            except pkg_resources.DistributionNotFound:
                print(f"Package {package}: Not installed")
    except ImportError:
        print("Unable to retrieve package versions")

    # Log environment variables (excluding sensitive ones)
    print("\nEnvironment variables:")
    for key in os.environ:
        if (
            "key" not in key.lower()
            and "secret" not in key.lower()
            and "password" not in key.lower()
        ):
            print(
                f"  {key}: {os.environ[key] if len(os.environ[key]) < 50 else '[LONG VALUE]'}"
            )


# Log startup info
log_marker("Starting Gemini Image Generation App", "START")
log_system_info()

# Ensure API key is available
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    log_marker("GEMINI_API_KEY environment variable is not set", "ERROR")
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it or create a .env file."
    )
else:
    log_marker("API key loaded successfully")


def save_binary_file(file_name, data):
    print(f"[{datetime.datetime.now()}] Saving binary file to: {file_name}")
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"[{datetime.datetime.now()}] File saved successfully: {file_name}")


def log_saved_images_summary(saved_images):
    """Log a summary of the saved images"""
    if not saved_images:
        log_marker("No images were saved in this generation run", "WARNING")
        return

    log_marker(f"Saved {len(saved_images)} images to outputs directory", "INFO")
    for i, img_path in enumerate(saved_images):
        print(f"[{datetime.datetime.now()}] Image {i+1}: {os.path.basename(img_path)}")

    # Get total size of saved images
    total_size = sum(os.path.getsize(img) for img in saved_images)
    print(
        f"[{datetime.datetime.now()}] Total size of saved images: {total_size / (1024*1024):.2f} MB"
    )


async def generate_image_async(prompt, uploaded_images, num_runs=1):
    print(
        f"[{datetime.datetime.now()}] Starting async generation with prompt: '{prompt}', num_runs: {num_runs}"
    )
    print(
        f"[{datetime.datetime.now()}] Number of provided reference images: {len([img for img in uploaded_images if img is not None])}"
    )

    start_time = time.time()
    # Create a thread pool executor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        print(
            f"[{datetime.datetime.now()}] Created thread pool executor with max_workers=8"
        )
        # Schedule the generation tasks
        tasks = []
        for i in range(num_runs):
            print(
                f"[{datetime.datetime.now()}] Scheduling generation task {i+1}/{num_runs}"
            )
            tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    executor, generate_single_image, prompt, uploaded_images, i + 1
                )
            )

        # Wait for all tasks to complete
        print(
            f"[{datetime.datetime.now()}] Waiting for all {len(tasks)} tasks to complete..."
        )
        results = await asyncio.gather(*tasks)
        print(
            f"[{datetime.datetime.now()}] All tasks completed in {time.time() - start_time:.2f} seconds"
        )

        # Separate results
        response_texts = []
        output_images = []
        saved_images = []

        for text, temp_img, saved_img in results:
            response_texts.append(text)
            output_images.append(temp_img)
            saved_images.append(saved_img)

        # Log summary of saved images
        valid_saved_images = [img for img in saved_images if img is not None]
        log_saved_images_summary(valid_saved_images)

        print(
            f"[{datetime.datetime.now()}] Generated {len(output_images)} images and {len(response_texts)} text responses"
        )
        return response_texts, output_images, saved_images


def generate_single_image(prompt, uploaded_images, run_number=1):
    gen_start_time = time.time()
    log_marker(f"Task {run_number}: Starting generation", "INFO")
    print(f"[{datetime.datetime.now()}] Task {run_number}: Prompt: '{prompt}'")

    # Filter out None values from uploaded_images
    uploaded_images = [img for img in uploaded_images if img is not None]
    print(
        f"[{datetime.datetime.now()}] Task {run_number}: Using {len(uploaded_images)} reference images"
    )

    if not prompt:
        log_marker(f"Task {run_number}: No prompt provided", "ERROR")
        return "Please provide a prompt text", None, None

    try:
        print(f"[{datetime.datetime.now()}] Task {run_number}: Creating Gemini client")
        client = genai.Client(
            api_key=api_key,
        )

        # Upload the input images if provided
        files = []
        for i, img in enumerate(uploaded_images):
            print(
                f"[{datetime.datetime.now()}] Task {run_number}: Uploading reference image {i+1}/{len(uploaded_images)}"
            )
            try:
                file = client.files.upload(file=img)
                files.append(file)
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Successfully uploaded image {i+1}, URI: {file.uri}"
                )
            except Exception as e:
                log_marker(
                    f"Task {run_number}: Error uploading image {i+1}: {str(e)}",
                    "WARNING",
                )
                # Continue with other images

        model = "gemini-2.0-flash-exp-image-generation"
        print(f"[{datetime.datetime.now()}] Task {run_number}: Using model: {model}")

        # Build the content parts
        parts = []
        for i, file in enumerate(files):
            print(
                f"[{datetime.datetime.now()}] Task {run_number}: Adding image part {i+1}/{len(files)}"
            )
            parts.append(
                types.Part.from_uri(
                    file_uri=file.uri,
                    mime_type=file.mime_type,
                )
            )
        print(f"[{datetime.datetime.now()}] Task {run_number}: Adding text prompt part")
        parts.append(types.Part.from_text(text=prompt))

        contents = [
            types.Content(
                role="user",
                parts=parts,
            )
        ]

        print(
            f"[{datetime.datetime.now()}] Task {run_number}: Configuring generation parameters"
        )
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_modalities=[
                "image",
                "text",
            ],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="OFF",  # Off
                ),
            ],
            response_mime_type="text/plain",
        )

        # Create a temporary file to store the output
        temp_output_path = None
        saved_output_path = None
        response_text = ""

        print(
            f"[{datetime.datetime.now()}] Task {run_number}: Starting Gemini content generation stream"
        )
        api_call_start = time.time()
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                not chunk.candidates
                or not chunk.candidates[0].content
                or not chunk.candidates[0].content.parts
            ):
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Received empty chunk, skipping"
                )
                continue

            if chunk.candidates[0].content.parts[0].inline_data:
                # Image response
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Received image response"
                )
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Image mime type: {inline_data.mime_type}, extension: {file_extension}"
                )

                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                )
                temp_output_path = temp_file.name
                temp_file.close()
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Created temporary file: {temp_output_path}"
                )

                # Save the binary data
                save_binary_file(temp_output_path, inline_data.data)
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Saved image data to temporary file"
                )

                # Also save a copy to the outputs directory with a unique filename
                output_path = os.path.join(
                    output_directory, generate_unique_filename(file_extension)
                )
                shutil.copy2(temp_output_path, output_path)
                saved_output_path = output_path
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Saved permanent copy to: {output_path}"
                )
            else:
                # Text response
                print(
                    f"[{datetime.datetime.now()}] Task {run_number}: Received text chunk: {chunk.text[:50]}..."
                )
                response_text += chunk.text

        api_duration = time.time() - api_call_start
        total_duration = time.time() - gen_start_time
        log_marker(
            f"Task {run_number}: Generation complete in {total_duration:.2f}s", "INFO"
        )
        print(
            f"[{datetime.datetime.now()}] Task {run_number}: API call duration: {api_duration:.2f}s"
        )
        print(
            f"[{datetime.datetime.now()}] Task {run_number}: Image path: {temp_output_path if temp_output_path else 'None'}"
        )
        print(
            f"[{datetime.datetime.now()}] Task {run_number}: Text response length: {len(response_text)} chars"
        )

        return response_text, temp_output_path, saved_output_path

    except Exception as e:
        log_marker(f"Task {run_number}: Error during generation: {str(e)}", "ERROR")
        import traceback

        print(
            f"[{datetime.datetime.now()}] Task {run_number}: ERROR traceback: {traceback.format_exc()}"
        )
        return f"Error: {str(e)}", None, None


def generate_wrapper(prompt, num_parallel_runs, *uploaded_images):
    log_marker(
        f"Generation requested with prompt: '{prompt}', num_parallel_runs: {num_parallel_runs}",
        "START",
    )
    print(
        f"[{datetime.datetime.now()}] Received {len(uploaded_images)} potential reference images"
    )

    # Run the async function in the event loop
    print(f"[{datetime.datetime.now()}] Creating new event loop")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Convert num_parallel_runs to int to ensure it's handled correctly
        num_runs = int(num_parallel_runs)
        print(f"[{datetime.datetime.now()}] Running {num_runs} parallel generations")

        start_time = time.time()
        texts, images, saved_paths = loop.run_until_complete(
            generate_image_async(prompt, uploaded_images, num_runs)
        )
        total_time = time.time() - start_time
        log_marker(f"All generations completed in {total_time:.2f} seconds", "END")
        print(
            f"[{datetime.datetime.now()}] Processing results: {len(texts)} texts, {len(images)} images"
        )

        return process_results(texts, images, saved_paths, num_runs)
    except Exception as e:
        log_marker(f"Error in generate_wrapper: {str(e)}", "ERROR")
        import traceback

        print(f"[{datetime.datetime.now()}] ERROR traceback: {traceback.format_exc()}")
        return (
            "Error in generation process. Check server logs.",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    finally:
        print(f"[{datetime.datetime.now()}] Closing event loop")
        loop.close()


def process_results(texts, images, saved_paths, num_runs):
    print(
        f"[{datetime.datetime.now()}] Processing results for display: {len(texts)} texts, {len(images)} images"
    )

    # Process the results list to fit our output format
    # If fewer results than expected, pad with None
    while len(texts) < 8:
        texts.append("")
    while len(images) < 8:
        images.append(None)
    while len(saved_paths) < 8:
        saved_paths.append(None)

    # Format the text responses
    formatted_text = ""
    for i, text in enumerate(texts):
        if (
            i < num_runs and text
        ):  # Only include results for the runs that were requested
            formatted_text += f"Result {i+1}: {text}\n\n"

    print(
        f"[{datetime.datetime.now()}] Formatted text length: {len(formatted_text)} chars"
    )
    print(
        f"[{datetime.datetime.now()}] Returning {min(len(images), 8)} images for display"
    )

    # Return all results for display (we don't return saved_paths to the UI)
    return (
        formatted_text,
        images[0],
        images[1],
        images[2],
        images[3],
        images[4],
        images[5],
        images[6],
        images[7],
    )


def ensure_output_directory():
    """Ensure that the outputs directory exists"""
    output_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(output_dir):
        log_marker(f"Creating outputs directory at: {output_dir}", "INFO")
        os.makedirs(output_dir)
        print(f"[{datetime.datetime.now()}] Created new outputs directory")
    else:
        print(
            f"[{datetime.datetime.now()}] Using existing outputs directory: {output_dir}"
        )

    # Log number of existing files in the directory
    try:
        existing_files = os.listdir(output_dir)
        print(
            f"[{datetime.datetime.now()}] Outputs directory contains {len(existing_files)} existing files"
        )
    except Exception as e:
        print(
            f"[{datetime.datetime.now()}] Error checking outputs directory contents: {str(e)}"
        )

    return output_dir


def generate_unique_filename(extension=".png"):
    """Generate a unique filename using timestamp and UUID"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
    return f"{timestamp}_{unique_id}{extension}"


# Call ensure_output_directory at startup
output_directory = ensure_output_directory()
log_marker(f"Outputs will be saved to: {output_directory}", "INFO")

# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Gemini Image Generation")
    gr.Markdown(
        "Enter a descriptive prompt and optionally upload reference images to guide the generation. All reference images will be considered together."
    )

    # Input section
    with gr.Column():
        prompt = gr.Textbox(
            label="Prompt (What do you want to generate?)",
            placeholder="Describe what image you want to generate, be specific about style, content, and mood...",
            lines=3,
        )

        # Add logging for when user enters a prompt
        def log_prompt_change(value):
            print(f"[{datetime.datetime.now()}] User entered prompt: '{value}'")
            return value

        prompt.change(fn=log_prompt_change, inputs=prompt, outputs=prompt)

        num_parallel_runs = gr.Slider(
            minimum=1,
            maximum=8,
            value=1,
            step=1,
            label="Number of parallel generations",
            info="How many times to generate images with the same prompt (1-8)",
        )

        # Add logging for when user adjusts the slider
        def log_slider_change(value):
            print(
                f"[{datetime.datetime.now()}] User set number of parallel runs to: {value}"
            )
            return value

        num_parallel_runs.change(
            fn=log_slider_change, inputs=num_parallel_runs, outputs=num_parallel_runs
        )

        # All images in one row
        with gr.Row():
            image1 = gr.Image(
                label="Reference Image 1", type="filepath", height=160, container=True
            )
            image2 = gr.Image(
                label="Reference Image 2", type="filepath", height=160, container=True
            )
            image3 = gr.Image(
                label="Reference Image 3", type="filepath", height=160, container=True
            )
            image4 = gr.Image(
                label="Reference Image 4", type="filepath", height=160, container=True
            )

        # Add logging for when images are uploaded
        def log_image_upload(image, index):
            if image is not None:
                print(
                    f"[{datetime.datetime.now()}] User uploaded image {index}: {image}"
                )
            else:
                print(f"[{datetime.datetime.now()}] User removed image {index}")
            return image

        image1.change(
            fn=lambda x: log_image_upload(x, 1), inputs=image1, outputs=image1
        )
        image2.change(
            fn=lambda x: log_image_upload(x, 2), inputs=image2, outputs=image2
        )
        image3.change(
            fn=lambda x: log_image_upload(x, 3), inputs=image3, outputs=image3
        )
        image4.change(
            fn=lambda x: log_image_upload(x, 4), inputs=image4, outputs=image4
        )

        example_prompts = [
            ["A surreal landscape with floating islands and waterfalls"],
            ["A cyberpunk city at night with neon lights and flying cars"],
            ["A portrait of a fantasy character with glowing eyes and ornate armor"],
        ]

        gr.Examples(examples=example_prompts, inputs=[prompt])

        # Add logging for examples
        def log_example_selected(example_index, example_data):
            print(
                f"[{datetime.datetime.now()}] User selected example {example_index+1}: '{example_data}'"
            )

        # Note: We can't easily attach event handlers to Examples component in Gradio

        generate_btn = gr.Button("Generate Images", variant="primary")

        # Add logging for when the button is clicked (this is in addition to the existing logs in generate_wrapper)
        def log_generate_click():
            print(f"[{datetime.datetime.now()}] User clicked 'Generate Images' button")

        generate_btn.click(
            fn=log_generate_click, inputs=None, outputs=None, queue=False
        )

    # Output section
    with gr.Column():
        with gr.Accordion("Generation Details", open=True):
            output_text = gr.Textbox(label="Response Text")

        # First row of output images
        with gr.Row():
            output_image1 = gr.Image(label="Result 1")
            output_image2 = gr.Image(label="Result 2")
            output_image3 = gr.Image(label="Result 3")
            output_image4 = gr.Image(label="Result 4")

        # Second row of output images
        with gr.Row():
            output_image5 = gr.Image(label="Result 5")
            output_image6 = gr.Image(label="Result 6")
            output_image7 = gr.Image(label="Result 7")
            output_image8 = gr.Image(label="Result 8")

    with gr.Accordion("About", open=False):
        gr.Markdown(
            f"""
        This app uses Google's Gemini 2.0 Flash experimental image generation model to create images based on text prompts and reference images.
        
        - The reference images are optional but can help guide the style and content of the generated image
        - Generation typically takes 10-30 seconds depending on complexity
        - Running multiple generations in parallel will produce variations of the same prompt
        - The model may sometimes return text responses along with or instead of images
        - All generated images are automatically saved to the `{os.path.relpath(output_directory)}` directory with timestamped filenames
        """
        )

    generate_btn.click(
        generate_wrapper,
        inputs=[prompt, num_parallel_runs, image1, image2, image3, image4],
        outputs=[
            output_text,
            output_image1,
            output_image2,
            output_image3,
            output_image4,
            output_image5,
            output_image6,
            output_image7,
            output_image8,
        ],
        show_progress=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Image Generation Gradio App")
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of reference image inputs (default: 4)",
    )
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    args = parser.parse_args()

    log_marker(
        f"Command-line arguments: num_images={args.num_images}, share={args.share}",
        "INFO",
    )

    # If you want more or fewer than 4 reference images, run with: python app.py --num-images N
    if args.num_images != 4:
        log_marker(
            f"Creating custom app with {args.num_images} reference image inputs", "INFO"
        )
        # Create a custom app with the specified number of image inputs
        with gr.Blocks(theme=gr.themes.Soft()) as custom_app:
            gr.Markdown("# Gemini Image Generation")
            gr.Markdown(
                "Enter a descriptive prompt and optionally upload reference images to guide the generation. All reference images will be considered together."
            )

            # Input section
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt (What do you want to generate?)",
                    placeholder="Describe what image you want to generate, be specific about style, content, and mood...",
                    lines=3,
                )

                # Add logging for when user enters a prompt
                def log_prompt_change(value):
                    print(f"[{datetime.datetime.now()}] User entered prompt: '{value}'")
                    return value

                prompt.change(fn=log_prompt_change, inputs=prompt, outputs=prompt)

                num_parallel_runs = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    label="Number of parallel generations",
                    info="How many times to generate images with the same prompt (1-8)",
                )

                # Add logging for when user adjusts the slider
                def log_slider_change(value):
                    print(
                        f"[{datetime.datetime.now()}] User set number of parallel runs to: {value}"
                    )
                    return value

                num_parallel_runs.change(
                    fn=log_slider_change,
                    inputs=num_parallel_runs,
                    outputs=num_parallel_runs,
                )

                # Create the specified number of image inputs
                image_inputs = []

                # Add all images in one row
                with gr.Row():
                    for i in range(args.num_images):
                        img = gr.Image(
                            label=f"Reference Image {i+1}",
                            type="filepath",
                            height=160,
                            container=True,
                        )
                        image_inputs.append(img)

                        # Add logging for when images are uploaded (we need to define this inside the loop)
                        img_index = (
                            i + 1
                        )  # Create a closure to capture the current index

                        def log_image_upload_custom(image, idx=img_index):
                            if image is not None:
                                print(
                                    f"[{datetime.datetime.now()}] User uploaded custom image {idx}: {image}"
                                )
                            else:
                                print(
                                    f"[{datetime.datetime.now()}] User removed custom image {idx}"
                                )
                            return image

                        img.change(fn=log_image_upload_custom, inputs=img, outputs=img)

                example_prompts = [
                    ["A surreal landscape with floating islands and waterfalls"],
                    ["A cyberpunk city at night with neon lights and flying cars"],
                    [
                        "A portrait of a fantasy character with glowing eyes and ornate armor"
                    ],
                ]

                gr.Examples(examples=example_prompts, inputs=[prompt])

                # Add logging for examples (note: can't easily attach events to Examples)
                def log_example_selected(example_index, example_data):
                    print(
                        f"[{datetime.datetime.now()}] User selected example {example_index+1}: '{example_data}'"
                    )

                # Note: We can't easily attach event handlers to Examples component in Gradio

                generate_btn = gr.Button("Generate Images", variant="primary")

                # Add logging for when the button is clicked
                def log_generate_click():
                    print(
                        f"[{datetime.datetime.now()}] User clicked 'Generate Images' button (custom app)"
                    )

                generate_btn.click(
                    fn=log_generate_click, inputs=None, outputs=None, queue=False
                )

            # Output section
            with gr.Column():
                with gr.Accordion("Generation Details", open=True):
                    output_text = gr.Textbox(label="Response Text")

                # First row of output images
                with gr.Row():
                    output_image1 = gr.Image(label="Result 1")
                    output_image2 = gr.Image(label="Result 2")
                    output_image3 = gr.Image(label="Result 3")
                    output_image4 = gr.Image(label="Result 4")

                # Second row of output images
                with gr.Row():
                    output_image5 = gr.Image(label="Result 5")
                    output_image6 = gr.Image(label="Result 6")
                    output_image7 = gr.Image(label="Result 7")
                    output_image8 = gr.Image(label="Result 8")

            with gr.Accordion("About", open=False):
                gr.Markdown(
                    f"""
                This app uses Google's Gemini 2.0 Flash experimental image generation model to create images based on text prompts and reference images.
                
                - The reference images are optional but can help guide the style and content of the generated image
                - Generation typically takes 10-30 seconds depending on complexity
                - Running multiple generations in parallel will produce variations of the same prompt
                - The model may sometimes return text responses along with or instead of images
                - All generated images are automatically saved to the `{os.path.relpath(output_directory)}` directory with timestamped filenames
                """
                )

            generate_btn.click(
                generate_wrapper,
                inputs=[prompt, num_parallel_runs] + image_inputs,
                outputs=[
                    output_text,
                    output_image1,
                    output_image2,
                    output_image3,
                    output_image4,
                    output_image5,
                    output_image6,
                    output_image7,
                    output_image8,
                ],
                show_progress=True,
            )

        log_marker(f"Launching custom app with share={args.share}", "START")
        custom_app.launch(share=args.share)
    else:
        # Use the default app with 4 image inputs
        log_marker(
            f"Launching default app with 4 reference images and share={args.share}",
            "START",
        )
        app.launch(share=args.share)
