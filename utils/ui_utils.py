import os
import gradio as gr
from utils.logging_utils import log_timestamp


def log_slider_change(value):
    """
    Log when a user changes the number of parallel runs slider.

    Args:
        value: The new slider value

    Returns:
        int: The slider value
    """
    log_timestamp(f"User set number of parallel runs to: {value}")
    return value


def log_image_upload(image, index):
    """
    Log when a user uploads or removes an image.

    Args:
        image: The uploaded image path or None if removed
        index: The index of the image

    Returns:
        The image
    """
    if image is not None:
        log_timestamp(f"User uploaded image {index}: {image}")
    else:
        log_timestamp(f"User removed image {index}")
    return image


def log_generate_click():
    """
    Log when a user clicks the generate button.
    """
    log_timestamp("User clicked 'Generate Images' button")


def disable_inputs():
    """
    Disable all input components during generation.

    Returns:
        list: List of Gradio component updates
    """
    return [
        gr.update(interactive=False) for _ in range(6)
    ]  # For all 6 input components


def enable_inputs():
    """
    Enable all input components after generation.

    Returns:
        list: List of Gradio component updates
    """
    return [gr.update(interactive=True) for _ in range(6)]  # For all 6 input components


def update_api_key(api_key):
    """
    Update the GEMINI_API_KEY environment variable with the provided value.

    Args:
        api_key: The API key to set

    Returns:
        tuple: (status_message, is_valid_key)
    """
    if api_key and api_key.strip():
        os.environ["GEMINI_API_KEY"] = api_key.strip()
        log_timestamp(f"GEMINI_API_KEY environment variable updated")
        return "API Key updated successfully", True
    return "API Key cannot be empty", False


def create_default_ui(output_directory, generate_wrapper_fn):
    """
    Create the default Gradio UI with 4 reference image inputs.

    Args:
        output_directory: Directory where output images are saved
        generate_wrapper_fn: Function to call for image generation

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# Gemini AI Image Generator")

        # Check if API key is set
        initial_api_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_is_set = bool(initial_api_key)

        # API Key input at the top of the app
        with gr.Accordion("API Settings", open=not api_key_is_set) as api_settings:
            api_key = gr.Textbox(
                label="Gemini API Key",
                value=initial_api_key,
                type="password",
                placeholder="Enter your Gemini API Key here",
            )
            api_key_update_btn = gr.Button("Update API Key", size="sm")
            api_key_status = gr.Textbox(label="Status", interactive=False)

        # Create main UI container that will be conditionally visible
        with gr.Column(visible=api_key_is_set) as main_ui:
            gr.Markdown(
                "Generate high-quality AI images by entering descriptive prompts. Optionally upload reference images to guide the generation style and content."
            )

            # Input section
            with gr.Column() as input_column:
                prompt = gr.Textbox(
                    label="Prompt (What do you want to generate?)",
                    placeholder="Describe what image you want to generate, be specific about style, content, and mood...",
                    lines=3,
                )

                num_parallel_runs = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    label="Number of parallel generations",
                    info="How many times to generate images with the same prompt (1-8)",
                )

                # Add logging for when user adjusts the slider
                num_parallel_runs.change(
                    fn=log_slider_change,
                    inputs=num_parallel_runs,
                    outputs=num_parallel_runs,
                )

                # All images in one row
                with gr.Row():
                    image1 = gr.Image(
                        label="Reference Image 1",
                        type="filepath",
                        height=160,
                        container=True,
                    )
                    image2 = gr.Image(
                        label="Reference Image 2",
                        type="filepath",
                        height=160,
                        container=True,
                    )
                    image3 = gr.Image(
                        label="Reference Image 3",
                        type="filepath",
                        height=160,
                        container=True,
                    )
                    image4 = gr.Image(
                        label="Reference Image 4",
                        type="filepath",
                        height=160,
                        container=True,
                    )

                # Add logging for when images are uploaded
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
                    [
                        "A portrait of a fantasy character with glowing eyes and ornate armor"
                    ],
                ]

                gr.Examples(examples=example_prompts, inputs=[prompt])

                generate_btn = gr.Button("Generate Images", variant="primary")

                # Add logging for when the button is clicked
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

        # Show a message if no API key is set
        with gr.Column(visible=not api_key_is_set) as api_key_prompt:
            gr.Markdown(
                "### Please set your Gemini API Key in the settings above to use the app"
            )

        # Set up API key update functionality with visibility toggle
        def handle_api_key_update(key):
            message, is_valid = update_api_key(key)
            return (
                message,
                gr.update(visible=is_valid),
                gr.update(visible=not is_valid),
                gr.update(open=not is_valid),
            )

        api_key_update_btn.click(
            fn=handle_api_key_update,
            inputs=api_key,
            outputs=[api_key_status, main_ui, api_key_prompt, api_settings],
        )

        # Set up UI interactivity
        output_images = [
            output_image1,
            output_image2,
            output_image3,
            output_image4,
            output_image5,
            output_image6,
            output_image7,
            output_image8,
        ]

        # First disable inputs, then run the generation, then re-enable inputs
        generate_btn.click(
            disable_inputs,
            None,
            [prompt, num_parallel_runs, image1, image2, image3, image4],
            queue=False,
        ).then(
            generate_wrapper_fn,
            inputs=[prompt, num_parallel_runs, image1, image2, image3, image4],
            outputs=[output_text] + output_images,
            show_progress=True,
        ).then(
            enable_inputs,
            None,
            [prompt, num_parallel_runs, image1, image2, image3, image4],
        )

        return app


def create_custom_ui(output_directory, num_images, generate_wrapper_fn):
    """
    Create a custom Gradio UI with a specified number of reference image inputs.

    Args:
        output_directory: Directory where output images are saved
        num_images: Number of reference image inputs
        generate_wrapper_fn: Function to call for image generation

    Returns:
        gr.Blocks: Gradio Blocks interface
    """
    with gr.Blocks(theme=gr.themes.Soft()) as custom_app:
        gr.Markdown("# Gemini AI Image Generator")

        # Check if API key is set
        initial_api_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_is_set = bool(initial_api_key)

        # API Key input at the top of the app
        with gr.Accordion("API Settings", open=not api_key_is_set) as api_settings:
            api_key = gr.Textbox(
                label="Gemini API Key",
                value=initial_api_key,
                type="password",
                placeholder="Enter your Gemini API Key here",
            )
            api_key_update_btn = gr.Button("Update API Key", size="sm")
            api_key_status = gr.Textbox(label="Status", interactive=False)

        # Create main UI container that will be conditionally visible
        with gr.Column(visible=api_key_is_set) as main_ui:
            gr.Markdown(
                "Generate high-quality AI images by entering descriptive prompts. Optionally upload reference images to guide the generation style and content."
            )

            # Input section
            with gr.Column() as custom_input_column:
                prompt = gr.Textbox(
                    label="Prompt (What do you want to generate?)",
                    placeholder="Describe what image you want to generate, be specific about style, content, and mood...",
                    lines=3,
                )

                num_parallel_runs = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    label="Number of parallel generations",
                    info="How many times to generate images with the same prompt (1-8)",
                )

                # Add logging for when user adjusts the slider
                num_parallel_runs.change(
                    fn=log_slider_change,
                    inputs=num_parallel_runs,
                    outputs=num_parallel_runs,
                )

                # Create the specified number of image inputs
                image_inputs = []

                # Add all images in one row
                with gr.Row():
                    for i in range(num_images):
                        img = gr.Image(
                            label=f"Reference Image {i+1}",
                            type="filepath",
                            height=160,
                            container=True,
                        )
                        image_inputs.append(img)

                        # Add logging for when images are uploaded
                        img_index = (
                            i + 1
                        )  # Create a closure to capture the current index
                        img.change(
                            fn=lambda x, idx=img_index: log_image_upload(x, idx),
                            inputs=img,
                            outputs=img,
                        )

                example_prompts = [
                    ["A surreal landscape with floating islands and waterfalls"],
                    ["A cyberpunk city at night with neon lights and flying cars"],
                    [
                        "A portrait of a fantasy character with glowing eyes and ornate armor"
                    ],
                ]

                gr.Examples(examples=example_prompts, inputs=[prompt])

                generate_btn = gr.Button("Generate Images", variant="primary")

                # Add logging for when the button is clicked
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

        # Show a message if no API key is set
        with gr.Column(visible=not api_key_is_set) as api_key_prompt:
            gr.Markdown(
                "### Please set your Gemini API Key in the settings above to use the app"
            )

        # Set up API key update functionality with visibility toggle
        def handle_api_key_update(key):
            message, is_valid = update_api_key(key)
            return (
                message,
                gr.update(visible=is_valid),
                gr.update(visible=not is_valid),
                gr.update(open=not is_valid),
            )

        api_key_update_btn.click(
            fn=handle_api_key_update,
            inputs=api_key,
            outputs=[api_key_status, main_ui, api_key_prompt, api_settings],
        )

        output_images = [
            output_image1,
            output_image2,
            output_image3,
            output_image4,
            output_image5,
            output_image6,
            output_image7,
            output_image8,
        ]

        # Set up UI interactivity - for custom UI with variable number of inputs
        input_components = [prompt, num_parallel_runs] + image_inputs

        # First disable inputs, then run the generation, then re-enable inputs
        generate_btn.click(
            lambda: [disable_inputs()[0] for _ in range(len(input_components))],
            None,
            input_components,
            queue=False,
        ).then(
            generate_wrapper_fn,
            inputs=input_components,
            outputs=[output_text] + output_images,
            show_progress=True,
        ).then(
            lambda: [enable_inputs()[0] for _ in range(len(input_components))],
            None,
            input_components,
        )

        return custom_app
