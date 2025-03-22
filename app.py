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

# Load environment variables from .env file
load_dotenv()

# Ensure API key is available
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. Please set it or create a .env file."
    )


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)


async def generate_image_async(prompt, uploaded_images, num_runs=1):
    # Create a thread pool executor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Schedule the generation tasks
        tasks = []
        for i in range(num_runs):
            tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    executor, generate_single_image, prompt, uploaded_images
                )
            )

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Separate text and image results
        response_texts = []
        output_images = []

        for text, image in results:
            response_texts.append(text)
            output_images.append(image)

        return response_texts, output_images


def generate_single_image(prompt, uploaded_images):
    # Filter out None values from uploaded_images
    uploaded_images = [img for img in uploaded_images if img is not None]

    if not prompt:
        return "Please provide a prompt text", None

    try:
        client = genai.Client(
            api_key=api_key,
        )

        # Upload the input images if provided
        files = []
        for img in uploaded_images:
            files.append(client.files.upload(file=img))

        model = "gemini-2.0-flash-exp-image-generation"

        # Build the content parts
        parts = []
        for i, file in enumerate(files):
            parts.append(
                types.Part.from_uri(
                    file_uri=file.uri,
                    mime_type=file.mime_type,
                )
            )
        parts.append(types.Part.from_text(text=prompt))

        contents = [
            types.Content(
                role="user",
                parts=parts,
            )
        ]

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
        response_text = ""

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
                continue

            if chunk.candidates[0].content.parts[0].inline_data:
                # Image response
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)

                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension
                )
                temp_output_path = temp_file.name
                temp_file.close()

                # Save the binary data
                save_binary_file(temp_output_path, inline_data.data)
            else:
                # Text response
                response_text += chunk.text

        return response_text, temp_output_path

    except Exception as e:
        return f"Error: {str(e)}", None


def generate_wrapper(prompt, num_parallel_runs, *uploaded_images):
    # Run the async function in the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            generate_image_async(prompt, uploaded_images, num_parallel_runs)
        )
    finally:
        loop.close()


# Create the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Gemini Image Generation")
    gr.Markdown(
        "Enter a descriptive prompt and optionally upload reference images to guide the generation. All reference images will be considered together."
    )

    # Input section
    prompt = gr.Textbox(
        label="Prompt (What do you want to generate?)",
        placeholder="Describe what image you want to generate, be specific about style, content, and mood...",
        lines=3,
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

    example_prompts = [
        ["A surreal landscape with floating islands and waterfalls"],
        ["A cyberpunk city at night with neon lights and flying cars"],
        ["A portrait of a fantasy character with glowing eyes and ornate armor"],
    ]

    gr.Examples(examples=example_prompts, inputs=[prompt])

    generate_btn = gr.Button("Generate Image", variant="primary")

    # Output section
    output_image = gr.Image(label="Generated Image")
    output_text = gr.Textbox(label="Response Text")

    with gr.Accordion("About", open=False):
        gr.Markdown(
            """
        This app uses Google's Gemini 2.0 Flash experimental image generation model to create images based on text prompts and reference images.
        
        - The reference images are optional but can help guide the style and content of the generated image
        - Generation typically takes 10-30 seconds depending on complexity
        - The model may sometimes return text responses along with or instead of images
        """
        )

    generate_btn.click(
        generate_wrapper,
        inputs=[prompt, image1, image2, image3, image4],
        outputs=[output_text, output_image],
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

    # If you want more or fewer than 4 reference images, run with: python app.py --num-images N
    if args.num_images != 4:
        # Create a custom app with the specified number of image inputs
        with gr.Blocks(theme=gr.themes.Soft()) as custom_app:
            gr.Markdown("# Gemini Image Generation")
            gr.Markdown(
                "Enter a descriptive prompt and optionally upload reference images to guide the generation. All reference images will be considered together."
            )

            # Input section
            prompt = gr.Textbox(
                label="Prompt (What do you want to generate?)",
                placeholder="Describe what image you want to generate, be specific about style, content, and mood...",
                lines=3,
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

            example_prompts = [
                ["A surreal landscape with floating islands and waterfalls"],
                ["A cyberpunk city at night with neon lights and flying cars"],
                [
                    "A portrait of a fantasy character with glowing eyes and ornate armor"
                ],
            ]

            gr.Examples(examples=example_prompts, inputs=[prompt])

            generate_btn = gr.Button("Generate Image", variant="primary")

            # Output section
            output_image = gr.Image(label="Generated Image")
            output_text = gr.Textbox(label="Response Text")

            with gr.Accordion("About", open=False):
                gr.Markdown(
                    """
                This app uses Google's Gemini 2.0 Flash experimental image generation model to create images based on text prompts and reference images.
                
                - The reference images are optional but can help guide the style and content of the generated image
                - Generation typically takes 10-30 seconds depending on complexity
                - The model may sometimes return text responses along with or instead of images
                """
                )

            generate_btn.click(
                generate_wrapper,
                inputs=[prompt] + image_inputs,
                outputs=[output_text, output_image],
                show_progress=True,
            )

        custom_app.launch(share=args.share)
    else:
        # Use the default app with 4 image inputs
        app.launch(share=args.share)
