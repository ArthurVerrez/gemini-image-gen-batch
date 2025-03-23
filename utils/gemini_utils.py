import os
import time
import traceback
import mimetypes
from google import genai
from google.genai import types
from utils.logging_utils import log_marker, log_timestamp
from utils.file_utils import (
    save_binary_file,
    create_temp_file,
    save_to_output_directory,
)


def initialize_gemini_client():
    """
    Initialize and return a Gemini API client.

    Returns:
        genai.Client: Initialized Gemini client
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log_marker("GEMINI_API_KEY environment variable is not set", "ERROR")
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. Please set it or create a .env file."
        )

    client = genai.Client(api_key=api_key)
    return client


def upload_reference_images(client, uploaded_images, run_number=1):
    """
    Upload reference images to the Gemini API.

    Args:
        client: Gemini API client
        uploaded_images: List of image file paths
        run_number: Task run number for logging

    Returns:
        list: List of uploaded file objects
    """
    # Filter out None values from uploaded_images
    uploaded_images = [img for img in uploaded_images if img is not None]
    log_timestamp(f"Task {run_number}: Using {len(uploaded_images)} reference images")

    files = []
    for i, img in enumerate(uploaded_images):
        log_timestamp(
            f"Task {run_number}: Uploading reference image {i+1}/{len(uploaded_images)}"
        )
        try:
            file = client.files.upload(file=img)
            files.append(file)
            log_timestamp(
                f"Task {run_number}: Successfully uploaded image {i+1}, URI: {file.uri}"
            )
        except Exception as e:
            log_marker(
                f"Task {run_number}: Error uploading image {i+1}: {str(e)}",
                "WARNING",
            )
            # Continue with other images

    return files


def build_gemini_content(files, prompt):
    """
    Build the content parts for a Gemini API request.

    Args:
        files: List of uploaded file objects
        prompt: Text prompt for image generation

    Returns:
        list: List of content objects for the API request
    """
    parts = []
    for i, file in enumerate(files):
        log_timestamp(f"Adding image part {i+1}/{len(files)}")
        parts.append(
            types.Part.from_uri(
                file_uri=file.uri,
                mime_type=file.mime_type,
            )
        )
    log_timestamp(f"Adding text prompt part")
    parts.append(types.Part.from_text(text=prompt))

    contents = [
        types.Content(
            role="user",
            parts=parts,
        )
    ]

    return contents


def get_generation_config():
    """
    Get the configuration for Gemini image generation.

    Returns:
        types.GenerateContentConfig: Configuration object
    """
    return types.GenerateContentConfig(
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


def generate_single_image(prompt, uploaded_images, output_directory, run_number=1):
    """
    Generate a single image using the Gemini API.

    Args:
        prompt: Text prompt for image generation
        uploaded_images: List of image file paths
        output_directory: Directory to save the output image
        run_number: Task run number for logging

    Returns:
        tuple: (response_text, temp_output_path, saved_output_path)
    """
    gen_start_time = time.time()
    log_marker(f"Task {run_number}: Starting generation", "INFO")
    log_timestamp(f"Task {run_number}: Prompt: '{prompt}'")

    # Filter out None values from uploaded_images
    uploaded_images = [img for img in uploaded_images if img is not None]
    log_timestamp(f"Task {run_number}: Using {len(uploaded_images)} reference images")

    if not prompt:
        log_marker(f"Task {run_number}: No prompt provided", "ERROR")
        return "Please provide a prompt text", None, None

    try:
        log_timestamp(f"Task {run_number}: Creating Gemini client")
        client = initialize_gemini_client()

        # Upload the input images if provided
        files = upload_reference_images(client, uploaded_images, run_number)

        model = "gemini-2.0-flash-exp-image-generation"
        log_timestamp(f"Task {run_number}: Using model: {model}")

        # Build the content parts
        contents = build_gemini_content(files, prompt)

        log_timestamp(f"Task {run_number}: Configuring generation parameters")
        generate_content_config = get_generation_config()

        # Create a temporary file to store the output
        temp_output_path = None
        saved_output_path = None
        response_text = ""

        log_timestamp(f"Task {run_number}: Starting Gemini content generation stream")
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
                log_timestamp(f"Task {run_number}: Received empty chunk, skipping")
                continue

            if chunk.candidates[0].content.parts[0].inline_data:
                # Image response
                log_timestamp(f"Task {run_number}: Received image response")
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                log_timestamp(
                    f"Task {run_number}: Image mime type: {inline_data.mime_type}, extension: {file_extension}"
                )

                # Create a temporary file
                temp_output_path = create_temp_file(inline_data.mime_type)
                log_timestamp(
                    f"Task {run_number}: Created temporary file: {temp_output_path}"
                )

                # Save the binary data
                save_binary_file(temp_output_path, inline_data.data)
                log_timestamp(f"Task {run_number}: Saved image data to temporary file")

                # Also save a copy to the outputs directory with a unique filename
                saved_output_path = save_to_output_directory(
                    temp_output_path, output_directory, file_extension
                )
            else:
                # Text response
                log_timestamp(
                    f"Task {run_number}: Received text chunk: {chunk.text[:50]}..."
                )
                response_text += chunk.text

        api_duration = time.time() - api_call_start
        total_duration = time.time() - gen_start_time
        log_marker(
            f"Task {run_number}: Generation complete in {total_duration:.2f}s", "INFO"
        )
        log_timestamp(f"Task {run_number}: API call duration: {api_duration:.2f}s")
        log_timestamp(
            f"Task {run_number}: Image path: {temp_output_path if temp_output_path else 'None'}"
        )
        log_timestamp(
            f"Task {run_number}: Text response length: {len(response_text)} chars"
        )

        return response_text, temp_output_path, saved_output_path

    except Exception as e:
        log_marker(f"Task {run_number}: Error during generation: {str(e)}", "ERROR")
        log_timestamp(f"Task {run_number}: ERROR traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}", None, None
