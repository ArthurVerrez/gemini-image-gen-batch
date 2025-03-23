import os
from dotenv import load_dotenv
import argparse
import asyncio
import time
import datetime

# Import utility modules
from utils.logging_utils import log_marker, log_system_info
from utils.file_utils import ensure_output_directory
from utils.gemini_utils import generate_single_image
from utils.async_utils import generate_image_async, process_results
from utils.ui_utils import (
    create_default_ui,
    create_custom_ui,
)

# Load environment variables from .env file
load_dotenv()

# Log startup info
log_marker("Starting Gemini Image Generation App", "START")
log_system_info()

# Call ensure_output_directory at startup
output_directory = ensure_output_directory()
log_marker(f"Outputs will be saved to: {output_directory}", "INFO")

# Check for API key - log warning but don't stop application
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    log_marker("GEMINI_API_KEY environment variable is not set", "WARNING")
    log_marker(
        "You will need to set the API key using the UI before generating images", "INFO"
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


def generate_wrapper(prompt, num_parallel_runs, *uploaded_images):
    """
    Wrapper function to handle image generation with progress tracking and UI updates.

    Args:
        prompt: Text prompt for image generation
        num_parallel_runs: Number of images to generate in parallel
        uploaded_images: Variable number of reference images

    Returns:
        tuple: Formatted results for the Gradio UI
    """
    log_marker(
        f"Generation requested with prompt: '{prompt}', num_parallel_runs: {num_parallel_runs}",
        "START",
    )
    print(f"Received {len(uploaded_images)} potential reference images")

    # Run the async function in the event loop
    print(f"Creating new event loop")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Convert num_parallel_runs to int to ensure it's handled correctly
        num_runs = int(num_parallel_runs)
        print(f"Running {num_runs} parallel generations")

        start_time = time.time()
        texts, images, saved_paths = loop.run_until_complete(
            generate_image_async(prompt, uploaded_images, output_directory, num_runs)
        )
        total_time = time.time() - start_time
        log_marker(f"All generations completed in {total_time:.2f} seconds", "END")
        print(f"Processing results: {len(texts)} texts, {len(images)} images")

        return process_results(texts, images, saved_paths, num_runs)
    except Exception as e:
        log_marker(f"Error in generate_wrapper: {str(e)}", "ERROR")
        import traceback

        print(f"ERROR traceback: {traceback.format_exc()}")
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
        print(f"Closing event loop")
        loop.close()


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
        custom_app = create_custom_ui(
            output_directory, args.num_images, generate_wrapper
        )

        log_marker(f"Launching custom app with share={args.share}", "START")
        custom_app.launch(share=args.share)
    else:
        # Use the default app with 4 image inputs
        log_marker(
            f"Launching default app with 4 reference images and share={args.share}",
            "START",
        )
        app = create_default_ui(output_directory, generate_wrapper)

        app.launch(share=args.share)
