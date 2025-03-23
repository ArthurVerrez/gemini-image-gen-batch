import asyncio
import concurrent.futures
import time
from utils.logging_utils import log_timestamp
from utils.gemini_utils import generate_single_image


async def generate_image_async(prompt, uploaded_images, output_directory, num_runs=1):
    """
    Generate multiple images asynchronously.

    Args:
        prompt: Text prompt for image generation
        uploaded_images: List of image file paths
        output_directory: Directory to save the output images
        num_runs: Number of images to generate in parallel

    Returns:
        tuple: (response_texts, output_images, saved_images)
    """
    log_timestamp(
        f"Starting async generation with prompt: '{prompt}', num_runs: {num_runs}"
    )
    log_timestamp(
        f"Number of provided reference images: {len([img for img in uploaded_images if img is not None])}"
    )

    start_time = time.time()
    # Create a thread pool executor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        log_timestamp(f"Created thread pool executor with max_workers=8")
        # Schedule the generation tasks
        tasks = []
        for i in range(num_runs):
            log_timestamp(f"Scheduling generation task {i+1}/{num_runs}")
            tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    executor,
                    generate_single_image,
                    prompt,
                    uploaded_images,
                    output_directory,
                    i + 1,
                )
            )

        # Wait for all tasks to complete
        log_timestamp(f"Waiting for all {len(tasks)} tasks to complete...")
        results = await asyncio.gather(*tasks)
        log_timestamp(f"All tasks completed in {time.time() - start_time:.2f} seconds")

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
        from utils.logging_utils import log_saved_images_summary

        log_saved_images_summary(valid_saved_images)

        log_timestamp(
            f"Generated {len(output_images)} images and {len(response_texts)} text responses"
        )
        return response_texts, output_images, saved_images


def process_results(texts, images, saved_paths, num_runs):
    """
    Process generation results for display in the UI.

    Args:
        texts: List of text responses
        images: List of image file paths
        saved_paths: List of saved image paths
        num_runs: Number of images that were generated

    Returns:
        tuple: Formatted results for Gradio UI
    """
    log_timestamp(
        f"Processing results for display: {len(texts)} texts, {len(images)} images"
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

    log_timestamp(f"Formatted text length: {len(formatted_text)} chars")
    log_timestamp(f"Returning {min(len(images), 8)} images for display")

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
