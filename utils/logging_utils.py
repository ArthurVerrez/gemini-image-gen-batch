import os
import sys
import platform
import datetime
import pkg_resources


def log_marker(message, marker_type="INFO"):
    """
    Log a message with a distinct marker for better visibility.

    Args:
        message: The message to log
        marker_type: Type of marker (START, END, ERROR, WARNING, INFO)
    """
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


def log_timestamp(message):
    """
    Log a message with a timestamp.

    Args:
        message: The message to log
    """
    print(f"[{datetime.datetime.now()}] {message}")


def log_system_info():
    """
    Log system information to help with debugging.
    """
    log_marker("System Information", "INFO")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")

    # Log relevant package versions
    try:
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


def log_saved_images_summary(saved_images):
    """
    Log a summary of the saved images.

    Args:
        saved_images: List of paths to saved images
    """
    if not saved_images:
        log_marker("No images were saved in this generation run", "WARNING")
        return

    log_marker(f"Saved {len(saved_images)} images to outputs directory", "INFO")
    for i, img_path in enumerate(saved_images):
        log_timestamp(f"Image {i+1}: {os.path.basename(img_path)}")

    # Get total size of saved images
    total_size = sum(os.path.getsize(img) for img in saved_images)
    log_timestamp(f"Total size of saved images: {total_size / (1024*1024):.2f} MB")
