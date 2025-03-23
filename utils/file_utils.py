import os
import uuid
import datetime
import shutil
import tempfile
from utils.logging_utils import log_marker, log_timestamp


def ensure_output_directory(directory_name="outputs"):
    """
    Ensure that the output directory exists.

    Args:
        directory_name: Name of the output directory

    Returns:
        str: Path to the output directory
    """
    output_dir = os.path.join(os.getcwd(), directory_name)
    if not os.path.exists(output_dir):
        log_marker(f"Creating outputs directory at: {output_dir}", "INFO")
        os.makedirs(output_dir)
        log_timestamp(f"Created new outputs directory")
    else:
        log_timestamp(f"Using existing outputs directory: {output_dir}")

    # Log number of existing files in the directory
    try:
        existing_files = os.listdir(output_dir)
        log_timestamp(
            f"Outputs directory contains {len(existing_files)} existing files"
        )
    except Exception as e:
        log_timestamp(f"Error checking outputs directory contents: {str(e)}")

    return output_dir


def generate_unique_filename(extension=".png"):
    """
    Generate a unique filename using timestamp and UUID.

    Args:
        extension: File extension with dot (e.g. '.png')

    Returns:
        str: A unique filename
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
    return f"{timestamp}_{unique_id}{extension}"


def save_binary_file(file_name, data):
    """
    Save binary data to a file.

    Args:
        file_name: Path to save the file
        data: Binary data to save
    """
    log_timestamp(f"Saving binary file to: {file_name}")
    with open(file_name, "wb") as f:
        f.write(data)
    log_timestamp(f"File saved successfully: {file_name}")


def create_temp_file(mime_type):
    """
    Create a temporary file with the appropriate extension based on MIME type.

    Args:
        mime_type: MIME type of the file

    Returns:
        str: Path to the temporary file
    """
    import mimetypes

    file_extension = mimetypes.guess_extension(mime_type)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    temp_path = temp_file.name
    temp_file.close()
    return temp_path


def save_to_output_directory(temp_path, output_directory, extension=None):
    """
    Save a file from a temporary location to the output directory with a unique name.

    Args:
        temp_path: Path to the temporary file
        output_directory: Directory to save the file to
        extension: Optional file extension override

    Returns:
        str: Path to the saved file
    """
    if extension is None:
        extension = os.path.splitext(temp_path)[1]

    output_path = os.path.join(output_directory, generate_unique_filename(extension))
    shutil.copy2(temp_path, output_path)
    log_timestamp(f"Saved permanent copy to: {output_path}")

    return output_path
