# Gemini Image Generation Batch

A web application that uses Google's Gemini 2.0 Flash experimental image generation model to create images based on text prompts and reference images.

## Features

- Generate multiple images with a single text prompt
- Optionally include reference images to guide the style and content
- Run multiple generations in parallel for variations
- All generated images are automatically saved with timestamped filenames

## Project Structure

The project is organized as follows:

```
├── app.py                 # Main application entry point
├── utils/                 # Utility modules
│   ├── __init__.py        # Package initialization
│   ├── async_utils.py     # Async operations utilities
│   ├── file_utils.py      # File and directory utilities
│   ├── gemini_utils.py    # Gemini API interaction utilities
│   ├── logging_utils.py   # Logging utilities
│   └── ui_utils.py        # Gradio UI utilities
├── outputs/               # Generated images output directory
├── requirements.txt       # Project dependencies
└── .env                   # Environment variables (API keys)
```

## Setup

1. Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the application:

```
python app.py
```

For custom number of reference image inputs:

```
python app.py --num-images 6
```

To create a shareable link:

```
python app.py --share
```

## Usage

1. Enter a detailed text prompt describing the image you want to generate
2. (Optional) Upload one or more reference images to guide the generation
3. Set the number of parallel generations (1-8)
4. Click "Generate Images"
5. All images will be saved in the "outputs" directory

## Using the App

1. Enter your prompt text describing the image you want to generate
2. Optionally upload up to 4 reference images to guide the style or content
3. Adjust the slider for how many parallel generations you want (1-8)
4. Click "Generate Images" to start the process
5. The generated images will appear below the input form in a grid layout

The app uses Gemini 2.0 Flash experimental image generation model and can generate up to 8 variations simultaneously.

## Batch Processing

The app supports batch processing through the "Number of parallel generations" slider:

- Select a value between 1 and 8 to generate that many versions of your prompt simultaneously
- All images are generated in parallel using the same prompt and reference images
- Results are displayed in a grid layout with up to 4 images per row
- Text responses (if any) will be collected in the "Generation Details" section

This feature is useful for:

- Exploring different variations of the same prompt
- Finding the best result among several options
- Saving time compared to generating images one by one

## Advanced Usage

### Command-line Arguments

The app supports several command-line arguments:

- `--num-images N`: Change the number of reference image inputs (default: 4)
- `--share`: Create a shareable public link (useful for demos)

Example:

```
# Run with 2 reference image inputs
python app.py --num-images 2

# Run with 6 reference image inputs
python app.py --num-images 6

# Create a public shareable link
python app.py --share
```

### Examples

The app includes several example prompts to get you started. Click on any example to load it into the prompt field.

## Limitations

- The Gemini 2.0 model has a maximum context size, so very large images might cause errors
- Generation can take a while depending on server load and how many parallel runs you request
- Running 8 generations simultaneously may be resource-intensive depending on your system
- The model may occasionally return text responses instead of images for certain prompts
