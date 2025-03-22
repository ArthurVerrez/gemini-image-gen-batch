# Gemini Image Generation App

A Gradio web application for generating images using Google's Gemini AI model. This app allows you to:

1. Input a text prompt describing what image you want to generate
2. Optionally upload up to 4 reference images to influence the generation
3. Generate multiple variants of the same prompt in parallel (up to 8)
4. View all generated results in a grid layout

## Setup

1. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

2. Set up your Gemini API key:

   ```
   export GEMINI_API_KEY="your_api_key_here"
   ```

   Or create a `.env` file with:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. Run the application:

   ```
   python app.py
   ```

4. Open the provided URL in your browser to use the app.

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
