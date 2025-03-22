# Gemini Image Generation App

A Gradio web application for generating images using Google's Gemini AI model. This app allows you to:

1. Input a text prompt describing what image you want to generate
2. Optionally upload up to 4 reference images to influence the generation
3. Generate and view the results directly in the browser

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
3. Click "Generate" to create the image
4. The generated image will appear below the input form

The app uses Gemini 2.0 Flash experimental image generation model.

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
- Generation can take a while depending on server load
- The model may occasionally return text responses instead of images for certain prompts
