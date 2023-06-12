# PDF Layout Analysis and OCR

This code performs PDF layout analysis and optical character recognition (OCR) using the layoutparser library and Tesseract OCR Engine. It detects the layout of a PDF document and extracts text from specific regions. The code is divided into several sections, each serving a specific purpose.

## Prerequisites
Before running the code, ensure that you have the following dependencies installed:

- poppler-utils: This package provides the pdf2image dependency. You can install it using `sudo apt-get install poppler-utils`. Restart the runtime/kernel after installation.
- tesseract-ocr-eng: This package installs the Tesseract OCR Engine for English language support. Install it using `sudo apt-get install tesseract-ocr-eng`. Restart the runtime/kernel after installation.

You also need to install the required Python libraries by running the following commands:
- `pip install layoutparser torchvision`
- `pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"`
- `pip install pdf2img`
- `pip install "layoutparser[ocr]"`

## Usage
1. Adjust the `pdf_file` variable with the filepath of your input PDF image.
2. The code will perform layout detection using four different models: `model1`, `model2`, `model3`, and `model4`. You can modify these models based on your requirements. Each model is initialized with a specific configuration and label map.
3. The layout detection results are stored in `layout_result1`, `layout_result2`, `layout_result3`, and `layout_result4` variables.
4. The code visualizes the layout detection results by drawing boxes on the input image using `lp.draw_box` function.
5. The code further processes the layout results to extract text blocks from each layout using `text_blocks1`, `text_blocks2`, `text_blocks3`, and `text_blocks4` variables.
6. The OCR process begins by initializing the TesseractAgent using `lp.TesseractAgent(languages='eng')`. This agent will perform OCR on the extracted text blocks.
7. The code crops the image around each text block, performs OCR using Tesseract OCR, and saves the OCR results back to the corresponding text blocks using `ocr_agent.detect(segment_image)`.
8. Finally, the code prints the extracted text and the corresponding bounding box coordinates for each text block in `text_blocks1`, `text_blocks2`, `text_blocks3`, and `text_blocks4`.

Make sure to have the necessary PDF file and adjust the code according to your requirements before running it.

**Note**: Remember to restart the runtime/kernel after installing the required dependencies to ensure they are properly loaded.

Feel free to modify the code and experiment with different models, configurations, and OCR settings to suit your specific needs.