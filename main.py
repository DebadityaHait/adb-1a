import os
import sys
import glob
import json
from pathlib import Path

# Add the 'src' directory to the Python path to allow for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the main processing logic from inference_pipeline
from src.inference_pipeline import InferencePipeline

def run():
    """
    Main function to process all PDFs in the input directory and save
    the JSON output in the output directory.
    """
    input_dir = '/app/input'
    output_dir = '/app/output'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching for PDF files in {input_dir}...")
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # Initialize the inference pipeline
    try:
        pipeline = InferencePipeline(models_dir='./models', use_multilingual=True)
        print("Inference pipeline initialized successfully.")
    except Exception as e:
        print(f"Error initializing inference pipeline: {e}")
        return

    for pdf_path in pdf_files:
        try:
            print(f"Processing {pdf_path}...")
            
            # Process the PDF using the inference pipeline
            result_json = pipeline.process_pdf(pdf_path)

            # Determine the output filename
            pdf_filename = Path(pdf_path).stem
            output_path = os.path.join(output_dir, f"{pdf_filename}.json")

            # Save the output
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            
            print(f"Successfully processed and saved output to {output_path}")

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    run() 