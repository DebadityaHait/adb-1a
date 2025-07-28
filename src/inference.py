#!/usr/bin/env python3
"""
Main inference script for PDF heading extraction system.

This script processes PDFs and extracts hierarchical headings using
the trained hybrid rule-based + ML system.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference_pipeline import InferencePipeline

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Extract headings from PDF files")
    parser.add_argument("--input", "-i", 
                       help="Input PDF file or directory")
    parser.add_argument("--output", "-o", 
                       help="Output JSON file or directory")
    parser.add_argument("--models-dir", default="models",
                       help="Directory containing trained models")
    parser.add_argument("--testing-folder", action="store_true",
                       help="Process all PDFs in testing folder")
    parser.add_argument("--validate-performance", 
                       help="Validate performance on specific PDF")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress output except errors")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = "ERROR"
    else:
        log_level = args.log_level
        
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize inference pipeline
        if not args.quiet:
            print("Initializing PDF heading extraction system...")
            print(f"Models directory: {args.models_dir}")
        
        pipeline = InferencePipeline(models_dir=args.models_dir)
        
        # Show system info
        if not args.quiet:
            system_info = pipeline.get_system_info()
            print(f"ML classifier loaded: {system_info['ml_classifier_loaded']}")
            print(f"Feature count: {system_info['feature_count']}")
            print("-" * 60)
        
        # Handle different modes
        if args.testing_folder:
            # Process testing folder
            if not args.quiet:
                print("Processing testing folder...")
            
            stats = pipeline.process_testing_folder()
            
            if not args.quiet:
                print(f"\nProcessing completed:")
                print(f"  Successfully processed: {stats['processed']} PDFs")
                print(f"  Failed: {stats['failed']} PDFs")
                print(f"  Total headings extracted: {stats['total_headings']}")
                print(f"  Total processing time: {stats['total_time']:.2f}s")
                print(f"  Average time per file: {stats['total_time']/(stats['processed'] + stats['failed']):.2f}s")
                print(f"\nResults saved to: outputs/")
        
        elif args.validate_performance:
            # Validate performance
            if not args.quiet:
                print(f"Validating performance on: {args.validate_performance}")
            
            results = pipeline.validate_performance(args.validate_performance)
            
            print(f"\nPerformance Validation Results:")
            print(f"  PDF: {results['pdf_path']}")
            print(f"  Pages: {results['page_count']}")
            print(f"  Headings found: {results['headings_found']}")
            print(f"  Processing time: {results['processing_time']:.2f}s")
            print(f"  Pages per second: {results['pages_per_second']:.2f}")
            print(f"  Meets time requirement (≤10s): {results['meets_time_requirement']}")
            print(f"  Estimated 50-page time: {results['estimated_50_page_time']:.2f}s")
        
        elif args.input:
            input_path = Path(args.input)
            
            if input_path.is_file():
                # Process single PDF
                if not args.quiet:
                    print(f"Processing single PDF: {args.input}")
                
                headings = pipeline.process_pdf(args.input)
                
                # Save or print results
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(headings, f, indent=2, ensure_ascii=False)
                    
                    if not args.quiet:
                        print(f"Results saved to: {args.output}")
                        print(f"Found {len(headings)} headings")
                else:
                    # Print to stdout
                    print(json.dumps(headings, indent=2, ensure_ascii=False))
            
            elif input_path.is_dir():
                # Process directory
                output_dir = args.output or "outputs"
                
                if not args.quiet:
                    print(f"Processing directory: {args.input}")
                    print(f"Output directory: {output_dir}")
                
                stats = pipeline.batch_process(args.input, output_dir)
                
                if not args.quiet:
                    print(f"\nBatch processing completed:")
                    print(f"  Successfully processed: {stats['processed']} PDFs")
                    print(f"  Failed: {stats['failed']} PDFs")
                    print(f"  Total headings extracted: {stats['total_headings']}")
                    print(f"  Total processing time: {stats['total_time']:.2f}s")
            
            else:
                print(f"Error: Input path '{args.input}' does not exist")
                sys.exit(1)
        
        else:
            # No input specified
            parser.print_help()
            print("\nExamples:")
            print("  # Process testing folder")
            print("  python inference.py --testing-folder")
            print("")
            print("  # Process single PDF")
            print("  python inference.py -i document.pdf -o headings.json")
            print("")
            print("  # Process directory")
            print("  python inference.py -i pdfs/ -o outputs/")
            print("")
            print("  # Validate performance")
            print("  python inference.py --validate-performance document.pdf")
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Make sure to train the model first by running: python train.py")
        sys.exit(1)
    
    except Exception as e:
        print(f"Inference failed with error: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 