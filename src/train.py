#!/usr/bin/env python3
"""
Main training script for PDF heading extraction system.

This script trains the hybrid rule-based + ML system for extracting
hierarchical headings from PDF documents.
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training_pipeline import TrainingPipeline

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PDF heading extraction system")
    parser.add_argument("--pdfs-dir", default="pdfs", 
                       help="Directory containing training PDF files")
    parser.add_argument("--ground-truth-dir", default="expected_outputs",
                       help="Directory containing ground truth JSON files")
    parser.add_argument("--models-dir", default="models",
                       help="Directory to save trained models")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(
        pdfs_dir=args.pdfs_dir,
        ground_truth_dir=args.ground_truth_dir,
        models_dir=args.models_dir
    )
    
    # Setup logging
    pipeline.setup_logging(args.log_level)
    
    try:
        print("Starting PDF heading extraction training...")
        print(f"PDFs directory: {args.pdfs_dir}")
        print(f"Ground truth directory: {args.ground_truth_dir}")
        print(f"Models will be saved to: {args.models_dir}")
        print("-" * 60)
        
        # Run complete training
        results = pipeline.run_complete_training()
        
        # Print results summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\nDataset Statistics:")
        print(f"  Total training blocks: {results['total_training_blocks']}")
        
        if 'dataset_analysis' in results:
            dist = results['dataset_analysis']['heading_distribution']
            print(f"  Heading distribution:")
            print(f"    H1: {dist.get('H1', 0)}")
            print(f"    H2: {dist.get('H2', 0)}")
            print(f"    H3: {dist.get('H3', 0)}")
            print(f"    Non-headings: {dist.get('non_heading', 0)}")
        
        if 'ml_training' in results:
            ml_results = results['ml_training']
            print(f"\nML Classifier Performance:")
            print(f"  Cross-validation F1: {ml_results.get('cv_mean_score', 0):.3f} ± {ml_results.get('cv_std_score', 0):.3f}")
            
            if 'top_features' in ml_results:
                print(f"  Top features: {[f[0] for f in ml_results['top_features'][:5]]}")
        
        if 'hybrid_evaluation' in results:
            hybrid_results = results['hybrid_evaluation']
            if 'performance_metrics' in hybrid_results:
                overall = hybrid_results['performance_metrics'].get('overall', {})
                print(f"\nHybrid System Performance:")
                print(f"  Overall F1-score: {overall.get('f1_score', 0):.3f}")
                print(f"  Overall Precision: {overall.get('precision', 0):.3f}")
                print(f"  Overall Recall: {overall.get('recall', 0):.3f}")
                print(f"  Overall Accuracy: {overall.get('accuracy', 0):.3f}")
        
        # Save detailed results
        results_file = Path(args.models_dir) / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"\nModels saved to: {args.models_dir}")
        print("\nTraining completed! You can now run inference with:")
        print(f"python inference.py --models-dir {args.models_dir}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 