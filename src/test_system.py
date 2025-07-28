#!/usr/bin/env python3
"""
Test script to validate the PDF heading extraction system.

This script runs basic tests to ensure all components are working correctly.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from src.pdf_extractor import PDFExtractor
        from src.ground_truth_aligner import GroundTruthAligner
        from src.feature_extractor import FeatureExtractor
        from src.rule_based_classifier import RuleBasedClassifier
        from src.ml_classifier import MLClassifier
        from src.hybrid_classifier import HybridClassifier
        from src.training_pipeline import TrainingPipeline
        from src.inference_pipeline import InferencePipeline
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_pdf_extraction():
    """Test PDF extraction on a sample file."""
    print("Testing PDF extraction...")
    
    try:
        from src.pdf_extractor import PDFExtractor
        
        extractor = PDFExtractor()
        
        # Test with first available PDF
        pdf_dir = Path("pdfs")
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if pdf_files:
                pdf_path = pdf_files[0]
                text_blocks = extractor.extract_text_blocks(str(pdf_path))
                
                if text_blocks:
                    print(f"✓ Extracted {len(text_blocks)} text blocks from {pdf_path.name}")
                    
                    # Test document stats
                    doc_stats = extractor.get_document_stats(text_blocks)
                    print(f"✓ Document stats: {doc_stats.get('total_pages', 0)} pages, "
                          f"avg font size: {doc_stats.get('avg_font_size', 0):.1f}")
                    return True
                else:
                    print("✗ No text blocks extracted")
                    return False
            else:
                print("✗ No PDF files found in pdfs/ directory")
                return False
        else:
            print("✗ pdfs/ directory not found")
            return False
            
    except Exception as e:
        print(f"✗ PDF extraction error: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction."""
    print("Testing feature extraction...")
    
    try:
        from src.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Create sample text block
        sample_blocks = [{
            "text": "Chapter 1: Introduction",
            "page": 1,
            "font_name": "Arial-Bold",
            "font_size": 14.0,
            "font_flags": 16,  # Bold flag
            "bbox": [100, 200, 300, 220],
            "x0": 100, "y0": 200, "x1": 300, "y1": 220,
            "width": 200, "height": 20
        }]
        
        doc_stats = {"avg_font_size": 12.0, "max_font_size": 16.0, 
                    "min_font_size": 10.0, "total_pages": 10}
        
        featured_blocks = extractor.extract_features(sample_blocks, doc_stats)
        
        if featured_blocks and "font_size_ratio" in featured_blocks[0]:
            feature_count = len(extractor.get_feature_names())
            print(f"✓ Feature extraction successful, {feature_count} features")
            return True
        else:
            print("✗ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"✗ Feature extraction error: {e}")
        return False

def test_rule_based_classifier():
    """Test rule-based classifier."""
    print("Testing rule-based classifier...")
    
    try:
        from src.rule_based_classifier import RuleBasedClassifier
        
        classifier = RuleBasedClassifier()
        
        # Create sample block with features
        sample_block = {
            "text": "Chapter 1: Introduction",
            "font_size_ratio": 1.3,  # Larger than average
            "is_bold": True,
            "is_title_case": True,
            "indentation": 50,
            "spacing_before": 20,
            "text_length": 23,
            "word_count": 3,
            "ends_with_colon": False,
            "is_first_on_page": True
        }
        
        predictions = classifier.predict([sample_block])
        
        if predictions and "rule_confidence" in predictions[0]:
            confidence = predictions[0]["rule_confidence"]
            is_heading = predictions[0]["rule_is_heading"]
            print(f"✓ Rule-based classification: heading={is_heading}, confidence={confidence:.3f}")
            return True
        else:
            print("✗ Rule-based classification failed")
            return False
            
    except Exception as e:
        print(f"✗ Rule-based classifier error: {e}")
        return False

def test_ground_truth_alignment():
    """Test ground truth alignment."""
    print("Testing ground truth alignment...")
    
    try:
        from src.ground_truth_aligner import GroundTruthAligner
        
        aligner = GroundTruthAligner()
        
        # Test with first available ground truth file
        gt_dir = Path("expected_outputs")
        if gt_dir.exists():
            gt_files = list(gt_dir.glob("*.json"))
            if gt_files:
                gt_file = gt_files[0]
                ground_truth = aligner.load_ground_truth(str(gt_file))
                
                if ground_truth:
                    print(f"✓ Loaded {len(ground_truth)} ground truth headings from {gt_file.name}")
                    
                    # Test text normalization
                    sample_text = "Chapter 1:  Introduction   \n"
                    normalized = aligner.normalize_text(sample_text)
                    print(f"✓ Text normalization: '{sample_text.strip()}' -> '{normalized}'")
                    return True
                else:
                    print("✗ No ground truth data loaded")
                    return False
            else:
                print("✗ No ground truth files found")
                return False
        else:
            print("✗ expected_outputs/ directory not found")
            return False
            
    except Exception as e:
        print(f"✗ Ground truth alignment error: {e}")
        return False

def test_ml_classifier():
    """Test ML classifier initialization."""
    print("Testing ML classifier...")
    
    try:
        from src.ml_classifier import MLClassifier
        
        classifier = MLClassifier()
        
        # Test feature names
        feature_names = ["font_size_ratio", "is_bold", "text_length"]
        
        # Create sample training data
        sample_data = [
            {"font_size_ratio": 1.3, "is_bold": True, "text_length": 20, 
             "is_heading": True, "heading_level": "H1"},
            {"font_size_ratio": 1.0, "is_bold": False, "text_length": 100, 
             "is_heading": False, "heading_level": None}
        ]
        
        X, y = classifier.prepare_training_data(sample_data, feature_names)
        
        if X.shape[0] == 2 and X.shape[1] == 3:
            print(f"✓ ML classifier data preparation: {X.shape[0]} samples, {X.shape[1]} features")
            return True
        else:
            print(f"✗ ML classifier data preparation failed: shape {X.shape}")
            return False
            
    except Exception as e:
        print(f"✗ ML classifier error: {e}")
        return False

def test_system_integration():
    """Test overall system integration."""
    print("Testing system integration...")
    
    try:
        # Test that training pipeline can be initialized
        from src.training_pipeline import TrainingPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = TrainingPipeline(
                pdfs_dir="pdfs",
                ground_truth_dir="expected_outputs", 
                models_dir=temp_dir
            )
            
            print("✓ Training pipeline initialization successful")
        
        # Test that inference pipeline can be initialized
        from src.inference_pipeline import InferencePipeline
        
        # This will fail to load models, but should not crash during init
        try:
            inference = InferencePipeline(models_dir="nonexistent")
        except:
            # Expected - no models exist yet
            pass
        
        print("✓ System integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ System integration error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("PDF HEADING EXTRACTION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        test_imports,
        test_pdf_extraction,
        test_feature_extraction,
        test_rule_based_classifier,
        test_ground_truth_alignment,
        test_ml_classifier,
        test_system_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready for training and inference.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 