#!/usr/bin/env python3
"""
Test script to demonstrate multilingual support for PDF heading extraction.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pdf_extractor import PDFExtractor
from src.feature_extractor import FeatureExtractor
from src.multilingual_support import MultilingualSupport
from src.enhanced_rule_classifier import EnhancedRuleBasedClassifier

def test_multilingual_on_hindi():
    """Test the multilingual system on Hindi PDF."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("MULTILINGUAL PDF HEADING DETECTION TEST")
    print("=" * 60)
    
    # Initialize components
    pdf_extractor = PDFExtractor()
    feature_extractor = FeatureExtractor()
    multilingual = MultilingualSupport()
    enhanced_classifier = EnhancedRuleBasedClassifier()
    
    # Test on Hindi PDF
    hindi_pdf = "testing/hindi.pdf"
    
    print(f"\n1. Testing on: {hindi_pdf}")
    print("-" * 40)
    
    # Extract text blocks
    text_blocks = pdf_extractor.extract_text_blocks(hindi_pdf)
    print(f"✓ Extracted {len(text_blocks)} text blocks")
    
    # Show some sample text
    print("\nSample text blocks:")
    for i, block in enumerate(text_blocks[:5]):
        text = block["text"][:50] + "..." if len(block["text"]) > 50 else block["text"]
        script = multilingual.detect_script(block["text"])
        print(f"  {i+1}. [{script}] '{text}'")
    
    # Calculate document statistics
    doc_stats = pdf_extractor.get_document_stats(text_blocks)
    
    # Extract features
    featured_blocks = feature_extractor.extract_features(text_blocks, doc_stats)
    
    # Get language statistics
    lang_stats = multilingual.get_language_stats(featured_blocks)
    print(f"\n2. Document Language Analysis:")
    print(f"   Primary script: {lang_stats['primary_script']}")
    print(f"   Script distribution: {lang_stats['script_distribution']}")
    print(f"   Is multilingual: {lang_stats['is_multilingual']}")
    
    # Test multilingual keyword detection
    print(f"\n3. Testing Multilingual Keyword Detection:")
    test_texts = [
        "अध्याय १: परिचय",  # Hindi chapter
        "भाग २: विषय",      # Hindi section  
        "सारांश",           # Hindi summary
        "Chapter 1: Introduction",  # English
        "ऊँ चा खडा हिमालय",  # Sample Hindi text from PDF
    ]
    
    for text in test_texts:
        keyword_score = multilingual.is_heading_keyword_multilingual(text)
        script = multilingual.detect_script(text)
        print(f"   '{text}' -> Script: {script}, Keyword Score: {keyword_score:.2f}")
    
    # Apply enhanced classification
    print(f"\n4. Applying Enhanced Classification:")
    enhanced_predictions = enhanced_classifier.predict(featured_blocks)
    
    # Find potential headings
    headings_found = []
    for pred in enhanced_predictions:
        if pred.get("rule_is_heading", False):
            headings_found.append({
                "text": pred["text"],
                "level": pred["rule_predicted_level"],
                "confidence": pred["rule_confidence"],
                "script": pred.get("detected_script", "unknown"),
                "page": pred["page"]
            })
    
    print(f"✓ Found {len(headings_found)} potential headings with enhanced classifier")
    
    # Show results
    if headings_found:
        print(f"\n5. Detected Headings:")
        for i, heading in enumerate(headings_found[:10]):  # Show top 10
            print(f"   {i+1}. [{heading['level']}] '{heading['text'][:60]}...' ")
            print(f"       Script: {heading['script']}, Confidence: {heading['confidence']:.3f}, Page: {heading['page']}")
    else:
        print(f"\n5. No headings detected with current thresholds")
        print("   Showing top candidates:")
        
        # Show top candidates by confidence
        candidates = sorted(enhanced_predictions, 
                          key=lambda x: x.get("rule_confidence", 0), reverse=True)[:10]
        
        for i, candidate in enumerate(candidates):
            conf = candidate.get("rule_confidence", 0)
            script = multilingual.detect_script(candidate["text"])
            text = candidate["text"][:50] + "..." if len(candidate["text"]) > 50 else candidate["text"]
            print(f"   {i+1}. '{text}'")
            print(f"       Script: {script}, Confidence: {conf:.3f}, Enhanced Scores: {candidate.get('enhanced_rule_scores', {})}")

def test_script_detection():
    """Test script detection on various languages."""
    print(f"\n" + "=" * 60)
    print("SCRIPT DETECTION TEST")
    print("=" * 60)
    
    multilingual = MultilingualSupport()
    
    test_cases = [
        ("Hello World", "latin"),
        ("अध्याय परिचय", "devanagari"), 
        ("章节介绍", "chinese"),
        ("مقدمة", "arabic"),
        ("Глава", "cyrillic"),
        ("Κεφάλαιο", "greek"),
        ("Mixed अध्याय Chapter", "mixed"),
        ("123456", "numeric"),
    ]
    
    print("\nScript Detection Results:")
    for text, expected in test_cases:
        detected = multilingual.detect_script(text)
        status = "✓" if detected == expected or expected == "mixed" else "✗"
        print(f"   {status} '{text}' -> Detected: {detected} (Expected: {expected})")

if __name__ == "__main__":
    test_multilingual_on_hindi()
    test_script_detection()
    
    print(f"\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    print("\nTo integrate multilingual support:")
    print("1. Replace RuleBasedClassifier with EnhancedRuleBasedClassifier")
    print("2. Add multilingual training data for better ML model")
    print("3. Adjust confidence thresholds based on language detection")
    print("4. Consider script-specific post-processing rules") 