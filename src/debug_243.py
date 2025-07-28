#!/usr/bin/env python3
"""Debug script to analyze 243.pdf specifically."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pdf_extractor import PDFExtractor
from src.feature_extractor import FeatureExtractor
from src.multilingual_support import MultilingualSupport
from src.enhanced_rule_classifier import EnhancedRuleBasedClassifier

def debug_243():
    print("="*60)
    print("DEBUGGING 243.PDF")
    print("="*60)
    
    # Initialize components
    pdf_extractor = PDFExtractor()
    feature_extractor = FeatureExtractor()
    multilingual = MultilingualSupport()
    enhanced_classifier = EnhancedRuleBasedClassifier()
    
    # Extract and analyze
    pdf_path = "testing/243.pdf"
    print(f"\n1. Extracting text from {pdf_path}")
    
    text_blocks = pdf_extractor.extract_text_blocks(pdf_path)
    print(f"   ✓ Extracted {len(text_blocks)} text blocks")
    
    # Show sample text blocks
    print(f"\n2. Sample text blocks (first 10):")
    for i, block in enumerate(text_blocks[:10]):
        text = block["text"][:80].replace('\n', ' ')
        script = multilingual.detect_script(block["text"])
        print(f"   {i+1:2d}. [{script:10}] '{text}...'")
        print(f"       Font: {block.get('font_size', 'N/A'):4}, Bold: {block.get('is_bold', False)}, Page: {block.get('page', 'N/A')}")
    
    # Document stats
    doc_stats = pdf_extractor.get_document_stats(text_blocks)
    print(f"\n3. Document Statistics:")
    print(f"   Average font size: {doc_stats.get('avg_font_size', 'N/A'):.1f}")
    print(f"   Max font size: {doc_stats.get('max_font_size', 'N/A'):.1f}")
    print(f"   Min font size: {doc_stats.get('min_font_size', 'N/A'):.1f}")
    
    # Feature extraction
    featured_blocks = feature_extractor.extract_features(text_blocks, doc_stats)
    
    # Language analysis
    lang_stats = multilingual.get_language_stats(featured_blocks)
    print(f"\n4. Language Analysis:")
    print(f"   Primary script: {lang_stats['primary_script']}")
    print(f"   Script distribution: {lang_stats['script_distribution']}")
    print(f"   Is multilingual: {lang_stats['is_multilingual']}")
    
    # Enhanced classification
    print(f"\n5. Enhanced Rule Classification:")
    enhanced_predictions = enhanced_classifier.predict(featured_blocks)
    
    # Find high confidence candidates
    candidates = []
    for pred in enhanced_predictions:
        confidence = pred.get("rule_confidence", 0)
        if confidence > 0.1:  # Lower threshold to see candidates
            candidates.append({
                "text": pred["text"][:60],
                "confidence": confidence,
                "is_heading": pred.get("rule_is_heading", False),
                "level": pred.get("rule_predicted_level"),
                "script": pred.get("detected_script"),
                "enhanced_scores": pred.get("enhanced_rule_scores", {}),
                "page": pred.get("page", 1)
            })
    
    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    print(f"   Found {len(candidates)} blocks with confidence > 0.1")
    
    if candidates:
        print(f"\n6. Top Candidates by Confidence:")
        for i, cand in enumerate(candidates[:15]):
            status = "✓ HEADING" if cand["is_heading"] else "✗ No heading"
            print(f"   {i+1:2d}. {status} (conf: {cand['confidence']:.3f})")
            print(f"       Text: '{cand['text']}...'")
            print(f"       Level: {cand['level']}, Script: {cand['script']}, Page: {cand['page']}")
            print(f"       Scores: {cand['enhanced_scores']}")
            print()
    else:
        print("   No candidates found with confidence > 0.1")
        
        # Show the highest confidence ones anyway
        all_preds = sorted(enhanced_predictions, key=lambda x: x.get("rule_confidence", 0), reverse=True)
        print(f"\n   Top 10 blocks by confidence (even if very low):")
        for i, pred in enumerate(all_preds[:10]):
            conf = pred.get("rule_confidence", 0)
            text = pred["text"][:50].replace('\n', ' ')
            script = pred.get("detected_script", "unknown")
            print(f"   {i+1:2d}. conf={conf:.4f} [{script}] '{text}...'")
    
    # Check for specific patterns
    print(f"\n7. Pattern Analysis:")
    
    # Look for common heading patterns
    heading_patterns = 0
    for block in text_blocks:
        text = block["text"].strip()
        if text:
            # Check various patterns
            if any(keyword in text.lower() for keyword in ["chapter", "section", "अध्याय", "भाग"]):
                heading_patterns += 1
                print(f"   Found keyword pattern: '{text[:60]}...'")
            elif text.endswith(":") and len(text.split()) <= 5:
                heading_patterns += 1
                print(f"   Found colon pattern: '{text[:60]}...'")
            elif text.isupper() and 5 <= len(text) <= 50:
                heading_patterns += 1
                print(f"   Found uppercase pattern: '{text[:60]}...'")
    
    print(f"   Total potential heading patterns found: {heading_patterns}")
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_243() 