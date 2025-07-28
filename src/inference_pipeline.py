import os
import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

from .pdf_extractor import PDFExtractor
from .feature_extractor import FeatureExtractor
from .rule_based_classifier import RuleBasedClassifier
from .enhanced_rule_classifier import EnhancedRuleBasedClassifier
from .ml_classifier import MLClassifier
from .hybrid_classifier import HybridClassifier
from .multilingual_hybrid_classifier import MultilingualHybridClassifier

class InferencePipeline:
    """Complete inference pipeline for processing PDFs and extracting headings."""
    
    def __init__(self, models_dir: str = "models", use_multilingual: bool = True):
        self.models_dir = Path(models_dir)
        self.use_multilingual = use_multilingual
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.feature_extractor = FeatureExtractor()
        
        # Use enhanced or original classifier
        if use_multilingual:
            self.rule_classifier = EnhancedRuleBasedClassifier()
            self.hybrid_classifier = MultilingualHybridClassifier()
        else:
            from .rule_based_classifier import RuleBasedClassifier
            self.rule_classifier = RuleBasedClassifier()
            self.hybrid_classifier = HybridClassifier()
            
        self.ml_classifier = MLClassifier()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load trained models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all trained models and components."""
        try:
            # Load ML classifier
            ml_model_path = self.models_dir / "ml_classifier.joblib"
            ml_scaler_path = self.models_dir / "ml_scaler.joblib"
            
            if ml_model_path.exists() and ml_scaler_path.exists():
                self.ml_classifier.load_model(str(ml_model_path), str(ml_scaler_path))
                self.logger.info("ML classifier loaded successfully")
            else:
                self.logger.warning("ML classifier not found, using rule-based only")
        
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF and extract headings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted headings in the required JSON format
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract text blocks
            self.logger.info(f"Processing PDF: {pdf_path}")
            text_blocks = self.pdf_extractor.extract_text_blocks(pdf_path)
            
            if not text_blocks:
                self.logger.warning(f"No text blocks extracted from {pdf_path}")
                return []
            
            # Step 2: Calculate document statistics
            doc_stats = self.pdf_extractor.get_document_stats(text_blocks)
            
            # Step 3: Extract features
            featured_blocks = self.feature_extractor.extract_features(text_blocks, doc_stats)
            
            # Step 4: Apply rule-based classification
            rule_predictions = self.rule_classifier.predict(featured_blocks)
            
            # Step 5: Apply ML classification (if available)
            if self.ml_classifier.model:
                ml_predictions = self.ml_classifier.predict(featured_blocks)
                
                # Step 6: Combine with hybrid system
                final_predictions = self.hybrid_classifier.predict(rule_predictions, ml_predictions)
            else:
                # Use rule-based only
                final_predictions = rule_predictions
                for pred in final_predictions:
                    pred["final_is_heading"] = pred.get("rule_is_heading", False)
                    pred["final_predicted_level"] = pred.get("rule_predicted_level")
                    pred["final_confidence"] = pred.get("rule_confidence", 0.0)
            
            # Step 7: Convert to output format
            headings = self._convert_to_output_format(final_predictions)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Processed {pdf_path} in {processing_time:.2f}s, found {len(headings)} headings")
            
            return headings
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def _convert_to_output_format(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert predictions to the required JSON output format."""
        headings = []
        
        for pred in predictions:
            if pred.get("final_is_heading", False):
                heading = {
                    "level": pred.get("final_predicted_level"),
                    "text": pred.get("text", "").strip(),
                    "page": pred.get("page", 1)
                }
                
                # Validate heading
                if heading["level"] and heading["text"] and len(heading["text"]) >= 2:
                    headings.append(heading)
        
        # Sort by page number and then by order in document
        headings.sort(key=lambda x: (x["page"], predictions.index(next(
            p for p in predictions 
            if p.get("text") == x["text"] and p.get("page") == x["page"]
        ))))
        
        return headings
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """
        Process all PDFs in a directory and save results.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save JSON results
            file_pattern: Pattern to match PDF files
            
        Returns:
            Processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(input_path.glob(file_pattern))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir}")
            return {"processed": 0, "failed": 0, "total_time": 0}
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process statistics
        stats = {
            "processed": 0,
            "failed": 0,
            "total_headings": 0,
            "total_time": 0,
            "files": {}
        }
        
        total_start_time = time.time()
        
        for pdf_file in pdf_files:
            try:
                file_start_time = time.time()
                
                # Process PDF
                headings = self.process_pdf(str(pdf_file))
                
                # Save results
                output_file = output_path / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(headings, f, indent=2, ensure_ascii=False)
                
                # Update statistics
                processing_time = time.time() - file_start_time
                stats["processed"] += 1
                stats["total_headings"] += len(headings)
                stats["files"][pdf_file.name] = {
                    "headings_found": len(headings),
                    "processing_time": processing_time,
                    "output_file": str(output_file)
                }
                
                self.logger.info(f"Saved {len(headings)} headings to {output_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {str(e)}")
                stats["failed"] += 1
                stats["files"][pdf_file.name] = {
                    "error": str(e),
                    "processing_time": 0
                }
        
        stats["total_time"] = time.time() - total_start_time
        
        # Save processing statistics
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Batch processing completed: {stats['processed']} successful, {stats['failed']} failed")
        self.logger.info(f"Total time: {stats['total_time']:.2f}s, Average: {stats['total_time']/len(pdf_files):.2f}s per file")
        
        return stats
    
    def process_testing_folder(self, testing_dir: str = "testing", 
                             output_dir: str = "outputs") -> Dict[str, Any]:
        """Process all PDFs in the testing folder."""
        self.logger.info("Processing testing folder...")
        return self.batch_process(testing_dir, output_dir)
    
    def validate_performance(self, pdf_path: str, max_pages: int = 50, 
                           max_time: float = 10.0) -> Dict[str, Any]:
        """
        Validate that processing meets performance requirements.
        
        Args:
            pdf_path: Path to test PDF
            max_pages: Maximum pages to test
            max_time: Maximum processing time in seconds
            
        Returns:
            Performance validation results
        """
        start_time = time.time()
        
        # Get document info first
        try:
            import fitz
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except:
            page_count = 1
        
        # Process the PDF
        headings = self.process_pdf(pdf_path)
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        results = {
            "pdf_path": pdf_path,
            "page_count": page_count,
            "headings_found": len(headings),
            "processing_time": processing_time,
            "pages_per_second": page_count / processing_time if processing_time > 0 else 0,
            "meets_time_requirement": processing_time <= max_time,
            "meets_page_requirement": page_count <= max_pages,
            "estimated_50_page_time": processing_time * (50 / max(page_count, 1))
        }
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the loaded system."""
        return {
            "models_directory": str(self.models_dir),
            "ml_classifier_loaded": self.ml_classifier.model is not None,
            "feature_count": len(self.feature_extractor.get_feature_names()),
            "rule_confidence_weights": self.rule_classifier.confidence_weights,
            "hybrid_thresholds": {
                "high_confidence": self.hybrid_classifier.rule_high_confidence_threshold,
                "low_confidence": self.hybrid_classifier.rule_low_confidence_threshold
            }
        } 