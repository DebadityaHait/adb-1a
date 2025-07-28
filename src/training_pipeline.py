import os
import json
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from pathlib import Path

from .pdf_extractor import PDFExtractor
from .ground_truth_aligner import GroundTruthAligner
from .feature_extractor import FeatureExtractor
from .rule_based_classifier import RuleBasedClassifier
from .enhanced_rule_classifier import EnhancedRuleBasedClassifier
from .ml_classifier import MLClassifier
from .hybrid_classifier import HybridClassifier

class TrainingPipeline:
    """Complete training pipeline for the PDF heading extraction system."""
    
    def __init__(self, pdfs_dir: str = "pdfs", ground_truth_dir: str = "expected_outputs",
                 models_dir: str = "models", use_multilingual: bool = True):
        self.pdfs_dir = Path(pdfs_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.models_dir = Path(models_dir)
        self.use_multilingual = use_multilingual
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.ground_truth_aligner = GroundTruthAligner()
        self.feature_extractor = FeatureExtractor()
        
        # Use enhanced or original classifier
        if use_multilingual:
            self.rule_classifier = EnhancedRuleBasedClassifier()
        else:
            from .rule_based_classifier import RuleBasedClassifier
            self.rule_classifier = RuleBasedClassifier()
            
        self.ml_classifier = MLClassifier()
        self.hybrid_classifier = HybridClassifier()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def run_complete_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        self.logger.info("Starting complete training pipeline...")
        
        results = {}
        
        # Step 1: Load and process all training PDFs
        self.logger.info("Step 1: Processing training PDFs...")
        all_labeled_blocks = []
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            pdf_id = pdf_file.stem
            gt_file = self.ground_truth_dir / f"{pdf_id}.json"
            
            if not gt_file.exists():
                self.logger.warning(f"Ground truth file not found for {pdf_id}")
                continue
            
            # Process single PDF
            labeled_blocks = self._process_single_pdf(pdf_file, gt_file)
            all_labeled_blocks.extend(labeled_blocks)
            
            self.logger.info(f"Processed {pdf_file.name}: {len(labeled_blocks)} blocks")
        
        results["total_training_blocks"] = len(all_labeled_blocks)
        
        # Step 2: Analyze dataset
        self.logger.info("Step 2: Analyzing dataset...")
        dataset_stats = self._analyze_dataset(all_labeled_blocks)
        results["dataset_analysis"] = dataset_stats
        
        # Step 3: Train ML classifier
        self.logger.info("Step 3: Training ML classifier...")
        ml_metrics = self._train_ml_classifier(all_labeled_blocks)
        results["ml_training"] = ml_metrics
        
        # Step 4: Evaluate hybrid system
        self.logger.info("Step 4: Evaluating hybrid system...")
        hybrid_metrics = self._evaluate_hybrid_system(all_labeled_blocks)
        results["hybrid_evaluation"] = hybrid_metrics
        
        # Step 5: Save models
        self.logger.info("Step 5: Saving trained models...")
        self._save_models()
        results["models_saved"] = True
        
        self.logger.info("Training pipeline completed successfully!")
        return results
    
    def _process_single_pdf(self, pdf_file: Path, gt_file: Path) -> List[Dict[str, Any]]:
        """Process a single PDF with its ground truth."""
        # Extract text blocks
        text_blocks = self.pdf_extractor.extract_text_blocks(str(pdf_file))
        if not text_blocks:
            return []
        
        # Calculate document statistics
        doc_stats = self.pdf_extractor.get_document_stats(text_blocks)
        
        # Extract features
        featured_blocks = self.feature_extractor.extract_features(text_blocks, doc_stats)
        
        # Load ground truth
        ground_truth = self.ground_truth_aligner.load_ground_truth(str(gt_file))
        
        # Align with ground truth
        labeled_blocks = self.ground_truth_aligner.create_labeled_dataset(
            featured_blocks, ground_truth
        )
        
        return labeled_blocks
    
    def _analyze_dataset(self, labeled_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the complete dataset."""
        stats = {
            "total_blocks": len(labeled_blocks),
            "heading_distribution": {"H1": 0, "H2": 0, "H3": 0, "non_heading": 0},
            "feature_statistics": {},
            "document_count": len(set(block.get("pdf_source", "unknown") for block in labeled_blocks))
        }
        
        # Count heading types
        for block in labeled_blocks:
            if block.get("is_heading", False):
                level = block.get("heading_level", "H2")
                if level in stats["heading_distribution"]:
                    stats["heading_distribution"][level] += 1
            else:
                stats["heading_distribution"]["non_heading"] += 1
        
        # Feature statistics
        feature_names = self.feature_extractor.get_feature_names()
        for feature_name in feature_names[:10]:  # Top 10 features
            values = [block.get(feature_name, 0) for block in labeled_blocks]
            if values:
                stats["feature_statistics"][feature_name] = {
                    "mean": float(pd.Series(values).mean()),
                    "std": float(pd.Series(values).std()),
                    "min": float(pd.Series(values).min()),
                    "max": float(pd.Series(values).max())
                }
        
        self.logger.info(f"Dataset analysis: {stats['heading_distribution']}")
        return stats
    
    def _train_ml_classifier(self, labeled_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the ML classifier component."""
        feature_names = self.feature_extractor.get_feature_names()
        
        # Prepare training data
        X, y = self.ml_classifier.prepare_training_data(labeled_blocks, feature_names)
        
        # Train model
        training_metrics = self.ml_classifier.train(X, y, balance_classes=True)
        
        # Evaluate on training data (for basic validation)
        eval_metrics = self.ml_classifier.evaluate(X, y)
        
        combined_metrics = {
            **training_metrics,
            "evaluation": eval_metrics
        }
        
        return combined_metrics
    
    def _evaluate_hybrid_system(self, labeled_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the complete hybrid system."""
        # Apply rule-based predictions
        rule_predictions = self.rule_classifier.predict(labeled_blocks)
        
        # Apply ML predictions
        ml_predictions = self.ml_classifier.predict(labeled_blocks)
        
        # Combine with hybrid system
        hybrid_predictions = self.hybrid_classifier.predict(rule_predictions, ml_predictions)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(hybrid_predictions)
        
        # Get hybrid system statistics
        hybrid_stats = self.hybrid_classifier.get_statistics(hybrid_predictions)
        
        return {
            "performance_metrics": metrics,
            "hybrid_statistics": hybrid_stats
        }
    
    def _calculate_performance_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics against ground truth."""
        # Get all unique heading levels in the data
        all_levels = set()
        for pred in predictions:
            gt_level = pred.get("heading_level")
            pred_level = pred.get("final_predicted_level") 
            if gt_level: 
                all_levels.add(gt_level)
            if pred_level:
                all_levels.add(pred_level)
        
        # Filter to only include expected levels (H1, H2, H3)
        valid_levels = [level for level in all_levels if level in ["H1", "H2", "H3"]]
        
        true_positives = {level: 0 for level in valid_levels}
        false_positives = {level: 0 for level in valid_levels}
        false_negatives = {level: 0 for level in valid_levels}
        true_negatives = 0
        
        for pred in predictions:
            gt_is_heading = pred.get("is_heading", False)
            gt_level = pred.get("heading_level")
            
            pred_is_heading = pred.get("final_is_heading", False)
            pred_level = pred.get("final_predicted_level")
            
            # Only process valid levels (H1, H2, H3)
            if gt_is_heading and pred_is_heading:
                if gt_level == pred_level and gt_level in valid_levels:
                    true_positives[gt_level] += 1
                else:
                    # Wrong level - only count if both levels are valid
                    if gt_level in valid_levels:
                        false_negatives[gt_level] += 1
                    if pred_level in valid_levels:
                        false_positives[pred_level] += 1
            elif gt_is_heading and not pred_is_heading:
                if gt_level in valid_levels:
                    false_negatives[gt_level] += 1
            elif not gt_is_heading and pred_is_heading:
                if pred_level in valid_levels:
                    false_positives[pred_level] += 1
            else:
                true_negatives += 1
        
        # Calculate metrics for each level
        metrics = {}
        overall_tp = sum(true_positives.values())
        overall_fp = sum(false_positives.values())
        overall_fn = sum(false_negatives.values())
        
        for level in valid_levels:
            tp = true_positives[level]
            fp = false_positives[level]
            fn = false_negatives[level]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[level] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            }
        
        # Overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics["overall"] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "accuracy": (overall_tp + true_negatives) / len(predictions)
        }
        
        return metrics
    
    def _save_models(self) -> None:
        """Save all trained models."""
        # Save ML model
        ml_model_path = self.models_dir / "ml_classifier.joblib"
        ml_scaler_path = self.models_dir / "ml_scaler.joblib"
        self.ml_classifier.save_model(str(ml_model_path), str(ml_scaler_path))
        
        # Save feature names
        feature_names_path = self.models_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_extractor.get_feature_names(), f)
        
        self.logger.info(f"Models saved to {self.models_dir}")
    
    def load_trained_models(self) -> None:
        """Load previously trained models."""
        ml_model_path = self.models_dir / "ml_classifier.joblib"
        ml_scaler_path = self.models_dir / "ml_scaler.joblib"
        
        if ml_model_path.exists() and ml_scaler_path.exists():
            self.ml_classifier.load_model(str(ml_model_path), str(ml_scaler_path))
            self.logger.info("Trained models loaded successfully")
        else:
            raise FileNotFoundError("Trained models not found. Run training first.")
    
    def setup_logging(self, level: str = "INFO") -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        ) 