from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

class HybridClassifier:
    """Hybrid classifier that combines rule-based and ML predictions."""
    
    def __init__(self, rule_high_confidence_threshold: float = 0.7,
                 rule_low_confidence_threshold: float = 0.3,
                 ml_weight: float = 0.6, rule_weight: float = 0.4):
        self.rule_high_confidence_threshold = rule_high_confidence_threshold
        self.rule_low_confidence_threshold = rule_low_confidence_threshold
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        
        self.logger = logging.getLogger(__name__)
    
    def predict(self, rule_predictions: List[Dict[str, Any]], 
                ml_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine rule-based and ML predictions using hybrid decision system.
        
        Args:
            rule_predictions: Predictions from rule-based classifier
            ml_predictions: Predictions from ML classifier
            
        Returns:
            List of final hybrid predictions
        """
        if len(rule_predictions) != len(ml_predictions):
            raise ValueError("Rule and ML predictions must have same length")
        
        hybrid_predictions = []
        
        for i, (rule_pred, ml_pred) in enumerate(zip(rule_predictions, ml_predictions)):
            hybrid_pred = self._combine_predictions(rule_pred, ml_pred, i)
            hybrid_predictions.append(hybrid_pred)
        
        # Post-process to ensure hierarchical consistency
        hybrid_predictions = self._apply_post_processing(hybrid_predictions)
        
        self.logger.info(f"Generated hybrid predictions for {len(hybrid_predictions)} blocks")
        return hybrid_predictions
    
    def _combine_predictions(self, rule_pred: Dict[str, Any], 
                           ml_pred: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Combine individual rule and ML predictions."""
        # Start with the original block data
        combined = {**rule_pred}
        
        rule_confidence = rule_pred.get("rule_confidence", 0.0)
        ml_confidence = ml_pred.get("ml_confidence", 0.0)
        
        rule_level = rule_pred.get("rule_predicted_level")
        ml_level = ml_pred.get("ml_predicted_level")
        
        rule_is_heading = rule_pred.get("rule_is_heading", False)
        ml_is_heading = ml_pred.get("ml_is_heading", False)
        
        # Decision logic based on confidence levels
        if rule_confidence >= self.rule_high_confidence_threshold:
            # High confidence in rules - use rule prediction
            final_level = rule_level
            final_is_heading = rule_is_heading
            final_confidence = rule_confidence
            decision_method = "rule_high_confidence"
            
        elif rule_confidence <= self.rule_low_confidence_threshold:
            # Low confidence in rules - use ML prediction
            final_level = ml_level
            final_is_heading = ml_is_heading
            final_confidence = ml_confidence
            decision_method = "ml_dominant"
            
        else:
            # Medium confidence - weighted combination
            final_level, final_is_heading, final_confidence = self._weighted_combination(
                rule_level, rule_is_heading, rule_confidence,
                ml_level, ml_is_heading, ml_confidence
            )
            decision_method = "weighted_combination"
        
        # Add hybrid prediction results
        combined.update({
            "final_predicted_level": final_level,
            "final_is_heading": final_is_heading,
            "final_confidence": final_confidence,
            "decision_method": decision_method,
            "ml_predicted_level": ml_level,
            "ml_is_heading": ml_is_heading,
            "ml_confidence": ml_confidence
        })
        
        return combined
    
    def _weighted_combination(self, rule_level: Optional[str], rule_is_heading: bool, rule_conf: float,
                            ml_level: Optional[str], ml_is_heading: bool, ml_conf: float) -> Tuple[Optional[str], bool, float]:
        """Combine predictions using weighted approach."""
        
        # If both agree on non-heading
        if not rule_is_heading and not ml_is_heading:
            combined_conf = (rule_conf * self.rule_weight + ml_conf * self.ml_weight)
            return None, False, combined_conf
        
        # If both agree on heading but different levels
        if rule_is_heading and ml_is_heading:
            if rule_level == ml_level:
                # Same level - high confidence
                combined_conf = (rule_conf * self.rule_weight + ml_conf * self.ml_weight)
                return rule_level, True, combined_conf
            else:
                # Different levels - choose based on confidence
                if rule_conf * self.rule_weight > ml_conf * self.ml_weight:
                    return rule_level, True, rule_conf * 0.8  # Reduce confidence due to disagreement
                else:
                    return ml_level, True, ml_conf * 0.8
        
        # One thinks heading, other doesn't
        if rule_is_heading and not ml_is_heading:
            # Rule says heading, ML says not
            if rule_conf > 0.6:  # Strong rule evidence
                return rule_level, True, rule_conf * 0.7
            else:
                return None, False, ml_conf * 0.8
        
        if not rule_is_heading and ml_is_heading:
            # ML says heading, rule says not
            if ml_conf > 0.6:  # Strong ML evidence
                return ml_level, True, ml_conf * 0.7
            else:
                return None, False, rule_conf * 0.8
        
        # Fallback
        return None, False, 0.5
    
    def _apply_post_processing(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply post-processing rules to ensure document structure validity."""
        processed = predictions.copy()
        
        # Apply hierarchical consistency
        processed = self._ensure_hierarchical_consistency(processed)
        
        # Filter false positives
        processed = self._filter_false_positives(processed)
        
        # Validate page number consistency
        processed = self._validate_page_consistency(processed)
        
        return processed
    
    def _ensure_hierarchical_consistency(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure heading hierarchy makes sense (no H3 directly after H1)."""
        processed = predictions.copy()
        last_heading_level = None
        
        for i, pred in enumerate(processed):
            if pred.get("final_is_heading", False):
                current_level = pred.get("final_predicted_level")
                
                if current_level and last_heading_level:
                    # Check for invalid jumps (e.g., H1 -> H3)
                    level_values = {"H1": 1, "H2": 2, "H3": 3}
                    current_value = level_values.get(current_level, 2)
                    last_value = level_values.get(last_heading_level, 2)
                    
                    # If jumping more than one level down, adjust
                    if current_value > last_value + 1:
                        # Reduce the level or confidence
                        if pred.get("final_confidence", 0) < 0.8:
                            # Lower confidence - adjust level
                            adjusted_level = f"H{last_value + 1}"
                            processed[i]["final_predicted_level"] = adjusted_level
                            processed[i]["final_confidence"] *= 0.8
                            processed[i]["hierarchy_adjusted"] = True
                
                last_heading_level = processed[i].get("final_predicted_level")
        
        return processed
    
    def _filter_false_positives(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter obvious false positives using context analysis."""
        processed = predictions.copy()
        
        for i, pred in enumerate(processed):
            if pred.get("final_is_heading", False):
                text = pred.get("text", "")
                
                # Filter very short text (likely formatting artifacts)
                if len(text.strip()) < 3:
                    processed[i]["final_is_heading"] = False
                    processed[i]["final_predicted_level"] = None
                    processed[i]["final_confidence"] *= 0.5
                    processed[i]["filtered_reason"] = "too_short"
                    continue
                
                # Filter special characters only
                if not any(c.isalnum() for c in text):
                    processed[i]["final_is_heading"] = False
                    processed[i]["final_predicted_level"] = None
                    processed[i]["final_confidence"] *= 0.5
                    processed[i]["filtered_reason"] = "no_alphanumeric"
                    continue
                
                # Filter very long text (unlikely to be heading)
                if len(text) > 200:
                    if processed[i].get("final_confidence", 0) < 0.8:
                        processed[i]["final_is_heading"] = False
                        processed[i]["final_predicted_level"] = None
                        processed[i]["final_confidence"] *= 0.6
                        processed[i]["filtered_reason"] = "too_long"
        
        return processed
    
    def _validate_page_consistency(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that headings are reasonably distributed across pages."""
        processed = predictions.copy()
        
        # Count headings per page
        page_heading_counts = {}
        for pred in processed:
            if pred.get("final_is_heading", False):
                page = pred.get("page", 1)
                page_heading_counts[page] = page_heading_counts.get(page, 0) + 1
        
        # Flag pages with excessive headings (likely false positives)
        for page, count in page_heading_counts.items():
            if count > 10:  # More than 10 headings per page seems excessive
                # Reduce confidence for lower-confidence headings on this page
                for i, pred in enumerate(processed):
                    if (pred.get("page") == page and 
                        pred.get("final_is_heading", False) and
                        pred.get("final_confidence", 0) < 0.6):
                        processed[i]["final_confidence"] *= 0.7
                        processed[i]["page_excess_warning"] = True
        
        return processed
    
    def get_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the hybrid predictions."""
        stats = {
            "total_blocks": len(predictions),
            "predicted_headings": sum(1 for p in predictions if p.get("final_is_heading", False)),
            "decision_methods": {},
            "heading_levels": {"H1": 0, "H2": 0, "H3": 0},
            "average_confidence": 0.0,
            "high_confidence_count": 0,
            "low_confidence_count": 0,
            "hierarchy_adjustments": 0,
            "filtered_predictions": 0
        }
        
        confidences = []
        
        for pred in predictions:
            # Decision method counts
            method = pred.get("decision_method", "unknown")
            stats["decision_methods"][method] = stats["decision_methods"].get(method, 0) + 1
            
            # Heading level counts
            if pred.get("final_is_heading", False):
                level = pred.get("final_predicted_level")
                if level in stats["heading_levels"]:
                    stats["heading_levels"][level] += 1
            
            # Confidence statistics
            conf = pred.get("final_confidence", 0.0)
            confidences.append(conf)
            
            if conf > 0.7:
                stats["high_confidence_count"] += 1
            elif conf < 0.3:
                stats["low_confidence_count"] += 1
            
            # Post-processing statistics
            if pred.get("hierarchy_adjusted", False):
                stats["hierarchy_adjustments"] += 1
            
            if pred.get("filtered_reason"):
                stats["filtered_predictions"] += 1
        
        stats["average_confidence"] = np.mean(confidences) if confidences else 0.0
        
        return stats 