from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from .multilingual_support import MultilingualSupport
from .emergency_detector import EmergencyHeadingDetector

class MultilingualHybridClassifier:
    """Multilingual-aware hybrid classifier with adaptive thresholds and emergency detection."""
    
    def __init__(self, rule_high_confidence_threshold: float = 0.7,
                 rule_low_confidence_threshold: float = 0.3,
                 ml_weight: float = 0.6, rule_weight: float = 0.4):
        self.rule_high_confidence_threshold = rule_high_confidence_threshold
        self.rule_low_confidence_threshold = rule_low_confidence_threshold
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        
        self.multilingual = MultilingualSupport()
        self.emergency_detector = EmergencyHeadingDetector()
        self.logger = logging.getLogger(__name__)
    
    def predict(self, rule_predictions: List[Dict[str, Any]], 
                ml_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine predictions with multilingual-aware decision system and emergency detection."""
        if len(rule_predictions) != len(ml_predictions):
            raise ValueError("Rule and ML predictions must have same length")
        
        # Detect document language characteristics
        lang_stats = self._analyze_document_language(rule_predictions)
        
        # Adjust thresholds based on language detection
        adjusted_thresholds = self._adjust_thresholds_for_language(lang_stats)
        
        self.logger.info(f"Language stats: {lang_stats}")
        self.logger.info(f"Adjusted thresholds: {adjusted_thresholds}")
        
        hybrid_predictions = []
        
        for i, (rule_pred, ml_pred) in enumerate(zip(rule_predictions, ml_predictions)):
            hybrid_pred = self._combine_predictions_multilingual(
                rule_pred, ml_pred, i, adjusted_thresholds, lang_stats
            )
            hybrid_predictions.append(hybrid_pred)
        
        # Apply post-processing
        hybrid_predictions = self._apply_post_processing(hybrid_predictions)
        
        # **EMERGENCY DETECTION**: Check if normal methods failed
        headings_found = sum(1 for pred in hybrid_predictions if pred.get("final_is_heading", False))
        
        if headings_found == 0:
            self.logger.info("🚨 No headings found with normal methods, activating emergency detector")
            
            # Try emergency detection
            emergency_headings = self.emergency_detector.detect_emergency_headings(rule_predictions)
            
            if emergency_headings:
                self.logger.info(f"🚨 Emergency detector found {len(emergency_headings)} headings")
                
                # Replace the predictions with emergency ones
                emergency_indices = []
                for emerg_heading in emergency_headings:
                    # Find corresponding index in hybrid_predictions
                    emerg_text = emerg_heading.get("text", "")
                    for i, hybrid_pred in enumerate(hybrid_predictions):
                        if hybrid_pred.get("text", "") == emerg_text:
                            hybrid_predictions[i] = emerg_heading
                            emergency_indices.append(i)
                            break
                
                self.logger.info(f"🚨 Applied emergency detection to {len(emergency_indices)} blocks")
        
        self.logger.info(f"Generated hybrid predictions for {len(hybrid_predictions)} blocks")
        return hybrid_predictions
    
    def _analyze_document_language(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the primary language/script of the document."""
        script_counts = {}
        total_confidence = 0
        
        for pred in predictions:
            text = pred.get("text", "")
            script = self.multilingual.detect_script(text)
            confidence = pred.get("rule_confidence", 0)
            
            if script not in script_counts:
                script_counts[script] = {"count": 0, "total_confidence": 0}
            
            script_counts[script]["count"] += 1
            script_counts[script]["total_confidence"] += confidence
            total_confidence += confidence
        
        # Determine primary script
        primary_script = max(script_counts.keys(), key=lambda x: script_counts[x]["count"])
        
        # Calculate average confidence by script
        avg_confidences = {}
        for script, data in script_counts.items():
            avg_confidences[script] = data["total_confidence"] / max(data["count"], 1)
        
        return {
            "primary_script": primary_script,
            "script_counts": script_counts,
            "avg_confidences": avg_confidences,
            "is_non_latin": primary_script not in ["latin", "unknown"],
            "total_blocks": len(predictions)
        }
    
    def _adjust_thresholds_for_language(self, lang_stats: Dict[str, Any]) -> Dict[str, float]:
        """Adjust confidence thresholds based on detected language."""
        base_high = self.rule_high_confidence_threshold
        base_low = self.rule_low_confidence_threshold
        
        primary_script = lang_stats["primary_script"]
        
        # Lower thresholds for non-Latin scripts (they tend to have lower confidence)
        if primary_script in ["devanagari", "arabic", "chinese"]:
            # Reduce thresholds by 40% for non-Latin scripts
            adjusted_high = base_high * 0.6
            adjusted_low = base_low * 0.6
        elif primary_script == "unknown":
            # Reduce thresholds by 30% for unknown scripts
            adjusted_high = base_high * 0.7
            adjusted_low = base_low * 0.7
        else:
            # Keep original thresholds for Latin scripts
            adjusted_high = base_high
            adjusted_low = base_low
        
        return {
            "high_confidence": adjusted_high,
            "low_confidence": adjusted_low
        }
    
    def _combine_predictions_multilingual(self, rule_pred: Dict[str, Any], 
                                        ml_pred: Dict[str, Any], index: int,
                                        thresholds: Dict[str, float],
                                        lang_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions with multilingual considerations."""
        combined = {**rule_pred}
        
        rule_confidence = rule_pred.get("rule_confidence", 0.0)
        ml_confidence = ml_pred.get("ml_confidence", 0.0)
        
        rule_level = rule_pred.get("rule_predicted_level")
        ml_level = ml_pred.get("ml_predicted_level")
        
        rule_is_heading = rule_pred.get("rule_is_heading", False)
        ml_is_heading = ml_pred.get("ml_is_heading", False)
        
        # Get text and script information
        text = rule_pred.get("text", "")
        detected_script = rule_pred.get("detected_script", self.multilingual.detect_script(text))
        
        # Apply script-specific boosts
        script_boost = self._get_script_boost(detected_script, text, lang_stats)
        adjusted_rule_confidence = min(rule_confidence + script_boost, 1.0)
        
        # Decision logic with adjusted thresholds
        high_threshold = thresholds["high_confidence"]
        low_threshold = thresholds["low_confidence"]
        
        if adjusted_rule_confidence >= high_threshold:
            # High confidence in rules
            final_level = rule_level
            final_is_heading = rule_is_heading
            final_confidence = adjusted_rule_confidence
            decision_method = "rule_high_confidence"
            
        elif adjusted_rule_confidence <= low_threshold:
            # Low confidence in rules - use ML
            final_level = ml_level
            final_is_heading = ml_is_heading
            final_confidence = ml_confidence
            decision_method = "ml_dominant"
            
        else:
            # Medium confidence - weighted combination
            final_level, final_is_heading, final_confidence = self._weighted_combination(
                rule_level, rule_is_heading, adjusted_rule_confidence,
                ml_level, ml_is_heading, ml_confidence
            )
            decision_method = "weighted_combination"
        
        # Add multilingual-specific information
        combined.update({
            "final_predicted_level": final_level,
            "final_is_heading": final_is_heading,
            "final_confidence": final_confidence,
            "decision_method": decision_method,
            "ml_predicted_level": ml_level,
            "ml_is_heading": ml_is_heading,
            "ml_confidence": ml_confidence,
            "script_boost": script_boost,
            "adjusted_rule_confidence": adjusted_rule_confidence,
            "detected_script": detected_script
        })
        
        return combined
    
    def _get_script_boost(self, script: str, text: str, lang_stats: Dict[str, Any]) -> float:
        """Get confidence boost based on script-specific patterns."""
        boost = 0.0
        
        # Boost for script consistency with document
        if script == lang_stats["primary_script"]:
            boost += 0.05
        
        # Script-specific boosts
        if script == "devanagari":
            # Hindi-specific patterns
            if any(char in text for char in ['॥', '।', '॰']):  # Devanagari punctuation
                boost += 0.1
            if len(text.split()) <= 5 and len(text) >= 10:  # Short phrases
                boost += 0.05
                
        elif script == "arabic":
            # Arabic-specific patterns
            if text.endswith('؟') or text.endswith('؛'):
                boost += 0.1
                
        elif script == "chinese":
            # Chinese-specific patterns
            if any(char in text for char in ['。', '？', '！', '；', '：']):
                boost += 0.1
        
        # Universal patterns
        if text.strip() and 5 <= len(text) <= 80:  # Good heading length
            boost += 0.03
            
        return boost
    
    def _weighted_combination(self, rule_level: Optional[str], rule_is_heading: bool, rule_conf: float,
                            ml_level: Optional[str], ml_is_heading: bool, ml_conf: float) -> Tuple[Optional[str], bool, float]:
        """Combine predictions using weighted approach (same as original)."""
        
        if not rule_is_heading and not ml_is_heading:
            combined_conf = (rule_conf * self.rule_weight + ml_conf * self.ml_weight)
            return None, False, combined_conf
        
        if rule_is_heading and ml_is_heading:
            if rule_level == ml_level:
                combined_conf = (rule_conf * self.rule_weight + ml_conf * self.ml_weight)
                return rule_level, True, combined_conf
            else:
                if rule_conf * self.rule_weight > ml_conf * self.ml_weight:
                    return rule_level, True, rule_conf * 0.8
                else:
                    return ml_level, True, ml_conf * 0.8
        
        if rule_is_heading and not ml_is_heading:
            if rule_conf > 0.4:  # Lower threshold for multilingual
                return rule_level, True, rule_conf * 0.7
            else:
                return None, False, ml_conf * 0.8
        
        if not rule_is_heading and ml_is_heading:
            if ml_conf > 0.4:  # Lower threshold for multilingual
                return ml_level, True, ml_conf * 0.7
            else:
                return None, False, rule_conf * 0.8
        
        return None, False, 0.5
    
    def _apply_post_processing(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply post-processing with multilingual considerations."""
        processed = predictions.copy()
        
        # Apply hierarchical consistency
        processed = self._ensure_hierarchical_consistency(processed)
        
        # Filter false positives (with multilingual awareness)
        processed = self._filter_false_positives_multilingual(processed)
        
        # Validate page number consistency
        processed = self._validate_page_consistency(processed)
        
        return processed
    
    def _filter_false_positives_multilingual(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter false positives with multilingual awareness."""
        processed = predictions.copy()
        
        for i, pred in enumerate(processed):
            if pred.get("final_is_heading", False):
                text = pred.get("text", "")
                script = pred.get("detected_script", "unknown")
                
                # More lenient filtering for non-Latin scripts
                min_length = 2 if script in ["devanagari", "chinese", "arabic"] else 3
                
                if len(text.strip()) < min_length:
                    processed[i]["final_is_heading"] = False
                    processed[i]["final_predicted_level"] = None
                    processed[i]["final_confidence"] *= 0.5
                    processed[i]["filtered_reason"] = "too_short"
                    continue
                
                # Check for meaningful content (script-aware)
                if script == "unknown" and not any(c.isalnum() for c in text):
                    processed[i]["final_is_heading"] = False
                    processed[i]["final_predicted_level"] = None
                    processed[i]["final_confidence"] *= 0.5
                    processed[i]["filtered_reason"] = "no_meaningful_content"
                    continue
                
                # More lenient length filtering for non-Latin scripts
                max_length = 300 if script in ["devanagari", "chinese", "arabic"] else 200
                
                if len(text) > max_length:
                    if processed[i].get("final_confidence", 0) < 0.6:
                        processed[i]["final_is_heading"] = False
                        processed[i]["final_predicted_level"] = None
                        processed[i]["final_confidence"] *= 0.6
                        processed[i]["filtered_reason"] = "too_long"
        
        return processed
    
    # Keep other methods from original hybrid classifier
    def _ensure_hierarchical_consistency(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure heading hierarchy makes sense."""
        processed = predictions.copy()
        last_heading_level = None
        
        for i, pred in enumerate(processed):
            if pred.get("final_is_heading", False):
                current_level = pred.get("final_predicted_level")
                
                if current_level and last_heading_level:
                    level_values = {"H1": 1, "H2": 2, "H3": 3}
                    current_value = level_values.get(current_level, 2)
                    last_value = level_values.get(last_heading_level, 2)
                    
                    if current_value > last_value + 1:
                        if pred.get("final_confidence", 0) < 0.8:
                            adjusted_level = f"H{last_value + 1}"
                            processed[i]["final_predicted_level"] = adjusted_level
                            processed[i]["final_confidence"] *= 0.8
                            processed[i]["hierarchy_adjusted"] = True
                
                last_heading_level = processed[i].get("final_predicted_level")
        
        return processed
    
    def _validate_page_consistency(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate page consistency."""
        processed = predictions.copy()
        
        page_heading_counts = {}
        for pred in processed:
            if pred.get("final_is_heading", False):
                page = pred.get("page", 1)
                page_heading_counts[page] = page_heading_counts.get(page, 0) + 1
        
        for page, count in page_heading_counts.items():
            if count > 15:  # More lenient for multilingual (was 10)
                for i, pred in enumerate(processed):
                    if (pred.get("page") == page and 
                        pred.get("final_is_heading", False) and
                        pred.get("final_confidence", 0) < 0.5):  # Lower threshold
                        processed[i]["final_confidence"] *= 0.8
                        processed[i]["page_excess_warning"] = True
        
        return processed 