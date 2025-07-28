from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np
from .multilingual_support import MultilingualSupport

class EnhancedRuleBasedClassifier:
    """Enhanced rule-based classifier with multilingual support."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.multilingual = MultilingualSupport()
        
        # Enhanced confidence weights
        self.confidence_weights = {
            "font_size": 0.25,
            "formatting": 0.25,
            "position": 0.20,
            "content": 0.15,
            "spacing": 0.10,
            "multilingual": 0.05  # New multilingual boost
        }
    
    def predict(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply enhanced rule-based predictions with multilingual support."""
        predictions = []
        
        # Get document language statistics
        lang_stats = self.multilingual.get_language_stats(text_blocks)
        self.logger.info(f"Document language stats: {lang_stats}")
        
        for block in text_blocks:
            # Add enhanced features
            enhanced_features = self._add_enhanced_features(block, lang_stats)
            block_enhanced = {**block, **enhanced_features}
            
            # Apply enhanced prediction
            prediction = self._predict_single_block_enhanced(block_enhanced, lang_stats)
            block_with_prediction = {**block_enhanced, **prediction}
            predictions.append(block_with_prediction)
        
        self.logger.info(f"Applied enhanced rule-based predictions to {len(predictions)} blocks")
        return predictions
    
    def _add_enhanced_features(self, block: Dict[str, Any], lang_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Add enhanced multilingual features to the block."""
        text = block.get("text", "")
        font_size = block.get("font_size", 12)
        font_flags = block.get("font_flags", 0)
        
        # Get document average font size
        doc_stats = {"avg_font_size": 12}  # Default, should be passed from pipeline
        
        # Enhanced content features
        content_features = self.multilingual.extract_enhanced_content_features(text)
        
        # Enhanced formatting features
        formatting_features = self.multilingual.extract_enhanced_formatting_features(
            text, font_size, font_flags, doc_stats["avg_font_size"]
        )
        
        return {**content_features, **formatting_features}
    
    def _predict_single_block_enhanced(self, block: Dict[str, Any], 
                                     lang_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhanced rules to a single text block."""
        # Apply original rule scores
        font_score, font_level = self._apply_font_rules(block)
        format_score, format_level = self._apply_enhanced_formatting_rules(block)
        position_score, position_level = self._apply_position_rules(block)
        content_score, content_level = self._apply_enhanced_content_rules(block)
        spacing_score, spacing_level = self._apply_spacing_rules(block)
        
        # New multilingual scoring
        multilingual_score, multilingual_level = self._apply_multilingual_rules(block, lang_stats)
        
        # Weighted confidence score
        total_confidence = (
            font_score * self.confidence_weights["font_size"] +
            format_score * self.confidence_weights["formatting"] +
            position_score * self.confidence_weights["position"] +
            content_score * self.confidence_weights["content"] +
            spacing_score * self.confidence_weights["spacing"] +
            multilingual_score * self.confidence_weights["multilingual"]
        )
        
        # Determine final prediction
        all_levels = [font_level, format_level, position_level, content_level, 
                     spacing_level, multilingual_level]
        level_votes = {"H1": 0, "H2": 0, "H3": 0, "non_heading": 0}
        
        for level in all_levels:
            if level:
                level_votes[level] += 1
        
        predicted_level = max(level_votes, key=level_votes.get)
        is_heading = predicted_level != "non_heading"
        
        # Enhanced final scoring with multilingual boost
        if multilingual_score > 0.5:  # Strong multilingual signal
            total_confidence = min(total_confidence * 1.2, 1.0)
        
        # Consensus strength
        consensus_strength = level_votes[predicted_level] / len(all_levels)
        final_confidence = total_confidence * consensus_strength
        
        return {
            "rule_predicted_level": predicted_level if is_heading else None,
            "rule_is_heading": is_heading,
            "rule_confidence": final_confidence,
            "enhanced_rule_scores": {
                "font": font_score,
                "formatting": format_score,
                "position": position_score,
                "content": content_score,
                "spacing": spacing_score,
                "multilingual": multilingual_score
            },
            "detected_script": self.multilingual.detect_script(block.get("text", ""))
        }
    
    def _apply_multilingual_rules(self, block: Dict[str, Any], 
                                lang_stats: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply multilingual-specific rules."""
        score = 0.0
        predicted_level = None
        
        text = block.get("text", "")
        
        # Keyword matching across languages
        keyword_score = block.get("enhanced_keyword_score", 0.0)
        if keyword_score > 0.6:
            score = 0.8
            predicted_level = "H1"
        elif keyword_score > 0.3:
            score = 0.6
            predicted_level = "H2"
        elif keyword_score > 0.0:
            score = 0.4
            predicted_level = "H3"
        
        # Script consistency bonus
        if block.get("enhanced_script_consistency", 0) > 0.8:
            score += 0.1
        
        # Length patterns that work across languages
        length_score = block.get("enhanced_length_score", 0.0)
        if length_score > 0.8:
            score += 0.2
        
        # Set prediction based on score
        if score >= 0.7:
            predicted_level = predicted_level or "H1"
        elif score >= 0.5:
            predicted_level = predicted_level or "H2"
        elif score >= 0.3:
            predicted_level = predicted_level or "H3"
        else:
            predicted_level = "non_heading"
        
        return min(score, 1.0), predicted_level
    
    def _apply_enhanced_formatting_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply enhanced formatting rules."""
        score = 0.0
        predicted_level = None
        
        # Use enhanced formatting features
        font_prominence = block.get("enhanced_font_prominence", 1.0)
        has_emphasis = block.get("enhanced_font_emphasis", False)
        is_bold = block.get("enhanced_is_bold", False)
        has_case_emphasis = block.get("enhanced_has_case_emphasis", False)
        
        # Font prominence
        if font_prominence >= 1.5:
            score += 0.6
            predicted_level = "H1"
        elif font_prominence >= 1.2:
            score += 0.4
            predicted_level = "H2"
        elif font_prominence >= 1.05:
            score += 0.2
            predicted_level = "H3"
        
        # Bold text
        if is_bold:
            score += 0.4
            if not predicted_level:
                predicted_level = "H2"
        
        # Case emphasis (for scripts that support it)
        if has_case_emphasis:
            score += 0.3
            if not predicted_level:
                predicted_level = "H2"
        
        # Font emphasis
        if has_emphasis:
            score += 0.2
        
        # Set final prediction
        if score >= 0.7:
            predicted_level = predicted_level or "H1"
        elif score >= 0.5:
            predicted_level = predicted_level or "H2"
        elif score >= 0.3:
            predicted_level = predicted_level or "H3"
        else:
            predicted_level = "non_heading"
        
        return min(score, 1.0), predicted_level
    
    def _apply_enhanced_content_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply enhanced content rules."""
        score = 0.0
        predicted_level = None
        
        # Use enhanced content features
        length_score = block.get("enhanced_length_score", 0.0)
        keyword_score = block.get("enhanced_keyword_score", 0.0)
        numeric_ratio = block.get("enhanced_numeric_ratio", 0.0)
        
        # Length scoring
        score += length_score * 0.4
        
        # Keyword scoring
        score += keyword_score * 0.5
        if keyword_score > 0.5:
            predicted_level = "H1"
        elif keyword_score > 0.2:
            predicted_level = "H2"
        
        # Numeric patterns (universal across languages)
        if numeric_ratio > 0.1 and numeric_ratio < 0.5:  # Some numbers but not all
            score += 0.2
            if not predicted_level:
                predicted_level = "H2"
        
        # Set final prediction
        if score >= 0.6:
            predicted_level = predicted_level or "H1"
        elif score >= 0.4:
            predicted_level = predicted_level or "H2"
        elif score >= 0.2:
            predicted_level = predicted_level or "H3"
        else:
            predicted_level = "non_heading"
        
        return max(min(score, 1.0), 0.0), predicted_level
    
    # Keep original methods for compatibility
    def _apply_font_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply font-based rules (unchanged from original)."""
        score = 0.0
        predicted_level = None
        
        font_ratio = block.get("font_size_ratio", 1.0)
        
        if font_ratio >= 1.5:
            score = 0.9
            predicted_level = "H1"
        elif font_ratio >= 1.3:
            score = 0.8
            predicted_level = "H1"
        elif font_ratio >= 1.15:
            score = 0.7
            predicted_level = "H2"
        elif font_ratio >= 1.05:
            score = 0.6
            predicted_level = "H3"
        else:
            score = 0.1
            predicted_level = "non_heading"
        
        return score, predicted_level
    
    def _apply_position_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply position-based rules (unchanged from original)."""
        score = 0.0
        predicted_level = None
        
        indentation = block.get("indentation", 0)
        is_first_on_page = block.get("is_first_on_page", False)
        is_top_of_page = block.get("is_top_of_page", False)
        
        if is_first_on_page:
            score += 0.4
            predicted_level = "H1"
        
        if is_top_of_page:
            score += 0.3
            if not predicted_level:
                predicted_level = "H1"
        
        if indentation < 50:
            score += 0.2
            if not predicted_level:
                predicted_level = "H1"
        
        if score >= 0.6:
            predicted_level = predicted_level or "H1"
        elif score >= 0.4:
            predicted_level = predicted_level or "H2"
        elif score >= 0.2:
            predicted_level = predicted_level or "H3"
        else:
            predicted_level = "non_heading"
        
        return min(score, 1.0), predicted_level
    
    def _apply_spacing_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply spacing-based rules (unchanged from original)."""
        score = 0.0
        predicted_level = None
        
        spacing_before = block.get("spacing_before", 0)
        spacing_ratio = block.get("spacing_ratio", 1.0)
        
        if spacing_ratio > 1.2:
            score += 0.3
        
        if spacing_before > 20:
            score += 0.2
            predicted_level = "H1"
        elif spacing_before > 10:
            score += 0.1
            if not predicted_level:
                predicted_level = "H2"
        
        if score >= 0.4:
            predicted_level = predicted_level or "H1"
        elif score >= 0.2:
            predicted_level = predicted_level or "H2"
        else:
            predicted_level = "non_heading"
        
        return min(score, 1.0), predicted_level 