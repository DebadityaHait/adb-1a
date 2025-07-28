from typing import List, Dict, Any, Tuple, Optional
import logging
import numpy as np

class RuleBasedClassifier:
    """Rule-based classifier for heading detection with confidence scoring."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.confidence_weights = {
            "font_size": 0.3,
            "formatting": 0.25,
            "position": 0.2,
            "content": 0.15,
            "spacing": 0.1
        }
    
    def predict(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply rule-based predictions to text blocks.
        
        Args:
            text_blocks: List of text blocks with extracted features
            
        Returns:
            List of text blocks with rule-based predictions and confidence scores
        """
        predictions = []
        
        for block in text_blocks:
            prediction = self._predict_single_block(block)
            block_with_prediction = {**block, **prediction}
            predictions.append(block_with_prediction)
        
        self.logger.info(f"Applied rule-based predictions to {len(predictions)} blocks")
        return predictions
    
    def _predict_single_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Apply rules to a single text block."""
        # Calculate individual rule scores
        font_score, font_level = self._apply_font_rules(block)
        format_score, format_level = self._apply_formatting_rules(block)
        position_score, position_level = self._apply_position_rules(block)
        content_score, content_level = self._apply_content_rules(block)
        spacing_score, spacing_level = self._apply_spacing_rules(block)
        
        # Weighted confidence score
        total_confidence = (
            font_score * self.confidence_weights["font_size"] +
            format_score * self.confidence_weights["formatting"] +
            position_score * self.confidence_weights["position"] +
            content_score * self.confidence_weights["content"] +
            spacing_score * self.confidence_weights["spacing"]
        )
        
        # Determine final prediction based on strongest signals
        all_levels = [font_level, format_level, position_level, content_level, spacing_level]
        level_votes = {"H1": 0, "H2": 0, "H3": 0, "non_heading": 0}
        
        for level in all_levels:
            if level:
                level_votes[level] += 1
        
        # Get most voted level
        predicted_level = max(level_votes, key=level_votes.get)
        is_heading = predicted_level != "non_heading"
        
        # Adjust confidence based on consensus
        consensus_strength = level_votes[predicted_level] / len(all_levels)
        final_confidence = total_confidence * consensus_strength
        
        return {
            "rule_predicted_level": predicted_level if is_heading else None,
            "rule_is_heading": is_heading,
            "rule_confidence": final_confidence,
            "rule_scores": {
                "font": font_score,
                "formatting": format_score,
                "position": position_score,
                "content": content_score,
                "spacing": spacing_score
            }
        }
    
    def _apply_font_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply font-based rules."""
        score = 0.0
        predicted_level = None
        
        font_ratio = block.get("font_size_ratio", 1.0)
        
        # Font size thresholds
        if font_ratio >= 1.5:  # Very large font
            score = 0.9
            predicted_level = "H1"
        elif font_ratio >= 1.3:  # Large font
            score = 0.8
            predicted_level = "H1"
        elif font_ratio >= 1.15:  # Medium-large font
            score = 0.7
            predicted_level = "H2"
        elif font_ratio >= 1.05:  # Slightly larger font
            score = 0.6
            predicted_level = "H3"
        elif font_ratio >= 0.95:  # Normal font
            score = 0.1
            predicted_level = "non_heading"
        else:  # Small font
            score = 0.0
            predicted_level = "non_heading"
        
        return score, predicted_level
    
    def _apply_formatting_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply formatting-based rules."""
        score = 0.0
        predicted_level = None
        
        is_bold = block.get("is_bold", False)
        is_italic = block.get("is_italic", False)
        is_all_caps = block.get("is_all_caps", False)
        is_title_case = block.get("is_title_case", False)
        
        # Bold text is often a heading
        if is_bold:
            score += 0.6
            predicted_level = "H2"  # Default for bold
        
        # All caps often indicates H1
        if is_all_caps and len(block.get("text", "")) > 3:
            score += 0.3
            if score >= 0.5:
                predicted_level = "H1"
        
        # Title case is common for headings
        if is_title_case:
            score += 0.2
            if not predicted_level:
                predicted_level = "H3"
        
        # Italic alone is less likely to be a heading
        if is_italic and not is_bold:
            score -= 0.1
        
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
    
    def _apply_position_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply position-based rules."""
        score = 0.0
        predicted_level = None
        
        indentation = block.get("indentation", 0)
        is_first_on_page = block.get("is_first_on_page", False)
        is_top_of_page = block.get("is_top_of_page", False)
        relative_position = block.get("relative_position", 0.5)
        
        # First item on page often a heading
        if is_first_on_page:
            score += 0.4
            predicted_level = "H1"
        
        # Top of page positioning
        if is_top_of_page:
            score += 0.3
            if not predicted_level:
                predicted_level = "H1"
        
        # Indentation patterns
        if indentation < 50:  # Left-aligned
            score += 0.2
            if not predicted_level:
                predicted_level = "H1"
        elif indentation < 100:  # Slightly indented
            score += 0.1
            if not predicted_level:
                predicted_level = "H2"
        else:  # More indented
            if not predicted_level:
                predicted_level = "H3"
        
        # Early in document
        if relative_position < 0.1:
            score += 0.1
        
        # Set final prediction
        if score >= 0.6:
            predicted_level = predicted_level or "H1"
        elif score >= 0.4:
            predicted_level = predicted_level or "H2"
        elif score >= 0.2:
            predicted_level = predicted_level or "H3"
        else:
            predicted_level = "non_heading"
        
        return min(score, 1.0), predicted_level
    
    def _apply_content_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply content-based rules."""
        score = 0.0
        predicted_level = None
        
        text = block.get("text", "")
        text_length = block.get("text_length", 0)
        word_count = block.get("word_count", 0)
        ends_with_colon = block.get("ends_with_colon", False)
        is_numbered = block.get("is_numbered", False)
        has_numbers = block.get("has_numbers", False)
        
        # Length heuristics
        if 5 <= text_length <= 100:  # Typical heading length
            score += 0.3
        elif text_length > 200:  # Too long for heading
            score -= 0.4
        
        # Word count
        if 1 <= word_count <= 8:  # Typical heading word count
            score += 0.2
        elif word_count > 15:  # Too many words
            score -= 0.3
        
        # Ends with colon (section headers)
        if ends_with_colon:
            score += 0.4
            predicted_level = "H2"
        
        # Numbered sections
        if is_numbered:
            score += 0.3
            predicted_level = "H2"
        
        # Contains numbers (chapters, sections)
        if has_numbers and word_count <= 5:
            score += 0.1
        
        # Common heading patterns
        text_lower = text.lower()
        heading_keywords = [
            "chapter", "section", "introduction", "conclusion", "abstract",
            "overview", "summary", "background", "methodology", "results",
            "discussion", "references", "appendix"
        ]
        
        if any(keyword in text_lower for keyword in heading_keywords):
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
    
    def _apply_spacing_rules(self, block: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Apply spacing-based rules."""
        score = 0.0
        predicted_level = None
        
        spacing_before = block.get("spacing_before", 0)
        spacing_after = block.get("spacing_after", 0)
        spacing_ratio = block.get("spacing_ratio", 1.0)
        
        # More spacing before than after (typical for headings)
        if spacing_ratio > 1.2:
            score += 0.3
        
        # Significant spacing before
        if spacing_before > 20:
            score += 0.2
            predicted_level = "H1"
        elif spacing_before > 10:
            score += 0.1
            if not predicted_level:
                predicted_level = "H2"
        
        # Some spacing after
        if spacing_after > 5:
            score += 0.1
        
        # Set final prediction
        if score >= 0.4:
            predicted_level = predicted_level or "H1"
        elif score >= 0.2:
            predicted_level = predicted_level or "H2"
        elif score >= 0.1:
            predicted_level = predicted_level or "H3"
        else:
            predicted_level = "non_heading"
        
        return min(score, 1.0), predicted_level
    
    def get_high_confidence_predictions(self, predictions: List[Dict[str, Any]], 
                                      threshold: float = 0.7) -> List[int]:
        """Get indices of high-confidence rule-based predictions."""
        high_confidence_indices = []
        
        for i, pred in enumerate(predictions):
            if pred.get("rule_confidence", 0) >= threshold:
                high_confidence_indices.append(i)
        
        return high_confidence_indices
    
    def get_low_confidence_predictions(self, predictions: List[Dict[str, Any]], 
                                     threshold: float = 0.3) -> List[int]:
        """Get indices of low-confidence predictions that need ML classification."""
        low_confidence_indices = []
        
        for i, pred in enumerate(predictions):
            if pred.get("rule_confidence", 0) <= threshold:
                low_confidence_indices.append(i)
        
        return low_confidence_indices 