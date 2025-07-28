import re
import math
from typing import List, Dict, Any, Optional
import numpy as np
import logging

class FeatureExtractor:
    """Extract comprehensive features from text blocks for heading classification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, text_blocks: List[Dict[str, Any]], 
                        doc_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract comprehensive features for each text block.
        
        Args:
            text_blocks: List of text blocks with formatting information
            doc_stats: Document-wide statistics
            
        Returns:
            List of text blocks with extracted features
        """
        featured_blocks = []
        
        for i, block in enumerate(text_blocks):
            features = self._extract_block_features(block, text_blocks, i, doc_stats)
            
            # Combine original block data with features
            featured_block = {**block, **features}
            featured_blocks.append(featured_block)
        
        self.logger.info(f"Extracted features for {len(featured_blocks)} text blocks")
        return featured_blocks
    
    def _extract_block_features(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                               block_idx: int, doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all features for a single text block."""
        features = {}
        
        # Font features
        features.update(self._extract_font_features(block, doc_stats))
        
        # Position features  
        features.update(self._extract_position_features(block, all_blocks, block_idx))
        
        # Content features
        features.update(self._extract_content_features(block))
        
        # Context features
        features.update(self._extract_context_features(block, all_blocks, block_idx))
        
        # Page features
        features.update(self._extract_page_features(block, doc_stats))
        
        return features
    
    def _extract_font_features(self, block: Dict[str, Any], doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract font-related features."""
        features = {}
        
        # Relative font size (compared to document average)
        avg_size = doc_stats.get("avg_font_size", block["font_size"])
        features["font_size_ratio"] = block["font_size"] / avg_size if avg_size > 0 else 1.0
        features["font_size_normalized"] = (block["font_size"] - doc_stats.get("min_font_size", 0)) / \
                                         max(doc_stats.get("max_font_size", 1) - doc_stats.get("min_font_size", 0), 1)
        
        # Font formatting
        features["is_bold"] = self._is_bold(block["font_flags"])
        features["is_italic"] = self._is_italic(block["font_flags"])
        features["font_flags"] = block["font_flags"]
        
        # Font family analysis
        font_name = block["font_name"].lower()
        features["is_serif"] = any(serif in font_name for serif in ["times", "serif", "georgia"])
        features["is_sans_serif"] = any(sans in font_name for sans in ["arial", "helvetica", "sans"])
        features["is_monospace"] = any(mono in font_name for mono in ["courier", "mono", "consolas"])
        
        return features
    
    def _extract_position_features(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                                 block_idx: int) -> Dict[str, Any]:
        """Extract position-related features."""
        features = {}
        
        # Indentation (distance from left margin)
        features["x_position"] = block["x0"]
        features["indentation"] = block["x0"]
        
        # Vertical position features
        features["y_position"] = block["y0"]
        features["vertical_center"] = (block["y0"] + block["y1"]) / 2
        
        # Spacing analysis
        spacing_before, spacing_after = self._calculate_spacing(block, all_blocks, block_idx)
        features["spacing_before"] = spacing_before
        features["spacing_after"] = spacing_after
        features["spacing_ratio"] = spacing_before / max(spacing_after, 1) if spacing_after > 0 else spacing_before
        
        # Block dimensions
        features["block_width"] = block["width"]
        features["block_height"] = block["height"]
        features["aspect_ratio"] = block["width"] / max(block["height"], 1)
        
        return features
    
    def _extract_content_features(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content-related features."""
        features = {}
        text = block["text"]
        
        # Basic text statistics
        features["text_length"] = len(text)
        features["word_count"] = len(text.split())
        features["char_count"] = len(text.replace(' ', ''))
        features["avg_word_length"] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Capitalization patterns
        features["is_all_caps"] = text.isupper()
        features["is_title_case"] = text.istitle()
        features["has_initial_cap"] = text[0].isupper() if text else False
        features["caps_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Punctuation analysis
        features["ends_with_colon"] = text.endswith(':')
        features["ends_with_period"] = text.endswith('.')
        features["has_numbers"] = bool(re.search(r'\d', text))
        features["punctuation_count"] = len(re.findall(r'[^\w\s]', text))
        features["punctuation_ratio"] = features["punctuation_count"] / max(len(text), 1)
        
        # Special patterns
        features["is_numbered"] = bool(re.match(r'^\d+[\.\)]\s', text))
        features["is_bulleted"] = bool(re.match(r'^[\•\*\-\+]\s', text))
        features["has_whitespace"] = '\n' in text or '\t' in text
        
        return features
    
    def _extract_context_features(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                                block_idx: int) -> Dict[str, Any]:
        """Extract context-related features."""
        features = {}
        
        # Position in document
        features["relative_position"] = block_idx / max(len(all_blocks), 1)
        features["is_first_on_page"] = self._is_first_on_page(block, all_blocks, block_idx)
        features["is_last_on_page"] = self._is_last_on_page(block, all_blocks, block_idx)
        
        # Surrounding context
        prev_block = all_blocks[block_idx - 1] if block_idx > 0 else None
        next_block = all_blocks[block_idx + 1] if block_idx < len(all_blocks) - 1 else None
        
        if prev_block:
            features["prev_font_size_ratio"] = block["font_size"] / max(prev_block["font_size"], 1)
            features["prev_same_font"] = block["font_name"] == prev_block["font_name"]
        else:
            features["prev_font_size_ratio"] = 1.0
            features["prev_same_font"] = False
        
        if next_block:
            features["next_font_size_ratio"] = block["font_size"] / max(next_block["font_size"], 1)
            features["next_same_font"] = block["font_name"] == next_block["font_name"]
        else:
            features["next_font_size_ratio"] = 1.0
            features["next_same_font"] = False
        
        return features
    
    def _extract_page_features(self, block: Dict[str, Any], doc_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract page-related features."""
        features = {}
        
        # Page position
        features["page_number"] = block["page"]
        features["relative_page_position"] = block["page"] / max(doc_stats.get("total_pages", 1), 1)
        
        # Position on page (approximation based on y-coordinate)
        features["is_top_of_page"] = block["y0"] < 100  # Rough approximation
        features["is_middle_of_page"] = 100 <= block["y0"] <= 700
        features["is_bottom_of_page"] = block["y0"] > 700
        
        return features
    
    def _calculate_spacing(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                         block_idx: int) -> tuple:
        """Calculate spacing before and after the current block."""
        spacing_before = 0
        spacing_after = 0
        
        # Find spacing before
        for i in range(block_idx - 1, -1, -1):
            prev_block = all_blocks[i]
            if prev_block["page"] == block["page"]:
                spacing_before = block["y0"] - prev_block["y1"]
                break
        
        # Find spacing after  
        for i in range(block_idx + 1, len(all_blocks)):
            next_block = all_blocks[i]
            if next_block["page"] == block["page"]:
                spacing_after = next_block["y0"] - block["y1"]
                break
        
        return max(spacing_before, 0), max(spacing_after, 0)
    
    def _is_first_on_page(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                         block_idx: int) -> bool:
        """Check if block is first on its page."""
        page = block["page"]
        for i in range(block_idx):
            if all_blocks[i]["page"] == page:
                return False
        return True
    
    def _is_last_on_page(self, block: Dict[str, Any], all_blocks: List[Dict[str, Any]], 
                        block_idx: int) -> bool:
        """Check if block is last on its page."""
        page = block["page"]
        for i in range(block_idx + 1, len(all_blocks)):
            if all_blocks[i]["page"] == page:
                return False
        return True
    
    def _is_bold(self, font_flags: int) -> bool:
        """Check if text is bold based on font flags."""
        return bool(font_flags & 2**4)
    
    def _is_italic(self, font_flags: int) -> bool:
        """Check if text is italic based on font flags."""
        return bool(font_flags & 2**1)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names for ML model."""
        return [
            # Font features
            "font_size_ratio", "font_size_normalized", "is_bold", "is_italic",
            "is_serif", "is_sans_serif", "is_monospace",
            
            # Position features
            "indentation", "spacing_before", "spacing_after", "spacing_ratio",
            "block_width", "block_height", "aspect_ratio",
            
            # Content features
            "text_length", "word_count", "avg_word_length", "is_all_caps",
            "is_title_case", "has_initial_cap", "caps_ratio", "ends_with_colon",
            "ends_with_period", "has_numbers", "punctuation_ratio",
            "is_numbered", "is_bulleted",
            
            # Context features
            "relative_position", "is_first_on_page", "is_last_on_page",
            "prev_font_size_ratio", "prev_same_font", "next_font_size_ratio",
            "next_same_font",
            
            # Page features
            "relative_page_position", "is_top_of_page", "is_middle_of_page",
            "is_bottom_of_page"
        ] 