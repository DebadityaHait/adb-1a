import json
import re
from typing import List, Dict, Any, Tuple, Optional
from fuzzywuzzy import fuzz, process
import logging

class GroundTruthAligner:
    """Align ground truth headings with extracted text blocks."""
    
    def __init__(self, similarity_threshold: float = 80):
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)
    
    def load_ground_truth(self, json_path: str) -> List[Dict[str, Any]]:
        """Load ground truth headings from JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # Direct array format
                    return data
                elif isinstance(data, dict) and "outline" in data:
                    # Wrapped format with "outline" key
                    return data["outline"]
                else:
                    self.logger.error(f"Unknown JSON format in {json_path}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error loading ground truth from {json_path}: {str(e)}")
            return []
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\!\?\-]', '', text)
        return text.lower()
    
    def find_text_match(self, gt_text: str, text_blocks: List[Dict[str, Any]]) -> Tuple[Optional[int], float]:
        """
        Find the best matching text block for a ground truth heading.
        
        Args:
            gt_text: Ground truth heading text
            text_blocks: List of extracted text blocks
            
        Returns:
            Tuple of (block_index, similarity_score)
        """
        normalized_gt = self.normalize_text(gt_text)
        
        best_match_idx = None
        best_score = 0
        
        for i, block in enumerate(text_blocks):
            normalized_block = self.normalize_text(block["text"])
            
            # Try different fuzzy matching strategies
            scores = [
                fuzz.ratio(normalized_gt, normalized_block),
                fuzz.partial_ratio(normalized_gt, normalized_block),
                fuzz.token_sort_ratio(normalized_gt, normalized_block),
                fuzz.token_set_ratio(normalized_gt, normalized_block)
            ]
            
            max_score = max(scores)
            
            if max_score > best_score:
                best_score = max_score
                best_match_idx = i
        
        return best_match_idx, best_score
    
    def create_labeled_dataset(self, text_blocks: List[Dict[str, Any]], 
                             ground_truth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create labeled dataset by aligning ground truth with text blocks.
        
        Args:
            text_blocks: Extracted text blocks from PDF
            ground_truth: Ground truth headings
            
        Returns:
            List of text blocks with heading labels
        """
        # Initialize all blocks as non-headings
        labeled_blocks = []
        for block in text_blocks:
            labeled_block = block.copy()
            labeled_block["is_heading"] = False
            labeled_block["heading_level"] = None
            labeled_block["confidence"] = 0.0
            labeled_blocks.append(labeled_block)
        
        matched_blocks = set()
        
        # Match each ground truth heading with text blocks
        for gt_heading in ground_truth:
            gt_text = gt_heading["text"]
            gt_page = gt_heading["page"]
            gt_level = gt_heading["level"]
            
            # Filter blocks by page for better matching
            page_blocks = [(i, block) for i, block in enumerate(labeled_blocks) 
                          if block["page"] == gt_page and i not in matched_blocks]
            
            if not page_blocks:
                # Try adjacent pages if no blocks on exact page
                page_blocks = [(i, block) for i, block in enumerate(labeled_blocks) 
                              if abs(block["page"] - gt_page) <= 1 and i not in matched_blocks]
            
            if page_blocks:
                # Find best match among page blocks
                page_text_blocks = [block for _, block in page_blocks]
                match_idx, score = self.find_text_match(gt_text, page_text_blocks)
                
                if match_idx is not None and score >= self.similarity_threshold:
                    actual_idx = page_blocks[match_idx][0]
                    labeled_blocks[actual_idx]["is_heading"] = True
                    labeled_blocks[actual_idx]["heading_level"] = gt_level
                    labeled_blocks[actual_idx]["confidence"] = score / 100.0
                    matched_blocks.add(actual_idx)
                    
                    self.logger.debug(f"Matched '{gt_text}' with score {score:.1f}")
                else:
                    self.logger.warning(f"Could not match heading: '{gt_text}' (best score: {score:.1f})")
        
        self.logger.info(f"Matched {len(matched_blocks)} out of {len(ground_truth)} ground truth headings")
        return labeled_blocks
    
    def get_heading_distribution(self, labeled_blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of heading levels in the labeled dataset."""
        distribution = {"H1": 0, "H2": 0, "H3": 0, "non_heading": 0}
        
        for block in labeled_blocks:
            if block["is_heading"]:
                level = block["heading_level"]
                if level in distribution:
                    distribution[level] += 1
            else:
                distribution["non_heading"] += 1
        
        return distribution 