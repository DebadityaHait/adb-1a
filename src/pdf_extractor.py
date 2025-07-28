import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Any, Tuple
import logging

class PDFExtractor:
    """Extract text and formatting information from PDFs using PyMuPDF."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text blocks with detailed formatting information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text blocks with formatting metadata
        """
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:  # Skip image blocks
                        continue
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Extract comprehensive formatting information
                            text_block = {
                                "text": span["text"].strip(),
                                "page": page_num + 1,
                                "font_name": span["font"],
                                "font_size": span["size"],
                                "font_flags": span["flags"],  # Bold, italic, etc.
                                "bbox": span["bbox"],  # [x0, y0, x1, y1]
                                "x0": span["bbox"][0],
                                "y0": span["bbox"][1],
                                "x1": span["bbox"][2],
                                "y1": span["bbox"][3],
                                "width": span["bbox"][2] - span["bbox"][0],
                                "height": span["bbox"][3] - span["bbox"][1],
                            }
                            
                            # Skip empty text blocks
                            if text_block["text"]:
                                text_blocks.append(text_block)
            
            doc.close()
            self.logger.info(f"Extracted {len(text_blocks)} text blocks from {pdf_path}")
            return text_blocks
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return []
    
    def get_document_stats(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate document-wide statistics for feature normalization."""
        if not text_blocks:
            return {}
        
        font_sizes = [block["font_size"] for block in text_blocks]
        
        stats = {
            "avg_font_size": sum(font_sizes) / len(font_sizes),
            "max_font_size": max(font_sizes),
            "min_font_size": min(font_sizes),
            "total_pages": max(block["page"] for block in text_blocks),
            "total_blocks": len(text_blocks)
        }
        
        return stats
    
    def is_bold(self, font_flags: int) -> bool:
        """Check if text is bold based on font flags."""
        return bool(font_flags & 2**4)  # Bold flag
    
    def is_italic(self, font_flags: int) -> bool:
        """Check if text is italic based on font flags."""
        return bool(font_flags & 2**1)  # Italic flag 