from typing import List, Dict, Any
import re
from .multilingual_support import MultilingualSupport

class EmergencyHeadingDetector:
    """Emergency heading detector for documents with very uniform formatting."""
    
    def __init__(self):
        self.multilingual = MultilingualSupport()
    
    def detect_emergency_headings(self, text_blocks: List[Dict[str, Any]], 
                                confidence_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect headings in documents where normal methods fail due to uniform formatting."""
        
        emergency_headings = []
        
        # Get document stats
        font_sizes = [block.get("font_size", 12) for block in text_blocks if block.get("font_size")]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        font_size_variance = self._calculate_variance(font_sizes) if font_sizes else 0
        
        # Check if this is a uniform formatting document
        is_uniform_formatting = font_size_variance < 2.0  # Very little font size variation
        
        if not is_uniform_formatting:
            return emergency_headings
        
        print(f"🚨 Emergency mode activated: Uniform formatting detected (variance: {font_size_variance:.2f})")
        
        for i, block in enumerate(text_blocks):
            text = block.get("text", "").strip()
            if not text or len(text) < 3:
                continue
            
            emergency_score = self._calculate_emergency_score(block, text_blocks, i)
            
            if emergency_score > confidence_threshold:
                # Determine heading level based on patterns
                level = self._determine_emergency_level(text, emergency_score)
                
                emergency_headings.append({
                    **block,
                    "emergency_heading": True,
                    "emergency_score": emergency_score,
                    "final_predicted_level": level,
                    "final_is_heading": True,
                    "final_confidence": min(emergency_score * 2, 0.8),  # Boost confidence
                    "decision_method": "emergency_detector"
                })
        
        print(f"🚨 Emergency detector found {len(emergency_headings)} potential headings")
        return emergency_headings
    
    def _calculate_emergency_score(self, block: Dict[str, Any], 
                                 all_blocks: List[Dict[str, Any]], index: int) -> float:
        """Calculate emergency heading score for uniform formatting documents."""
        text = block.get("text", "").strip()
        score = 0.0
        
        # 1. Multilingual keyword matching (high weight in emergency mode)
        keyword_score = self.multilingual.is_heading_keyword_multilingual(text)
        if keyword_score > 0:
            score += keyword_score * 0.8  # High weight
        
        # 2. Position-based scoring
        if index < 3:  # First few blocks
            score += 0.3
        elif index < len(all_blocks) * 0.1:  # First 10% of document
            score += 0.2
        
        # 3. Text patterns that suggest headings
        
        # Short descriptive phrases
        word_count = len(text.split())
        if 1 <= word_count <= 8:
            score += 0.3
        
        # Ends with colon
        if text.endswith(':'):
            score += 0.4
        
        # Contains numbers (section/chapter numbering)
        if re.search(r'\d+', text) and word_count <= 6:
            score += 0.3
        
        # Title case or all caps (for scripts that support it)
        script = self.multilingual.detect_script(text)
        if script in ['latin', 'cyrillic', 'greek']:
            if text.istitle() or text.isupper():
                score += 0.25
        
        # 4. Script-specific patterns
        if script == 'devanagari':
            # Hindi-specific patterns
            if any(char in text for char in ['॥', '।', '॰']):  # Devanagari punctuation
                score += 0.3
            
            # Common Hindi heading words
            hindi_patterns = ['नीति', 'विषय', 'अध्याय', 'भाग', 'परिचय', 'समस्या', 'समाधान', 'निष्कर्ष']
            if any(pattern in text for pattern in hindi_patterns):
                score += 0.4
        
        # 5. Standalone lines (likely headings)
        if '\n' not in text and len(text) <= 100:
            score += 0.2
        
        # 6. Whitespace context (more space before/after)
        if index > 0 and index < len(all_blocks) - 1:
            prev_text = all_blocks[index - 1].get("text", "")
            next_text = all_blocks[index + 1].get("text", "")
            
            # Check if this block is separated from others
            if (len(prev_text.strip()) == 0 or len(next_text.strip()) == 0):
                score += 0.15
        
        # 7. Avoid very long paragraphs
        if len(text) > 200:
            score *= 0.3
        
        # 8. Boost for certain patterns
        emergency_patterns = [
            r'^[A-Z][a-z\s]{5,50}$',  # Title case English
            r'^\d+[\.\)]\s*.{5,50}$',  # Numbered items
            r'^[IV]+[\.\s]',          # Roman numerals
            r'पाठ्यक्रम|कोर्स|विषय|अध्याय',  # Course/chapter in Hindi
        ]
        
        for pattern in emergency_patterns:
            if re.search(pattern, text):
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def _determine_emergency_level(self, text: str, score: float) -> str:
        """Determine heading level in emergency mode."""
        
        # High score items are likely H1
        if score > 0.6:
            return "H1"
        
        # Medium score items are likely H2  
        if score > 0.3:
            return "H2"
        
        # Low score items are H3
        return "H3"
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        variance = sum(squared_diffs) / len(values)
        return variance 