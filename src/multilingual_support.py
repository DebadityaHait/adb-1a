import re
import unicodedata
from typing import List, Dict, Any, Set, Tuple
import logging

class MultilingualSupport:
    """Add multilingual support for heading detection across different languages and scripts."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Multilingual heading keywords
        self.heading_keywords = {
            'english': [
                "chapter", "section", "introduction", "conclusion", "abstract",
                "overview", "summary", "background", "methodology", "results",
                "discussion", "references", "appendix", "contents", "index"
            ],
            'hindi': [
                "अध्याय", "भाग", "प्रस्तावना", "निष्कर्ष", "सार", "परिचय",
                "विषय", "सूची", "संदर्भ", "अनुक्रमणिका", "विवरण", "सारांश"
            ],
            'spanish': [
                "capítulo", "sección", "introducción", "conclusión", "resumen",
                "antecedentes", "metodología", "resultados", "discusión", "referencias"
            ],
            'french': [
                "chapitre", "section", "introduction", "conclusion", "résumé",
                "contexte", "méthodologie", "résultats", "discussion", "références"
            ],
            'german': [
                "kapitel", "abschnitt", "einleitung", "fazit", "zusammenfassung",
                "hintergrund", "methodik", "ergebnisse", "diskussion", "literatur"
            ],
            'chinese': [
                "章", "节", "介绍", "结论", "摘要", "背景", "方法", "结果", "讨论", "参考文献"
            ],
            'arabic': [
                "فصل", "قسم", "مقدمة", "خاتمة", "ملخص", "خلفية", "منهجية", "نتائج", "مناقشة", "مراجع"
            ]
        }
        
        # Common punctuation across languages
        self.universal_punctuation = [':', '.', '!', '?', ';', '-', '—', '–', '(', ')', '[', ']']
        
        # Script detection patterns
        self.script_patterns = {
            'latin': re.compile(r'[a-zA-ZÀ-ÿ]'),
            'devanagari': re.compile(r'[\u0900-\u097F]'),
            'chinese': re.compile(r'[\u4e00-\u9fff]'),
            'arabic': re.compile(r'[\u0600-\u06FF]'),
            'cyrillic': re.compile(r'[\u0400-\u04FF]'),
            'greek': re.compile(r'[\u0370-\u03FF]')
        }
    
    def detect_script(self, text: str) -> str:
        """Detect the primary script used in the text."""
        if not text.strip():
            return 'unknown'
        
        script_counts = {}
        for script, pattern in self.script_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                script_counts[script] = matches
        
        if not script_counts:
            return 'unknown'
        
        return max(script_counts, key=script_counts.get)
    
    def is_heading_keyword_multilingual(self, text: str) -> float:
        """Check if text contains heading keywords in any supported language."""
        text_lower = text.lower().strip()
        max_score = 0.0
        
        for language, keywords in self.heading_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Give higher score for exact matches
                    if text_lower == keyword:
                        max_score = max(max_score, 0.8)
                    else:
                        max_score = max(max_score, 0.4)
        
        return max_score
    
    def extract_enhanced_content_features(self, text: str) -> Dict[str, Any]:
        """Extract language-agnostic content features."""
        features = {}
        
        if not text:
            return {f"enhanced_{k}": 0 for k in ["script_consistency", "char_diversity", 
                   "numeric_ratio", "punct_pattern", "keyword_score"]}
        
        # Script consistency
        script = self.detect_script(text)
        features["enhanced_script_type"] = script
        features["enhanced_script_consistency"] = 1.0 if script != 'unknown' else 0.0
        
        # Character diversity (works for all scripts)
        unique_chars = len(set(text))
        total_chars = len(text)
        features["enhanced_char_diversity"] = unique_chars / max(total_chars, 1)
        
        # Numeric content ratio
        numeric_chars = sum(1 for c in text if c.isdigit())
        features["enhanced_numeric_ratio"] = numeric_chars / max(total_chars, 1)
        
        # Universal punctuation patterns
        punct_count = sum(1 for c in text if c in self.universal_punctuation)
        features["enhanced_punct_pattern"] = punct_count / max(total_chars, 1)
        
        # Multilingual keyword scoring
        features["enhanced_keyword_score"] = self.is_heading_keyword_multilingual(text)
        
        return features
    
    def extract_enhanced_formatting_features(self, text: str, font_size: float, 
                                           font_flags: int, avg_font_size: float) -> Dict[str, Any]:
        """Extract enhanced formatting features that work across scripts."""
        features = {}
        
        # Enhanced font size analysis
        features["enhanced_font_prominence"] = font_size / max(avg_font_size, 1)
        features["enhanced_font_emphasis"] = 1.0 if font_size > avg_font_size * 1.1 else 0.0
        
        # Universal formatting detection
        features["enhanced_is_bold"] = bool(font_flags & 2**4)
        features["enhanced_is_italic"] = bool(font_flags & 2**1)
        
        # Script-aware case analysis
        script = self.detect_script(text)
        if script in ['latin', 'cyrillic', 'greek']:
            # Scripts with case distinctions
            features["enhanced_has_case_emphasis"] = text.isupper() or text.istitle()
            features["enhanced_case_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        else:
            # Scripts without case (Chinese, Hindi, Arabic, etc.)
            features["enhanced_has_case_emphasis"] = False
            features["enhanced_case_ratio"] = 0.0
        
        # Universal text length analysis
        features["enhanced_length_score"] = 1.0 if 5 <= len(text) <= 100 else 0.5 if len(text) <= 200 else 0.0
        
        return features
    
    def enhance_rule_based_scoring(self, block: Dict[str, Any], 
                                 base_score: float, base_level: str) -> Tuple[float, str]:
        """Enhance rule-based scoring with multilingual considerations."""
        enhanced_score = base_score
        enhanced_level = base_level
        
        text = block.get("text", "")
        
        # Boost score for multilingual keyword matches
        keyword_score = self.is_heading_keyword_multilingual(text)
        if keyword_score > 0:
            enhanced_score = min(enhanced_score + keyword_score, 1.0)
            if keyword_score >= 0.6:
                enhanced_level = "H1"
            elif keyword_score >= 0.3:
                enhanced_level = "H2"
        
        # Script-specific adjustments
        script = self.detect_script(text)
        
        if script == 'devanagari':
            # Hindi-specific patterns
            if any(char in text for char in ['॥', '।', '॰']):  # Devanagari punctuation
                enhanced_score += 0.1
            
        elif script == 'arabic':
            # Arabic-specific patterns
            if text.endswith('؟') or text.endswith('؛'):  # Arabic punctuation
                enhanced_score += 0.1
                
        elif script == 'chinese':
            # Chinese-specific patterns
            if any(char in text for char in ['。', '？', '！', '；', '：']):  # Chinese punctuation
                enhanced_score += 0.1
        
        # Universal numeric patterns (chapter numbers, etc.)
        if re.match(r'^[\d\u0660-\u0669\u06F0-\u06F9]+[\.\)]\s*', text):  # Including Arabic numerals
            enhanced_score += 0.2
            if not enhanced_level or enhanced_level == "non_heading":
                enhanced_level = "H2"
        
        return min(enhanced_score, 1.0), enhanced_level
    
    def get_language_stats(self, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about languages/scripts in the document."""
        script_counts = {}
        total_blocks = len(text_blocks)
        
        for block in text_blocks:
            text = block.get("text", "")
            script = self.detect_script(text)
            script_counts[script] = script_counts.get(script, 0) + 1
        
        # Determine primary script
        primary_script = max(script_counts, key=script_counts.get) if script_counts else 'unknown'
        
        return {
            "primary_script": primary_script,
            "script_distribution": script_counts,
            "is_multilingual": len([s for s in script_counts if script_counts[s] > total_blocks * 0.1]) > 1,
            "total_blocks": total_blocks
        } 