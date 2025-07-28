# PDF Heading Extraction Model Training Instructions

## Project Context

You are building a lightweight machine learning model for an Adobe hackathon that extracts structured document outlines from PDFs. The system needs to:

- Process PDFs up to 50 pages within 10 seconds
- Extract hierarchical headings (H1, H2, H3) with page numbers
- Run offline on CPU-only architecture (AMD64, 8 CPUs, 16GB RAM)
- Model size limited to 200MB
- Output results in specific JSON format
- Be containerized with Docker


## Current Dataset Structure

You have 8 training PDFs with corresponding ground truth JSON files containing expected heading extractions, plus 4 test PDFs. The expected output format includes heading level, text content, and page numbers.

## Rule-Based + ML Hybrid Approach

This approach combines the speed and interpretability of rule-based systems with the accuracy of machine learning:

### Phase 1: Rule-Based Feature Extraction

- Extract visual and positional features from PDF text (font size, boldness, indentation, spacing)
- Apply heuristic rules to identify obvious heading patterns
- Create confidence scores for rule-based predictions


### Phase 2: ML Classification

- Train a lightweight classifier (Random Forest/XGBoost) on extracted features
- Use ML model for ambiguous cases where rules are uncertain
- Combine rule-based and ML predictions with weighted confidence scoring


### Phase 3: Post-Processing

- Apply document structure validation
- Ensure hierarchical consistency (H1 → H2 → H3)
- Filter false positives using context analysis


## Implementation Instructions

### Step 1: Environment Setup

Create a Python environment with the following dependencies:

- `pymupdf` (for fast PDF text extraction with positioning)
- `scikit-learn` (for lightweight ML models)
- `pandas` and `numpy` (for data processing)
- `joblib` (for model serialization and parallel processing)
- `json` (for handling input/output formats)


### Step 2: Data Processing Pipeline

#### 2.1 PDF Text Extraction

- Use PyMuPDF to extract text with detailed formatting information
- For each text block, capture: font name, font size, font flags (bold/italic), bounding box coordinates, page number
- Preserve original text content and whitespace patterns
- Store extraction results in a structured format for feature engineering


#### 2.2 Ground Truth Alignment

- Load expected output JSON files from `expected_outputs/` folder
- Match ground truth headings with extracted text blocks using fuzzy string matching
- Handle cases where ground truth text might have minor differences from PDF extraction
- Create labeled dataset with features and corresponding heading levels


#### 2.3 Feature Engineering

Extract these key features for each text block:

- **Font features**: Relative font size (compared to document average), font weight, font family
- **Position features**: Indentation level, vertical spacing before/after, horizontal alignment
- **Content features**: Text length, capitalization patterns, punctuation analysis
- **Context features**: Position in document, surrounding text characteristics
- **Page features**: Position on page (top/middle/bottom), page number


### Step 3: Rule-Based Component

#### 3.1 Heuristic Rules Development

Create rule sets based on common heading patterns:

- Font size thresholds (headings typically 1.2x+ larger than body text)
- Bold/italic formatting indicators
- Indentation patterns (H1 typically has 0 indentation, H2/H3 progressively indented)
- Whitespace patterns (headings usually have more spacing before them)
- Position-based rules (chapter titles often appear at page tops)


#### 3.2 Confidence Scoring

- Assign confidence scores (0-1) to rule-based predictions
- High confidence: Multiple rules agree strongly
- Medium confidence: Some rules agree, some ambiguous
- Low confidence: Rules conflict or are uncertain


### Step 4: Machine Learning Component

#### 4.1 Model Selection and Training

- Use Random Forest classifier for interpretability and speed
- Alternative: XGBoost for potentially higher accuracy
- Train on features where rule-based confidence is low or medium
- Use stratified sampling to handle class imbalance (more body text than headings)
- Implement cross-validation using the 8 training PDFs


#### 4.2 Model Optimization

- Feature selection using importance scores
- Hyperparameter tuning with focus on inference speed
- Model quantization to reduce size (target <50MB for the ML component)
- Test performance on single CPU core to ensure speed requirements


### Step 5: Hybrid Decision System

#### 5.1 Prediction Combination

- For high-confidence rule predictions: Use rule-based result directly
- For low-confidence cases: Use ML model prediction
- For medium-confidence cases: Weighted combination of both approaches
- Implement fallback hierarchy: Rules → ML → Conservative classification


#### 5.2 Post-Processing Validation

- Ensure heading hierarchy makes sense (no H3 directly after H1)
- Validate page number consistency
- Apply document structure templates if patterns are detected
- Filter obvious false positives (very short text, special characters only)


### Step 6: Training Process

#### 6.1 Data Preparation

- Process all 8 training PDFs through the extraction pipeline
- Create feature matrices aligned with ground truth labels
- Split data for training/validation (consider document-level splits)
- Generate class weights to handle imbalanced data


#### 6.2 Model Training Workflow

- Train rule-based component first and evaluate performance
- Identify cases where rules fail or have low confidence
- Train ML classifier on these challenging cases
- Validate hybrid system performance on held-out data
- Iterate on feature engineering and rule refinement


### Step 7: Inference Pipeline

#### 7.1 Processing New PDFs

- Extract text and features using the same pipeline as training
- Apply rule-based predictions with confidence scoring
- Use ML model for low-confidence predictions
- Combine results using the hybrid decision system
- Format output according to the specified JSON structure


#### 7.2 Batch Processing Setup

- Process all PDFs in `testing/` folder
- Generate output JSON files in `outputs/` folder
- Use parallel processing to leverage multiple CPU cores
- Implement memory-efficient processing for large documents
- Add progress tracking and error handling


### Step 8: Output Formatting

#### 8.1 JSON Structure Creation

- Convert model predictions to the required JSON format
- Ensure proper escaping of special characters in text
- Validate that all required fields are present (level, text, page)
- Sort results by page number and hierarchical order
- Handle edge cases (empty documents, no headings found)


#### 8.2 Quality Assurance

- Implement basic sanity checks on output
- Verify page numbers are within document range
- Check for reasonable heading distribution
- Log any potential issues or low-confidence predictions


### Step 9: Performance Optimization

#### 9.1 Speed Optimization

- Profile the entire pipeline to identify bottlenecks
- Optimize PDF processing (consider caching font analysis)
- Vectorize feature computation where possible
- Use efficient data structures for large documents
- Implement early stopping for obvious cases


#### 9.2 Memory Management

- Stream processing for large PDFs to avoid memory issues
- Clean up intermediate data structures
- Monitor memory usage during batch processing
- Implement garbage collection if needed


### Step 10: Model Evaluation and Iteration

#### 10.1 Performance Metrics

- Calculate precision, recall, and F1-score for each heading level
- Measure processing time per page and per document
- Evaluate memory usage and model size
- Test edge cases and document variations


#### 10.2 Model Refinement

- Analyze failure cases and improve rules/features
- Retrain ML component if accuracy is insufficient
- Fine-tune confidence thresholds for optimal performance
- Consider ensemble methods if single model isn't sufficient


## Success Criteria

Your implementation should achieve:

- High accuracy on heading detection (aim for >90% F1-score)
- Fast processing (well under 10 seconds for 50-page PDFs)
- Lightweight model (total system <200MB)
- Robust handling of various document formats and layouts
- Clean, properly formatted JSON output


## Final Deliverables

1. Trained hybrid model saved as serialized files
2. Complete inference pipeline script
3. Batch processing script for the testing folder
4. Output JSON files in the specified format
5. Performance metrics and validation results

Focus on creating a robust, fast system that can handle the diversity of document formats you might encounter in the hackathon evaluation.