# Adobe India Hackathon: Connecting the Dots

This repository contains the solution for the Adobe India Hackathon.

## Round 1A: Understand Your Document

### Approach

Our solution for extracting a structured outline (Title, H1, H2, H3) from PDFs is based on a hybrid approach that combines rule-based heuristics with a lightweight machine learning model.

1.  **Text Extraction:** We use the `PyMuPDF` library to extract text blocks along with their metadata (font size, font name, position on page).
2.  **Feature Engineering:** For each text block, we extract features like relative font size, capitalization, line position, presence of numbering (e.g., "1.1", "A."), and boldness.
3.  **Classification:** A trained classifier (Random Forest) predicts the heading level (H1, H2, H3, or body text) based on these features. The Title is typically identified as the largest text on the first page.
4.  **Multilingual Support (Bonus):** We handle Japanese and other languages by using font properties and layout cues that are language-agnostic.

### Models and Libraries Used

-   **Libraries:** `PyMuPDF`, `scikit-learn`, `pandas`, `numpy`
-   **Models:** We use a `scikit-learn` Random Forest classifier (`ml_classifier.joblib`) which is included in the `/models` directory. The model size is under 200MB.

### How to Build and Run the Solution

The solution is containerized using Docker and is designed to run completely offline on a CPU.

1.  **Build the Docker Image:**
    ```bash
    docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
    ```

2.  **Run the Container:**
    Create `input` and `output` directories in your current host directory. Place your test PDFs in the `input` folder.

    ```bash
    # For Linux/macOS
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

    # For Windows (Command Prompt)
    docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output --network none mysolutionname:somerandomidentifier

    # For Windows (PowerShell)
    docker run --rm -v ${pwd}/input:/app/input -v ${pwd}/output:/app/output --network none mysolutionname:somerandomidentifier
    ```
    The container will automatically process all PDFs in `/app/input` and generate corresponding `.json` files in `/app/output`.

## Output Format

The system outputs JSON files with the following structure:

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

The title is automatically extracted from the largest heading among the first 4 headings in close proximity on the first page. 