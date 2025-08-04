# Prompt Dissection & Perturbation Pipeline

This project consists of three core modules:

- **PromptAnatomy**: Prompt Dissection Pipeline (Semantic Component Classification)
- **ComPerturb**: Component Perturbation (Adversarial Attack Strategies)
- **PCM.py**: Prompt Complexity Metric

---

## Project Structure

```
.
├── PromptAnatomy/      # Prompt Dissection Pipeline
│   ├── process_sentence.py
│   └── new_auto_recognition.py
├── ComPerturb/         # Component Perturbation
│   ├── main.py
│   ├── strategies/
│   │   ├── COD.py      # Component Deletion
│   │   ├── SCI.py      # Special Character Insertion
│   │   ├── SER.py      # Sentence Rewriting
│   │   ├── SYR.py      # Synonym Replacement
│   │   └── WOD.py      # Word Deletion
│   ├── models/
│   └── utils/
├── PCM.py              # Prompt Complexity Metric
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## NLTK Data Download

Download required NLTK data before first run:

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

---

## Module Descriptions

### 1. PromptAnatomy (Prompt Dissection Pipeline)

**Purpose**: Automatically classify and dissect prompts into semantic components (Role, Directive, Additional Information, Output Formatting, Examples).

**Key Files**: 
- `PromptAnatomy/process_sentence.py`
- `PromptAnatomy/new_auto_recognition.py`

**Usage**:
```bash
python PromptAnatomy/process_sentence.py
```
```bash
python PromptAnatomy/new_auto_recognition.py
```

**Output**: Structured JSON file with classified prompt components for further perturbation.

### 2. ComPerturb (Component Perturbation)

**Purpose**: Apply various adversarial perturbation strategies to classified prompt components.

**Key Files**: 
- `ComPerturb/main.py`
- `ComPerturb/strategies/` (contains all perturbation strategies)

**Available Strategies**:
- **COD**: Component Deletion - Removes entire components
- **SCI**: Special Character Insertion - Inserts random characters as special characters
- **SER**: Sentence Rewriting - Rewrites sentences using synonyms
- **SYR**: Synonym Replacement - Replaces random words with synonyms
- **WOD**: Word Deletion - Deletes random words

**Perturbation Parameters**:
The perturbation functions use percentage-based calculations:
- **SYR (Synonym Replacement)**: Replaces 20% of words with synonyms
- **WOD (Word Deletion)**: Deletes 20% of words  
- **SCI (Special Character Insertion)**: Inserts 20% of characters as special characters

Parameters can be adjusted in `ComPerturb/utils/prompt_utils.py`:
```python
replacement_ratio = 0.2  # Replace 20% of words with synonyms
deletion_ratio = 0.2     # Delete 20% of words
insertion_ratio = 0.2    # Insert 20% of characters as special chars
```

**Usage**:
```bash
python ComPerturb/main.py --strategy COD --tag Role --input_file input.json --output_file output.json
```

**Parameters**:
- `--strategy`: Strategy abbreviation (COD/SCI/SER/SYR/WOD)
- `--tag`: Target component tag (Role/Directive/Additional Information/Output Formatting/Examples)
- `--input_file`: Input file path (.json or .xlsx)
- `--output_file`: Output file path (.json or .xlsx)

### 3. PCM.py (Prompt Complexity Measurement)

**Purpose**: Analyze prompt complexity across multiple dimensions including lexical, syntactic, semantic, and structural complexity.

**Key Features**:
- TF-IDF based lexical complexity
- Chunk tree depth for syntactic complexity
- Embedding-based semantic complexity
- Structural complexity analysis
- Comprehensive PCM score calculation

**Usage**:
```bash
python PCM.py
```

---

## Workflow

1. **Prompt Dissection**  
   Run `PromptAnatomy/process_sentence.py`, `PromptAnatomy/new_auto_recognition.py` to classify raw prompts into semantic components.

2. **Component Perturbation**  
   Run `ComPerturb/main.py` to apply perturbation strategies to classified components.

3. **Complexity Analysis**  
   Run `PCM.py` to analyze prompt complexity.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Run prompt dissection 
python PromptAnatomy/process_sentence.py
python PromptAnatomy/new_auto_recognition.py

# Run component perturbation (You can use datasets)
python ComPerturb/main.py --strategy COD --tag Role --input_file input.json --output_file output.json

# Run complexity analysis
python PCM.py
```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for detailed package versions

---

For detailed usage and parameter descriptions, please refer to the source code and comments in each module.
