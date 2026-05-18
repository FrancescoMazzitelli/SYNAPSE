# SYNAPSE: SYNthetic Agentic Population Survey Engine

## Overview

This project generates synthetic survey responses using LLM-based agents. It creates realistic survey data by:

1. Sampling representative demographic profiles from existing datasets
2. Building AI agents that adopt these personas
3. Running survey questions through the agents to generate consistent responses

## Usage

```bash
cd synth-survey-gen
python main.py path/to/config/ [path/to/runfolder]
```

### Configuration

Configs are stored in `configs/Generic/`. The main configuration file is `config.json`:

- **model**: LLM settings (model name, context length, temperature)
- **synthesis**: Population sampling and agent generation settings
- **survey**: Question type handling (numeric, text, multiple choice)

### Data Format

The generic pipeline expects two files:

1. **Data file** (CSV/Excel): Contains respondent records with demographic variables
2. **Description file** (CSV/Excel/JSON): Maps column headers to descriptions and survey questions

## Architecture

- `main.py`: Entry point for the generic survey pipeline
- `synthesize.py`: Agent building and population sampling
- `preprocess.py`: Data mapping and question preparation
- `gp_survey_engine.py`: Core survey execution engine
- `postprocess.py`: Response serialization and output

## Requirements

```bash
pip install -r requirements.txt
```
