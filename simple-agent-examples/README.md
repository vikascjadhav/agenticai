# Simple LangChain Agent Examples

A collection of simple examples demonstrating LangChain agents with custom tools and local language models.

## Project Overview

This project showcases a basic LangChain agent that integrates with a local Gemma model to provide weather information and generate lists of fruits and vegetables. The agent demonstrates how to create custom tools and use them within a conversational AI framework.

## Features

- **Weather Tool**: Provides mock weather information for any location
- **Fruits and Vegetables Tool**: Generates lists of common fruits and vegetables
- **Local Gemma Integration**: Uses a locally-hosted Gemma model via OpenAI-compatible API
- **Agent Framework**: Demonstrates LangChain's agent creation and invocation

## Prerequisites

- Python 3.8 or higher
- A local Gemma model running on `http://localhost:1234/v1` (e.g., using LM Studio, Ollama, or similar)
- API key for the local model (set in `.env` file)

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root
2. Add your API key:
   ```
   API_KEY=your_api_key_here
   ```

## Usage

1. Ensure your local Gemma model is running on `http://localhost:1234/v1`
2. Run the agent:
   ```bash
   python lanchain-app.py
   ```

The agent will process the query "weather in Tokyo and Mumbai" and demonstrate tool usage.

## Dependencies

- `langchain>=1.2.2` - Core LangChain framework
- `langchain-core>=1.2.17` - Base abstractions
- `langchain-openai==1.1.10` - OpenAI API integration (used for local Gemma)
- `python-dotenv==1.2.1` - Environment variable management
- `pydantic==2.12.5` - Data validation

## Project Structure

- `lanchain-app.py` - Main application file (note: filename contains typo, should be `langchain-app.py`)
- `requirements.txt` - Python dependencies
- `script.sh` - Shell script (currently empty)

## Notes

- The main file is named `lanchain-app.py` which appears to be a typo for `langchain-app.py`
- This is a demonstration project showing basic agent functionality
- Weather data is mocked for demonstration purposes
- Ensure your local model server is properly configured and running before execution

## Contributing

Feel free to extend this project by adding more tools, improving the agent logic, or integrating with real APIs.