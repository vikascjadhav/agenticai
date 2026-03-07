# Simple LangChain Agent Examples

A collection of simple examples demonstrating LangChain agents with custom tools and a local model served by LM Studio.

## Project Overview

This project showcases a basic LangChain agent that integrates with a local Gemma model to provide weather information and generate lists of fruits and vegetables. The local model is served by LM Studio using an OpenAI-compatible endpoint.

## Features

- **Weather Tool**: Provides mock weather information for any location
- **Fruits and Vegetables Tool**: Generates lists of common fruits and vegetables
- **Local Gemma Integration**: Uses `google/gemma-3-4b` hosted locally through LM Studio's OpenAI-compatible API
- **Agent Framework**: Demonstrates LangChain's agent creation and invocation

## Prerequisites

- Python 3.12 or higher
- LM Studio running locally with:
  - OpenAI-compatible API enabled at `http://localhost:1234/v1`
  - Model loaded: `google/gemma-3-4b`
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

1. Start LM Studio and load the model `google/gemma-3-4b`
2. Ensure the LM Studio local server is enabled on `http://localhost:1234/v1`
3. Run the agent:
   ```bash
   python langchain-app.py
   ```

The agent will process the query "weather in Tokyo and Mumbai" and demonstrate tool usage.

## Dependencies

- `langchain>=1.2.2` - Core LangChain framework
- `langchain-core>=1.2.17` - Base abstractions
- `langchain-openai==1.1.10` - OpenAI-compatible API integration used to call LM Studio locally
- `python-dotenv==1.2.1` - Environment variable management
- `pydantic==2.12.5` - Data validation

## Project Structure

- `langchain-app.py` - Main application file
- `requirements.txt` - Python dependencies
- `script.sh` - Shell script (currently empty)

## Notes

- Model configuration is defined in `local_gemma_model()` in `langchain-app.py`:
  - `base_url="http://localhost:1234/v1"`
  - `model="google/gemma-3-4b"`
- This is a demonstration project showing basic agent functionality
- Weather data is mocked for demonstration purposes
- Ensure LM Studio local server is running before execution

## Contributing

Feel free to extend this project by adding more tools, improving the agent logic, or integrating with real APIs.
