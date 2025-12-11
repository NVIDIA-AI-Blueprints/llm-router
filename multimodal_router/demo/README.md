# Multimodal LLM Router Demo

A simple web application that showcases the functionality of a multimodal LLM router which directs conversation chats to different LLM endpoints based on intelligent routing decisions.

## Features

- 💬 **Chat Interface**: Clean, intuitive chat UI built with Gradio
- 🖼️ **Image Upload**: Support for multimodal conversations with image uploads
- 🤖 **Intelligent Routing**: Automatic model selection based on request analysis
- 📊 **Model Transparency**: See which model was selected for each request
- 💾 **Session History**: Message history maintained during your session

## Architecture

The demo follows a two-step process for each chat request:

1. **Step 1 - Routing**: Send the request to the router backend which analyzes the message and selects the optimal model
2. **Step 2 - Execution**: Send the request to the selected model and display the response

## Supported Models

The demo is configured to work with three models:

- **GPT-4o** (OpenAI API)
- **Nemotron Nano v2** (NVIDIA Build API)
- **Nemotron Nano VLM 12B** (NVIDIA Build API) - Multimodal vision-language model

## Prerequisites

1. **Router Service**: The router backend must be running on `http://localhost:8001`
   ```bash
   # From the multimodal_router directory
   ./scripts/run_local.sh
   ```

2. **API Keys**: You need valid API keys for:
   - OpenAI API
   - NVIDIA Build API

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env_template.txt .env
   # Edit .env and add your API keys
   ```

## Configuration

Edit the `.env` file with your configuration:

```env
# Router Configuration
ROUTER_ENDPOINT=http://localhost:8001/sfc_router/chat/completions

# OpenAI API Configuration
OPENAI_API_KEY=sk-...

# NVIDIA Build API Configuration
NVIDIA_API_KEY=nvapi-...
```

## Running the Demo

1. **Start the router service** (in a separate terminal):
   ```bash
   cd multimodal_router
   ./scripts/run_local.sh
   ```

2. **Launch the demo**:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open your browser and navigate to:
   ```
   http://localhost:7860
   ```

## Usage

1. **Text-only queries**: Simply type your message and click "Send"
2. **Multimodal queries**: Upload an image and optionally add text, then click "Send"
4. **Chat history**: Scroll through your conversation history
5. **Clear chat**: Click "Clear Chat" to start a new session

## Example Queries

- "Explain quantum computing in simple terms"
- "Carefully a Python function to calculate fibonacci numbers"
- "Describe what you see in this image" (with image upload)

## Troubleshooting

### Router Connection Issues
```
Router error: Connection refused
```
**Solution**: Ensure the router service is running on `http://localhost:8001`

### API Key Issues
```
API key not configured for [model]
```
**Solution**: Check that your `.env` file has the correct API keys

### Image Upload Issues
```
Error encoding image
```
**Solution**: Ensure the image is in a supported format (JPEG, PNG, etc.)

## Development

### Project Structure
```
demo/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env_template.txt    # Example environment configuration
```

### Customization

- **Change port**: Modify the `server_port` parameter in `app.py`
- **Add models**: Update the `MODELS` dictionary in `app.py`
- **Modify UI**: Adjust the Gradio components in the `create_demo()` function

