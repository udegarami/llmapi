from fastapi import FastAPI, UploadFile, HTTPException, Depends
import os
import logging
from tempfile import NamedTemporaryFile
import whisper
from gpt4all import GPT4All
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dependency for securely accessing OpenAI API key
def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key is not configured. Please set the 'OPENAI_API_KEY' environment variable.")
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured. Please set the 'OPENAI_API_KEY' environment variable.")
    
    logger.debug("OpenAI API key successfully retrieved.")
    return api_key

@app.post("/process-audio/")
async def process_audio(
    file: UploadFile, 
    model_choice: str = "OpenAI API",  # Default to OpenAI API for responses
    openai_api_key: str = Depends(get_openai_api_key)
):
    temp_audio_path = None
    try:
        # Save the uploaded file temporarily
        logger.debug(f"Received file: {file.filename}")
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        logger.debug(f"File saved temporarily at: {temp_audio_path}")

        # Load Whisper model for transcription
        logger.debug("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.debug("Whisper model loaded successfully.")

        # Transcribe audio
        logger.debug("Transcribing audio using Whisper...")
        transcription = whisper_model.transcribe(temp_audio_path)
        transcribed_text = transcription["text"]

        if not transcribed_text:
            logger.error("Transcription failed: No text generated.")
            raise HTTPException(status_code=500, detail="Transcription failed: No text generated.")

        logger.debug(f"Transcription successful: {transcribed_text}")

        # GPT Processing Section
        if transcribed_text:
            if model_choice == "GPT4All":
                # Load GPT4All model
                logger.debug("Loading GPT4All model...")
                gpt_model = GPT4All("ggml-gpt4all-j")  # Replace with your GPT4All model path
                gpt_response = gpt_model.chat(transcribed_text)
                logger.debug(f"GPT4All response: {gpt_response}")
                return {"transcription": transcribed_text, "chatgpt_response": gpt_response}

            elif model_choice == "OpenAI API":
                # Use OpenAI API for generating response
                logger.debug("Connecting to OpenAI API...")
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",  # Use the appropriate GPT model
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": transcribed_text},
                        ],
                        temperature=0,
                    )
                    openai_response = response.choices[0].message.content
                    logger.debug(f"OpenAI API response: {openai_response}")
                    return {"transcription": transcribed_text, "chatgpt_response": openai_response}

                except Exception as e:
                    logger.error(f"Error with OpenAI API: {e}")
                    raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

        # Clean up temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            logger.debug(f"Removing temporary audio file: {temp_audio_path}")
            os.remove(temp_audio_path)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

