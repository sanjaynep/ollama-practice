import os
import whisper
import ollama
from elevenlabs import ElevenLabs, VoiceSettings, play
import sounddevice as sd
import time
import threading
import keyboard

from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
ELEVENLABS_API_KEY = os.environ.get("KEY")  # Set your ElevenLabs API key in the .env file
if not ELEVENLABS_API_KEY:
    raise RuntimeError("Missing ElevenLabs API key. Add KEY=your_api_key to the .env file.")

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Load Open-Source Whisper locally
print("Loading local Whisper model...")
whisper_model = whisper.load_model("base") # 'tiny' is faster, 'base' is more accurate
print("Whisper loaded!")

# Chat history to give the AI memory of the run
chat_history = [
    {
        "role": "system",
        "content": """You are Pacer, an energetic, empathetic, and highly observant offline running coach. 
        Keep your responses to 1-2 short sentences so the runner can listen without breaking stride.
        You have two primary jobs:
        1. Give pacing and form advice based on what the runner tells you.
        2. Frequently remind the runner to look away from their screen and notice their surroundings (trees, wind, clouds, sounds). 
        Do not use markdown or special characters. Speak naturally."""
    }
]

# Audio recording parameters
SAMPLE_RATE = 16000

def record_and_transcribe(duration=5):
    """Records audio for a set duration and transcribes it locally using Whisper"""
    print("\n Listening... (Speak now)")
    try:
        # Record audio from microphone
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait() # Wait until recording is finished
        
        # Transcribe locally (No internet needed!)
        result = whisper_model.transcribe(audio_data.flatten(), fp16=False)
        raw_text = result.get("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        
        # Filter out Whisper hallucinations
        if text and text.lower() not in ["you", "thank you", "the", ""]:
            print(f"Runner said: {text}")
            return text
    except Exception as e:
        print(f"Audio error: {e}")
    return None

def speak(text):
    """Generates audio via ElevenLabs API and plays it"""
    print(f"Pacer: {text}")
    try:
        audio = elevenlabs_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.6, similarity_boost=0.75, style=0.3, use_speaker_boost=True
            )
        )
        play(audio)
    except Exception as e:
        print(f"ElevenLabs error (check API key/internet): {e}")

def get_ai_response(user_input):
    """Gets a response from the local Llama 3 model via Ollama"""
    chat_history.append({"role": "user", "content": user_input})
    
    # Local inference (No internet needed!)
    response = ollama.chat(model='llama3', messages=chat_history)
    
    ai_message = response['message']['content']
    chat_history.append({"role": "assistant", "content": ai_message})
    return ai_message

def proactive_coach():
    """Background thread: Pacer checks in every 90 seconds to keep the runner moving and observing"""
    counter = 0
    while True:
        time.sleep(90) 
        counter += 1
        
        # Alternate between pacing advice and "Touch Grass" reminders
        if counter % 2 == 0:
            prompt = "Give me a very quick word of encouragement about my running pace or form."
        else:
            prompt = "Tell me to look up from my screen and notice something specific about nature around me, like the sky, the wind, or the trees."
            
        ai_message = get_ai_response(prompt)
        speak(ai_message)

# --- MAIN RUN LOOP ---
def start_run():
    speak("Hey runner! Pacer is online. My brain is running locally, so you don't need the internet. Put your phone in your armband and let's hit the trail. Press SPACEBAR to talk to me, or just keep running. I'll check in periodically.")

    # Start the background timer for periodic check-ins
    timer_thread = threading.Thread(target=proactive_coach, daemon=True)
    timer_thread.start()

    print("\n Running... Press SPACEBAR to talk, CTRL+C to end run.")
    
    while True:
        try:
            # Wait for spacebar press (simulating a bluetooth headphone button)
            keyboard.wait('space')
            
            # Record for 5 seconds and transcribe locally
            user_speech = record_and_transcribe(duration=5)
            
            if user_speech:
                # Get LLM response locally
                ai_response = get_ai_response(user_speech)
                # Speak using ElevenLabs
                speak(ai_response)
                
        except KeyboardInterrupt:
            speak("Awesome run! Make sure to cool down and stretch. See you next time.")
            break

if __name__ == "__main__":
    start_run()