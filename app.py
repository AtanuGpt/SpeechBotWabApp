import os
import io
import tempfile
import faiss
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import azure.cognitiveservices.speech as speechsdk

app = Flask(__name__)

# Load .env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# FAISS setup
PERSIST_DIR = "./storage"
VECTOR_DIM = 1536
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# Welcome message
WELCOME_MESSAGE = "Hello, I am your voice chat assistant. How can I help you today?"

# Azure Speech SDK config (used per-request)
def get_speech_synthesizer():
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
    speech_config.speech_synthesis_language = "en-US"

    # No speaker device (important for Azure Web App)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_output_device=False)
    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# Load answer using LlamaIndex + FAISS
def fetch_answer(question):
    try:
        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context=storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        print("❌ Error fetching answer:", e)
        return "Sorry, I couldn't find an answer."

@app.route("/")
def index():
    return render_template("index.html", welcome_message=WELCOME_MESSAGE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    if not question:
        return jsonify({"answer": "Please ask a question."})
    answer = fetch_answer(question)
    return jsonify({"answer": answer})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"status": "error", "message": "No text provided"}), 400

    ssml_text = f"""
    <speak version='1.0' xml:lang='en-US'>
        <voice name='en-IN-NeerjaNeural'>
            <prosody rate='-5%'>{text}</prosody>
        </voice>
    </speak>
    """

    try:
        # Check if running locally (host = 127.0.0.1 or flask dev server)
        is_local = request.host.startswith("127.0.0.1") or "localhost" in request.host

        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
        speech_config.speech_synthesis_language = "en-US"

        if is_local:
            # Use speaker for localhost (sound playback)
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            result = synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return jsonify({"status": "success", "message": "Played locally via speaker"})
            else:
                return jsonify({"status": "error", "message": "Speech synthesis failed"}), 500

        else:
            # Azure environment — return audio stream
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_output_device=False)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            result = synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                stream = speechsdk.AudioDataStream(result)

                # Save to temp file and read to memory
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    stream.save_to_wav_file(temp_wav.name)
                    temp_wav.seek(0)
                    audio_bytes = temp_wav.read()

                audio_stream = io.BytesIO(audio_bytes)
                audio_stream.seek(0)

                return send_file(
                    audio_stream,
                    mimetype="audio/wav",
                    as_attachment=False,
                    download_name="speech.wav"
                )
            else:
                return jsonify({"status": "error", "message": "Speech synthesis failed"}), 500

    except Exception as e:
        print("❌ Speech error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
