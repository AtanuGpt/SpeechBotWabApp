import os
import io
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import azure.cognitiveservices.speech as speechsdk

# Initialize app
app = Flask(__name__)
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# FAISS setup
PERSIST_DIR = "./storage"
VECTOR_DIM = 1536
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

# Welcome message
WELCOME_MESSAGE = "Hello, I am your voice chat assistant. How can I help you today?"

# === ROUTES ===

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
        # Determine if running locally
        is_local = request.host.startswith("127.0.0.1") or "localhost" in request.host

        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = "en-IN-NeerjaNeural"
        speech_config.speech_synthesis_language = "en-US"

        if is_local:
            # Local: use default speaker
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            result = synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return jsonify({"status": "success", "message": "Played locally via speaker"})
            else:
                return jsonify({"status": "error", "message": "Speech synthesis failed"}), 500

        else:
            # Azure: return audio stream to frontend
            stream = speechsdk.audio.PullAudioOutputStream()
            audio_config = speechsdk.audio.AudioOutputConfig(stream=stream)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

            result = synthesizer.speak_ssml_async(ssml_text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_stream = speechsdk.AudioDataStream(result)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    audio_stream.save_to_wav_file(temp_wav.name)
                    temp_wav.seek(0)
                    audio_bytes = temp_wav.read()

                return send_file(
                    io.BytesIO(audio_bytes),
                    mimetype="audio/wav",
                    as_attachment=False,
                    download_name="speech.wav"
                )
            else:
                return jsonify({"status": "error", "message": "Speech synthesis failed"}), 500
            
    except Exception as e:
        print("❌ Speech error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

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

# === MAIN ===

if __name__ == "__main__":
    app.run(debug=False)
