from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import whisper
import pyaudio
import pyttsx3
import random
import string
import wave
import re
import logging
# ----------------------------------------------------------------------------------------------------------------------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
DATASET_PATH = INPUT_DIR / "intent_dataset.csv"
AUDIO_PATH = INPUT_DIR / "user_last_command.wav"
LOG_PATH = OUTPUT_DIR / "assistant.log"
EVALUATION_PATH = OUTPUT_DIR / "evaluation.csv"
RESULT_PATH = OUTPUT_DIR / "result.png"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
WHISPER_MODEL = "base"
CONFIDENCE_THRESHOLD = 0.80
FRAMES_PER_BUFFER = 3200
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
SILENCE_THRESHOLD = 500
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s | %(message)s")
LOGGER = logging.getLogger(__name__)
# ----------------------------------------------------------------------------------------------------------------------
# 1) LOAD INTENT DATASET:
def load_intent_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    required_columns = {"text", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    return df
# ----------------------------------------------------------------------------------------------------------------------
# 2) WHISPER ASR:
def load_whisper_model():
    return whisper.load_model(WHISPER_MODEL)
def transcribe(model, audio_path):
    try:
        result = model.transcribe(str(audio_path), fp16=False)
        text = result["text"].strip()
        if not text:
            raise ValueError("ASR produced an empty transcript!")
        return text
    except Exception as error:
        raise RuntimeError(f"ASR Error: {error}")
# ----------------------------------------------------------------------------------------------------------------------
# 3) TEXT CLEANING:
def lowercase_text(text):
    return text.lower()
def remove_punctuations(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans("", "", punctuations))
def remove_extra_whitespaces(text):
    return re.sub(r"\s+", " ", text).strip()
def clean_text(text):
    text = lowercase_text(text)
    text = remove_punctuations(text)
    text = remove_extra_whitespaces(text)
    return text
# ----------------------------------------------------------------------------------------------------------------------
# 4) AUDIO PROCESSING:
def normalize_audio(audio_data):
    audio = np.asarray(audio_data, dtype=np.float32)
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return audio.astype(np.int16)
    normalized = (audio / max_amplitude) * 32767
    return normalized.astype(np.int16)
def remove_silence(audio_data):
    audio = np.asarray(audio_data, dtype=np.int16)
    if len(audio) == 0:
        raise ValueError("Empty audio!")
    amplitude = np.abs(audio.astype(np.int32))
    active_indices = np.where(amplitude > SILENCE_THRESHOLD)[0]
    if len(active_indices) == 0:
        raise ValueError("Audio contains only silence!")
    start = active_indices[0]
    end = active_indices[-1] + 1
    return audio[start:end]
def process_audio_file(audio_path):
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            sample_width = audio.getsampwidth()
            frame_rate = audio.getframerate()
            channels = audio.getnchannels()
        audio_data = np.frombuffer(frames, dtype=np.int16)
        if len(audio_data) == 0:
            raise ValueError("Empty audio!")
        audio_data = normalize_audio(audio_data)
        audio_data = remove_silence(audio_data)
        with wave.open(str(audio_path), "wb") as processed_audio:
            processed_audio.setnchannels(channels)
            processed_audio.setsampwidth(sample_width)
            processed_audio.setframerate(frame_rate)
            processed_audio.writeframes(audio_data.tobytes())
    except Exception as error:
        raise RuntimeError(f"Audio Processing Error: {error}")
# ----------------------------------------------------------------------------------------------------------------------
# 5) INTENT CLASSIFICATION:
def load_classifier():
    return pipeline("zero-shot-classification", model=MODEL_NAME)
def classify_intent(classifier, dataset, text):
    candidate_labels = (dataset["intent"].unique().tolist())
    result = classifier(text, candidate_labels=candidate_labels)
    confidence = result["scores"][0]
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_intent = "NOT_RECOGNIZED"
    else:
        predicted_intent = result["labels"][0]
    return (predicted_intent, confidence, result)
# ----------------------------------------------------------------------------------------------------------------------
# 6) ACTIONS:
def greeting():
    return "Hello! How can I help you?"
def get_time():
    return datetime.now().strftime("%H:%M:%S")
def get_date():
    return datetime.now().strftime("%Y-%m-%d")
def tell_joke():
    jokes = ["Why do programmers prefer dark mode? Because light attracts bugs!",
             "Why do programmers hate nature? It has too many bugs!",
             "Why did the programmer quit his job? Because he didn't get arrays!",
             "Why was the Python programmer confused? Because he couldn't C!",
             "What do you call a programmer from Finland? Nerdic!"]
    return random.choice(jokes)
def goodbye():
    return "Goodbye!"
# ----------------------------------------------------------------------------------------------------------------------
# 7) INTENT ROUTER:
ACTION_MAP = {"GREETING": greeting, "GET_TIME": get_time, "GET_DATE": get_date, "JOKE": tell_joke, "GOODBYE": goodbye}
def execute_action(intent):
    action = ACTION_MAP.get(intent)
    if action is None:
        raise ValueError(f"Unknown Intent: {intent}")
    return action()
# ----------------------------------------------------------------------------------------------------------------------
# 8) RESPONSE GENERATION:
def generate_response(intent, action_result):
    if intent == "GREETING":
        return action_result
    if intent == "GET_TIME":
        return (f"The current time is {action_result}.")
    if intent == "GET_DATE":
        return (f"Today's date is {action_result}.")
    if intent == "JOKE":
        return action_result
    if intent == "GOODBYE":
        return action_result
    if intent == "NOT_RECOGNIZED":
        return ("I'm sorry, I didn't understand that. Please try again.")
    return ("I'm sorry, something went wrong!")
# ----------------------------------------------------------------------------------------------------------------------
# 9) TTS:
def speak_response(response):
    try:
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
        engine.stop()
    except Exception as error:
        raise RuntimeError(f"TTS Error: {error}")
# ----------------------------------------------------------------------------------------------------------------------
# 10) MICROPHONE:
def record_audio(audio_path):
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)
        print("Recording started.\n...")
        frames = []
        for _ in range(int(SAMPLE_RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames.append(data)
        print("Recording finished.")
        sample_width = p.get_sample_size(AUDIO_FORMAT)
        audio = wave.open(str(audio_path), "wb")
        audio.setnchannels(CHANNELS)
        audio.setsampwidth(sample_width)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(b"".join(frames))
        audio.close()
    except Exception as error:
        raise RuntimeError(f"Microphone Error: {error}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()
# ----------------------------------------------------------------------------------------------------------------------
# 11) PROCESS ONE COMMAND:
def process_command(command_number, expected_intent, dataset, whisper_model, classifier):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        record_audio(AUDIO_PATH)
    except Exception as error:
        print(f"ERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Microphone Error | {error}")
        return {"Command Number": command_number, "Timestamp": timestamp, "Expected Intent": expected_intent,
                "Transcript": "", "Cleaned Text": "", "Predicted Intent": "", "Confidence": 0,
                "Recognition": "Not Recognized", "Accuracy": "N/A", "Action": "Microphone Error",
                "Response": "Microphone error."}
    try:
        process_audio_file(AUDIO_PATH)
    except Exception as error:
        print(f"ERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Audio Processing Error | {error}")
        return {"Command Number": command_number, "Timestamp": timestamp, "Expected Intent": expected_intent,
                "Transcript": "", "Cleaned Text": "", "Predicted Intent": "", "Confidence": 0,
                "Recognition": "Not Recognized", "Accuracy": "N/A", "Action": "Audio Processing Error",
                "Response": "Audio processing error."}
    try:
        transcript = transcribe(whisper_model, AUDIO_PATH)
    except Exception as error:
        print(f"ERROR: {error}")
        LOGGER.error(f"Command #{command_number} | ASR Error | {error}")
        return {"Command Number": command_number, "Timestamp": timestamp, "Expected Intent": expected_intent,
                "Transcript": "", "Cleaned Text": "", "Predicted Intent": "", "Confidence": 0,
                "Recognition": "Not Recognized", "Accuracy": "N/A", "Action": "ASR Error",
                "Response": "Speech recognition error."}
    cleaned_text = clean_text(transcript)
    predicted_intent, confidence, _ = classify_intent(classifier, dataset, cleaned_text)
    if predicted_intent == "NOT_RECOGNIZED":
        recognition = "Not Recognized"
    else:
        recognition = "Recognized"
    if recognition == "Recognized":
        if predicted_intent == expected_intent:
            accuracy = "Correct"
        else:
            accuracy = "Incorrect"
    else:
        accuracy = "N/A"
    action_name = ""
    try:
        if predicted_intent == "NOT_RECOGNIZED":
            action_name = "Not Recognized"
            response = generate_response(predicted_intent, None)
        else:
            action_name = predicted_intent
            action_result = execute_action(predicted_intent)
            response = generate_response(predicted_intent, action_result)
    except Exception as error:
        print(f"ERROR: {error}")
        LOGGER.error(f"Command #{command_number} | Unknown Intent | {error}")
        action_name = "Unknown Intent"
        response = ("I'm sorry, something went wrong.")
    try:
        print("\nResponse:")
        print(response)
        speak_response(response)
    except Exception as error:
        print(f"TTS ERROR: {error}")
        LOGGER.error(f"Command #{command_number} | TTS Error | {error}")
    LOGGER.info(f"Command #{command_number} | "
                f"Expected Intent: {expected_intent} | "
                f"Transcript: {transcript} | "
                f"Cleaned Text: {cleaned_text} | "
                f"Predicted Intent: {predicted_intent} | "
                f"Confidence: {confidence:.4f} | "
                f"Recognition: {recognition} | "
                f"Accuracy: {accuracy} | "
                f"Action: {action_name} | "
                f"Response: {response}")
    return {"Command Number": command_number, "Timestamp": timestamp, "Expected Intent": expected_intent,
            "Transcript": transcript, "Cleaned Text": cleaned_text, "Predicted Intent": predicted_intent,
            "Confidence": confidence, "Recognition": recognition, "Accuracy": accuracy, "Action": action_name,
            "Response": response}
# ----------------------------------------------------------------------------------------------------------------------
# 12) SELECT EXPECTED INTENT:
def select_expected_intent(dataset):
    intents = (dataset["intent"].unique().tolist())
    print("------------------")
    print("Available Intents:")
    for index, intent in enumerate(intents,start=1):
        print(f"{index}. {intent}")
    while True:
        try:
            choice = int(input(f"\nSelect Expected Intent [1-{len(intents)}]: "))
            if 1 <= choice <= len(intents):
                return intents[choice - 1]
            print(f"Please choose a number between 1 and {len(intents)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
# ----------------------------------------------------------------------------------------------------------------------
# 13) CLI / CONVERSATION LOOP:
def run_assistant(dataset, whisper_model, classifier):
    session_results = []
    command_number = 1
    print("\nVOICE ASSISTANT PIPELINE")
    print("-------------------------------------------")
    print("For each command:")
    print("    1.Select the Expected Intent")
    print("    2.Press ENTER to start recording")
    print("    3.You may speak for 5 seconds")
    while True:
        print(f"\nCommand #{command_number}")
        expected_intent = select_expected_intent(dataset)
        print(f"Expected Intent: {expected_intent}")
        input("")
        result = process_command(command_number, expected_intent, dataset, whisper_model, classifier)
        session_results.append(result)
        print(f"Expected    : {result['Expected Intent']}")
        print(f"Transcript  : {result['Transcript']}")
        print(f"Cleaned     : {result['Cleaned Text']}")
        print(f"Intent      : {result['Predicted Intent']}")
        print(f"Confidence  : {result['Confidence']:.2%}")
        print(f"Recognition : {result['Recognition']}")
        print(f"Accuracy    : {result['Accuracy']}")
        print(f"Response    : {result['Response']}")
        if result["Predicted Intent"] == "GOODBYE":
            print("\nGoodbye! Voice Assistant stopped.")
            break
        if expected_intent == "GOODBYE" and result["Predicted Intent"] != "GOODBYE":
            print("\nWARNING: You selected GOODBYE as the Expected Intent,but the detected intent was not GOODBYE.")
            print("         Therefore, we will continue!")
        command_number += 1
    return session_results
# ----------------------------------------------------------------------------------------------------------------------
# 14) SESSION EVALUATION:
def evaluate_session(session_results):
    evaluation_df = pd.DataFrame(session_results)
    total_commands = len(evaluation_df)
    recognized = (evaluation_df["Recognition"] == "Recognized").sum()
    not_recognized = (evaluation_df["Recognition"] == "Not Recognized").sum()
    correct = (evaluation_df["Accuracy"] == "Correct").sum()
    incorrect = (evaluation_df["Accuracy"] == "Incorrect").sum()
    recognition_rate = (recognized / total_commands if total_commands > 0 else 0)
    accuracy = (correct / recognized if recognized > 0 else 0)
    evaluation_df.to_csv(EVALUATION_PATH, index=False)
    print("\nSESSION STATISTICS")
    print("-------------------------------------------")
    print(f"Total Commands       : {total_commands}")
    print(f"Recognized           : {recognized}")
    print(f"Not Recognized       : {not_recognized}")
    print(f"Recognized Correct   : {correct}")
    print(f"Recognized Incorrect : {incorrect}")
    print(f"Recognition Rate     : {recognition_rate:.2%}")
    print(f"Accuracy             : {accuracy:.2%}")
    print(f"Evaluation saved to  : {EVALUATION_PATH}")
    return (total_commands, not_recognized, correct, incorrect, recognition_rate, accuracy)
# ----------------------------------------------------------------------------------------------------------------------
# 15) RESULT VISUALIZATION:
def create_result_plot(total_commands, not_recognized, correct, incorrect, recognition_rate, accuracy):
    labels = ["Not Recognized", "Recognized\n(Correct)", "Recognized\n(Incorrect)"]
    values = [not_recognized, correct, incorrect]
    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, values)
    plt.title("Voice Assistant Session Results")
    plt.ylabel("Number of Commands")
    if total_commands > 0:
        plt.ylim(0, total_commands +max(1, total_commands * 0.15))
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    plt.text(0.02, 0.95,
             f"Total Commands : {total_commands}\n"
             f"Recognition Rate : {recognition_rate:.2%}\n"
             f"Accuracy : {accuracy:.2%}",
             transform=plt.gca().transAxes,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi=300)
    plt.close()
    print(f"Result plot saved to : {RESULT_PATH}")
# ----------------------------------------------------------------------------------------------------------------------
# 16) MAIN:
def main():
    print("VOICE ASSISTANT APPLICATION")
    print("-------------------------------------------")
    try:
        df = load_intent_dataset(DATASET_PATH)
        print("Loading Whisper model...")
        whisper_model = load_whisper_model()
        print("Loading Intent Classifier...")
        classifier = load_classifier()
        print("Models loaded successfully.")
        session_results = run_assistant(df, whisper_model, classifier)
        total_commands, not_recognized, correct, incorrect, recognition_rate, accuracy = evaluate_session(session_results)
        create_result_plot(total_commands, not_recognized, correct, incorrect, recognition_rate, accuracy)
    except Exception as error:
        print(f"\nFATAL ERROR: {error}")
        LOGGER.critical(f"Fatal Error | {error}")
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
