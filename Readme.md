# Voice Assistant v1

A Python-based Voice Assistant that processes spoken commands through a complete speech-to-action pipeline.
The system captures audio from a microphone, processes the audio, converts speech to text using Whisper, cleans the transcript, classifies the user's intent, executes the corresponding action, generates a response, and converts the response back to speech using Text-to-Speech.
This project was developed as a practical end-to-end implementation of an Audio → ASR → NLP → Intent Classification → Action → TTS pipeline.

## Features

- Microphone-based voice input
- Audio recording with PyAudio
- Audio normalization
- Silence removal
- Speech-to-Text using OpenAI Whisper
- Basic text preprocessing
- Zero-Shot Intent Classification
- Confidence thresholding
- Intent routing
- Intent-based actions
- Dynamic response generation
- Text-to-Speech using pyttsx3
- Application logging
- Error handling
- Command-line interface
- Session evaluation
- CSV evaluation report
- Result visualization

## Pipeline:
- Microphone → Audio Processing → Whisper ASR → Transcript → Text Cleaning → Intent Classification → Confidence Threshold → Recognized / Not Recognized → Intent Router → Action → Response Generation → Text-to-Speech → Next Command → `GOODBYE`

## Technologies, Frameworks & Libraries:
- Programming Language
  - Python
- Data Processing
  - pandas
  - NumPy
- Machine Learning / NLP
  - Hugging Face Transformers
  - Zero-Shot Classification
- Speech Recognition
  - OpenAI Whisper
- Audio Processing
  - PyAudio
  - wave
- Text-to-Speech
  - pyttsx3
- Visualization
  - Matplotlib
- Logging
  - Python `logging`

## Learning Outcomes:

- Python application design
- Audio recording
- Digital audio preprocessing
- Speech recognition
- Whisper ASR
- Text preprocessing
- NLP
- Intent classification
- Confidence thresholds
- Intent routing
- Function-based actions
- Response generation
- Text-to-Speech
- Error handling
- Logging
- CLI application design
- Dataset handling
- Session evaluation
- Data visualization

## Project Structure:

```text
Voice_Assistant_v1/
│
├── input/
│   ├── intent_dataset.csv
│   └── user_last_command.wav
│
├── output/
│   ├── assistant.log
│   ├── evaluation.csv
│   └── result.png
│
├── main.py
├── tested_scenario.txt
└── README.md
```

## Testing:

- The system was tested using 30 voice commands covering:
  - Correct intent predictions
  - Incorrect intent predictions
  - NOT_RECOGNIZED cases
  - Intent mismatches between Expected Intent and Predicted Intent
  - Different phrasings of the same intent
  - Goodbye handling
  - Invalid CLI input
- The complete test scenario used during the session is available in: `tested_scenario.txt`

## Evaluation:

- After the assistant terminates, a session evaluation is generated which includes:
  - Total number of commands
  - Recognized commands
  - Not recognized commands
  - Correct recognized commands
  - Incorrect recognized commands
  - Recognition rate
  - Recognition accuracy among recognized commands
- The evaluation results are saved to: `output/evaluation.csv`

## Visualization:

- The evaluation results are also visualized using Matplotlib and the generated plot contains:
  - Not Recognized commands
  - Recognized & Correct commands
  - Recognized & Incorrect commands
  - Recognition Rate
  - Accuracy
- The result plot is saved to: `output/result.png`

## Logging:

- The application uses Python's built-in logging module and each successful command records information such as:
  - Command number
  - Expected intent
  - Transcript
  - Cleaned text
  - Predicted intent
  - Confidence
  - Recognition status
  - Accuracy
  - Action
  - Response

## Error Handling:

- The system includes error handling for several stages of the pipeline:
  - Microphone Errors
  - Audio Processing Errors
  - ASR Errors
  - TTS Errors
  - Unknown Intent Errors

## Limitations:

- This version of the Voice Assistant uses a _Zero-Shot Classification_ model instead of a dedicated intent classification model trained specifically for this dataset.
- During testing, this resulted in some cases where:
  - An unrelated sentence was classified as an existing intent
  - A valid command received confidence below the threshold
  - Semantically similar commands received different confidence scores
- **These results indicate that the current classification approach can be improved.**

## Future Improvements:

- The next version of the project will focus on improving the intent classification system.
- Planned improvements include:
  - Replacing the current Zero-Shot Classification approach with a dedicated trainable intent classification model
  - Training and evaluating the new model using the existing dataset and intent set
  - Comparing the new model against the current Zero-Shot Classification approach using the same test scenarios
  - Evaluating model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
  - Improving the reliability of `NOT_RECOGNIZED` detection and reducing incorrect intent predictions
