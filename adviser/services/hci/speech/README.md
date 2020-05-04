# Purpose:
The speech folder holds various classes that are used in audio input and output handling. They are very strong baselines that can be used in any dialog systems, but like any other service, they can be exchanged easily.

# File Descriptions:
* `FeatureExtractor.py`: Holds feature extractors for both the backchannel service and the emotion recognition service
* `SpeechInputDecoder.py`: Holds a transformer based automatic speech recognition system
* `SpeechInputFeatureExtractor.py`: Holds a feature extractor for the automatic speech recognition system
* `SpeechOutputGenerator.py`: Holds a neural text to speech synthesis with a neural vocoder that produces sounds represented as arrays
* `SpeechOutputPlayer.py`: This service plays the audio array produced by the TTS synthesis. It is a separate service so the computations can be done on one device and the playback can happen on another one
* `SpeechRecorder.py`: This service provides a hotkey triggered audio recorder, with a naive turn taking system and a naive voice sanitizing system
* `cleaners.py`: The code in here provides general utility for the preprocessing of text to be synthesized as speech
* `speech_utility.py`: Provides simple utility to handle audio in files
