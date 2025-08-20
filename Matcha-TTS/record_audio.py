import os
from pathlib import Path

import numpy as np
import sounddevice as sd
import wavio
from pynput import keyboard

EMOJI_MAPPING = {
    'crying': '😭',
    'cool': '😎',
    'thinking': '🤔',
    'affectionate': '😍',
    'rofl': '🤣',
    'pleasant': '🙂',
    'surprise': '😮',
    'eye-roll': '🙄',
    'sweat': '😅',
    'angry': '😡',
    'grin': '😁'
}

print(os.getcwd())
SCRIPT_LOCATION_BASE_PATH = os.getcwd()

class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start_recording(self, filename, fs=44100, channels=1):
        self.frames = []
        self.recording = True
        stream = sd.InputStream(callback=self.callback, channels=channels, samplerate=fs)
        stream.start()
        print("Recording... Press any key but Enter to stop recording.")

        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

        stream.stop()
        stream.close()
        print("Recording stopped.")

        # Check if frames are collected
        if len(self.frames) > 0:
            # Convert frames to a NumPy array
            audio_data = np.concatenate(self.frames, axis=0)
            # Normalize audio data to fit within int16 range
            audio_data = np.clip(audio_data * 32767, -32768, 32767)
            audio_data = audio_data.astype(np.int16)  # Convert to int16

            wavio.write(filename, audio_data, fs, sampwidth=2)
            print(f"Recording saved to {filename}")
        else:
            print("No audio data recorded.")

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def on_press(self, key):
        print("Key pressed, stopping recording.")
        self.recording = False
        return False

def main():
    speaker_name = input("Enter the name of the speaker: ")
    emotion_to_record = ""

    while not emotion_to_record in EMOJI_MAPPING:
        emotion_to_record = input("Enter the name of recorded emotion (cool, thinking, affectionate, rofl, pleasant, surprise, eye-roll, sweat, crying, angry, grin):")  
    
    # Prompt for starting line number
    try :
        start_line = int(input("Enter the line number to start from (1-based index): ")) - 1
    except ValueError:
        start_line = 1

    emotion_script_location = SCRIPT_LOCATION_BASE_PATH + f"/script-{emotion_to_record}.txt"
    with open(emotion_script_location, 'r') as file:
        sentences = file.readlines()
        
    recording_dir = Path(f"recordings/{emotion_to_record}")
    recording_dir.mkdir(parents=True, exist_ok=True)
    print(f"'{recording_dir}' created successfully (or already exists)")

    # Ensure starting line is within range
    if start_line < 0 or start_line >= len(sentences):
        start_line = 1

    for i, line in enumerate(sentences[start_line:], start=start_line+1):
            new_filename = f"recordings/{emotion_to_record}/{speaker_name}-{emotion_to_record}-{i}.wav"
            input(f"Sentence {i}: {line} {EMOJI_MAPPING[emotion_to_record]}\nPress ENTER when you're ready to record.")

            recorder = Recorder()
            recorder.start_recording(new_filename)

if __name__ == "__main__":
    main()
