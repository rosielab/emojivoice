import os
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import wavio
import soundfile as sf
from pynput import keyboard
from pynput.keyboard import Key

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
        self.stream = None

    def start_recording(self, filename, fs=22050, channels=1):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(callback=self.callback, channels=channels, samplerate=fs)
        self.stream.start()
        print("Recording... Press ENTER to stop.")

        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

        self.stream.stop()
        self.stream.close()
        print("Recording stopped.")

        input("...")

        if len(self.frames) > 0:
            audio_data = np.concatenate(self.frames, axis=0)
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            wavio.write(filename, audio_data, fs, sampwidth=2)
            print(f"Recording saved to {filename}")
        else:
            print("No audio data recorded.")

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def on_press(self, key):
        """ Stop recording only when Enter is pressed """
        if key == Key.enter:
            print("Enter pressed, stopping recording.")
            self.recording = False
            return False  # Stop listener

    def play_audio(self, filename):
        """ Plays the recorded audio """
        if os.path.exists(filename):
            print("Playing recorded audio...")
            data, fs = sf.read(filename, dtype='int16')
            sd.play(data, fs)
            sd.wait()
        else:
            print("No recorded file found to play.")

def main():
    speaker_name = input("Enter the name of the speaker: ")
    emotion_to_record = ""

    while emotion_to_record not in EMOJI_MAPPING:
        emotion_to_record = input(
            "Enter the name of recorded emotion (cool, thinking, affectionate, rofl, pleasant, surprise, eye-roll, sweat, crying, angry, grin): ")

    try:
        start_line = int(input("Enter the line number to start from (1-based index): ")) - 1
    except ValueError:
        start_line = 1

    emotion_script_location = SCRIPT_LOCATION_BASE_PATH + f"/script-{emotion_to_record}.txt"
    with open(emotion_script_location, 'r') as file:
        sentences = file.readlines()

    recording_dir = Path(f"recordings/{emotion_to_record}")
    recording_dir.mkdir(parents=True, exist_ok=True)
    print(f"'{recording_dir}' created successfully (or already exists)")

    if start_line < 0 or start_line >= len(sentences):
        start_line = 1

    for i, line in enumerate(sentences[start_line:], start=start_line + 1):
        new_filename = f"recordings/{emotion_to_record}/{speaker_name}-{emotion_to_record}-{i}.wav"
        recorder = Recorder()

        while True:
            input(f"Sentence {i}: {line.strip()} {EMOJI_MAPPING[emotion_to_record]}\nPress ENTER when you're ready to record.")
            recorder.start_recording(new_filename)

            while True:
                user_input = input("Press 'p' to play, 'r' to re-record, or ENTER to confirm: ").strip().lower()

                if user_input == 'p':
                    recorder.play_audio(new_filename)
                elif user_input == 'r':
                    print("Re-recording...")
                    break
                elif user_input == '':
                    print("Recording confirmed.")
                    break

            if user_input == '':
                break


if __name__ == "__main__":
    main()
