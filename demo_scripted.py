import pygame
import os

pygame.mixer.init()

# Folder containing your audio files
folder_path = "path_to_your_folder"

# Create a list of audio files (Assuming filenames are 1.wav, 2.wav, ..., 15.wav)
audio_files = [f"{i}.wav" for i in range(1, 11)]

def play_audio(file):
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass

for audio in audio_files:
    file_path = os.path.join(folder_path, audio)
    print(f"Press Enter to play: {audio}")
    input()  # Wait for Enter press
    play_audio(file_path)