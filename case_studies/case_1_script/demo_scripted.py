import pygame
import os
import argparse

pygame.mixer.init()

parser = argparse.ArgumentParser(description="Set file path based on argument")
parser.add_argument('voice', choices=['base', 'default', 'emoji'], 
                    help="Choose a voice to set the file path")
args = parser.parse_args()

if args.voice == 'base':
    folder_path = './scripted_audio/base_matcha'
elif args.voice == 'default':
    folder_path = './scripted_audio/default_fine_tuned'
elif args.voice == 'emoji':
    folder_path = './scripted_audio/emoji_fine_tuned'

# Create a list of audio files (Assuming filenames are 1.wav, 2.wav, ..., 15.wav)
audio_files = [f"{i}.wav" for i in range(1, 12)]

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