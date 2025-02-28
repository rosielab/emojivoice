import qi
import argparse
import sys
import time
import os
import paramiko
from pydub import AudioSegment
from scp import SCPClient

import pyaudio
import wave

demo = "storytelling" # Replace with the name of the demo you want to run
pepper_ip = "10.0.0.4"  # Replace with Pepper's IP address
script = "storytelling/fairytale_script.txt" # Replace with the name of the script file

storytelling_output_path = demo + "/outputs/"
storybuilding_output_path = demo + "/outputs/"

class Authenticator:

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def initialAuthData(self):
        return {'user': self.username, 'token': self.password}

class AuthenticatorFactory:

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def newAuthenticator(self):
        return Authenticator(self.username, self.password)
    

def transfer_file_with_scp(pepper_ip, local_file, remote_path, username='nao', password='nao'):
    # try to transfer the file to Pepper using SCP, if it fails, try again
    try:
        # Connect to Pepper
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(pepper_ip, username=username, password=password)
        
        with SCPClient(ssh.get_transport()) as scp:
            print(f"Transferring file '{local_file}' to '{remote_path}'...")
            scp.put(local_file, remote_path)
        ssh.close()

    except Exception as e:
        print(f"Failed to transfer file: {e}")
        print("Retrying...")
        time.sleep(0.25)
        transfer_file_with_scp(pepper_ip, local_file, remote_path, username, password)


def delete_remote_file(pepper_ip, remote_path, username='nao', password='nao'):
    try:
        # Connect to Pepper
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(pepper_ip, username=username, password=password)

        # Run the command to delete the file
        command = f"rm -f {remote_path}"
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # Check for errors
        err = stderr.read().decode()
        if err:
            print(f"Failed to delete file: {err}")
        else:
            print(f"File '{remote_path}' deleted successfully.")

        ssh.close()

    except Exception as e:
        print(f"Failed to delete file: {e}")

def format_audio_file(input_file): #pepper only supports 16 bit audio files
    audio = AudioSegment.from_file(input_file +".wav")
    audio = audio.set_sample_width(2)
    audio.export(input_file+"_16b.wav", format="wav")

def run_animation_on_pepper(animation_name = "top"):
    # connect to pepper
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(pepper_ip, username='nao', password='nao')

    command = f"qicli call ALAnimationPlayer.runTag '{animation_name}'"
    stdin, stdout, stderr = ssh.exec_command(command)
    ssh.close()

def play_audio_file(ip, file_path, file_name="output"):
    audio_player_service = app.session.service("ALAudioPlayer")
    remote_path = "/home/nao/" + file_name +"_16b.wav"
    format_audio_file(file_path)

    transfer_file_with_scp(ip, file_path+"_16b.wav", remote_path)
    print(remote_path)

    run_animation_on_pepper()

    audio_player_service.playFile(remote_path)

    delete_remote_file(pepper_ip, remote_path)

def wait_for_file_update(file_path, demo):
    # Wait for the file to be created
    while not os.path.exists(file_path):
        print(f"Waiting for file '{file_path}' to be created...")
        time.sleep(0.1)

    # Wait for the file to update
    if demo != "storytelling":
        initial_modification_time = os.path.getmtime(file_path)
        while os.path.getmtime(file_path) == initial_modification_time:
            print(f"Waiting for file '{file_path}' to update...")
            time.sleep(0.1)

    # Wait for the file to finish writing
    last_size = 0
    while os.path.getsize(file_path) == last_size:
        last_size = os.path.getsize(file_path)
        print(f"Waiting for file '{file_path}' to finish writing...")
        time.sleep(0.2)


# Replace the URL with the IP of Pepper, get the ip from pressing the power button once
app = qi.Application(sys.argv, url="tcps://" + pepper_ip + ":9503")
logins = ("nao", "nao")
factory = AuthenticatorFactory(*logins)
app.session.setClientAuthenticatorFactory(factory)
app.start()

p = pyaudio.PyAudio()

if demo == "storybuilding":
    #loop forever, exit with ctrl+c
    while True:
        wait_for_file_update(storybuilding_output_path + "output.wav", "storybuilding")
        print("Playing output.wav")
        play_audio_file(pepper_ip, storybuilding_output_path + "output")
    
elif demo == "storytelling":
    #check the script txt file to see how many lines there are, then play that many audio files
    num_lines = sum(1 for line in open(script))
    print(f"Number of lines in script.txt: {num_lines}")

    for i in range(num_lines): 
        wait_for_file_update(storytelling_output_path + "to_play-" + str(i) + ".wav", "storytelling")
        print(f"Playing file {i}")
        play_audio_file(pepper_ip, storytelling_output_path + 'to_play-' +  str(i), 'to_play-' +  str(i))

        #test code to play audio file on the computer

        #play_audio_file(pepper_ip, "storytelling/outputs/to_play-0.wav") 
        #play_audio_file(pepper_ip, "../test.wav")

        # wf = wave.open(storytelling_output_path + "to_play-" + str(i) + ".wav", 'rb')
        # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
        #                 channels=wf.getnchannels(),
        #                 rate=wf.getframerate(),
        #                 output=True)
        # data = wf.readframes(1024)
        # while data:
        #     stream.write(data)
        #     data = wf.readframes(1024)
        # stream.stop_stream()
        # stream.close()
