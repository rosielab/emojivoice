import qi
import argparse
import sys
import time
import os
import paramiko
from scp import SCPClient


pepper_ip = "10.0.0.4"  # Replace with Pepper's IP address

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
    try:
        # Connect to Pepper
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(pepper_ip, username=username, password=password)

        # Create SCP client
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_file, remote_path)
            print(f"File '{local_file}' transferred to '{remote_path}'")

        ssh.close()

    except Exception as e:
        print(f"Failed to transfer file: {e}")


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

def play_audio_file(ip, file_path):
    audio_player_service = app.session.service("ALAudioPlayer")
    remote_path = "/home/nao/" + file_path
    transfer_file_with_scp(ip, file_path, remote_path)
    #time.sleep(1.0)

    audio_player_service.playFile(remote_path,1.0,-1.0)

    delete_remote_file(pepper_ip, remote_path)


# Replace the URL with the IP of Pepper, get the ip from pressing the power button once
app = qi.Application(sys.argv, url="tcps://10.0.0.4:9503")
logins = ("nao", "nao")
factory = AuthenticatorFactory(*logins)
app.session.setClientAuthenticatorFactory(factory)
app.start()

play_audio_file("10.0.0.4", "Matcha-TTS/Zach-affectionate-6.wav")
