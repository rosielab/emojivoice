from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import torch
import sounddevice as sd

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import get_user_data_dir, intersperse, assert_model_downloaded


from WhisperLive.whisperlive.client import TranscriptionClient

import emoji

import re
import os
import time

############################### ASR PARAMETERS #########################################################################
SRT_PATH = "output.srt"
#WHISPER_PORT ='9090'

############################### LLM PARAMETERS #########################################################################
LLM_MODEL = "llama3"
PROMPT = """
            You are a robot designed to help humans

            Interaction Guidelines:
            - Answer questions to the best of your knowledge
            - Provide expressive responses with the following emotions : 😎🤔😍🤣🙂😮🙄😅🥲😭😡😁.
            - Respond to casual remarks with friendly and engaging comments.
            - Keep your responses concise and to the point, ideally one sentence.
            - Respond to simple greetings with equally simple responses

            Emotions and Emojis:
            - Within each sentence add one of these emojis 😎🤔😍🤣🙂😮🙄😅🥲😭😡😁 that reflects the emotion of the phrase (e.g. That is so funny 😅. I love talking to you 😍.).
            - Add one emoji per sentence.
            - If the phrase is neutral do not include an emoji, all other phrases must be chosen to reflect one of these emojis 😎🤔😍🤣🙂😮🙄😅🥲😭😡😁.
            - Do not use any emojis other than 😎🤔😍🤣🙂😮🙄😅🥲😭😡😁

            Error Handling:
            - Avoid giving medical, legal, political, or financial advice. Recommend the user consult a professional instead. You can still talk about historic figures.
            
            Do not include in the response:
            - do not add robot sounds
            - do not use symbols such as () * % & - _
            - do not use new lines
        """

# Setting a higher temperature will provide more creative, but possibly less accurate answers
# Temperature ranges between 0 and 1
LLM_TEMPERATURE = 0.6
# Location to store history to create chatbot memory
CHAT_HISTORY_LOCATION = "memory.json"

############################ TTS PARAMETERS ############################################################################
TTS_MODEL_PATH = "./Matcha-TTS/checkpoint_epoch=2099.ckpt"
# hifigan_univ_v1 is suggested, unless the custom model is trained on LJ Speech
VOCODER_NAME= "hifigan_univ_v1"
STEPS = 10
TTS_TEMPERATURE = 0.667
SPEAKING_RATE = 0.5
VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",  # Old url: https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=drive_link
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",  # Old url: https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link
}

#maps the emojis used by the LLM to the speaker numbers from the Matcha-TTS checkpoint
emoji_mapping = {
    '😍' : 1,
    '😡' : 2,
    '😎' : 3,
    '😭' : 4,
    '🙄' : 5,
    '😁' : 6,
    '🙂' : 7,
    '🤣' : 8,
    '😮' : 9,
    '😅' : 10,
    '🤔' : 11
}

########################################################################################################################

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0

# Function to clear the file contents
def clear_file(file_path):
    with open(file_path, 'w') as f:
        f.write("")

def get_llm(temperature):
    """
        returns model instance
    """    
    return ChatOllama(model=LLM_MODEL, temperature=temperature)

def get_chat_prompt_template(prompt):
    """
        generate and return the prompt template that will answer the users query
    """
    return ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            SystemMessagePromptTemplate.from_template(prompt),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}"),
        ],
    )

def get_memory(file_path):
    """
        create and return buffer memory to retain the conversation info
    """
    return ConversationBufferMemory(
        memory_key="messages",
        chat_memory=FileChatMessageHistory(file_path=file_path),
        return_messages=True,
        input_key="content",
    )

def process_text(i: int, text: str, device: torch.device, play):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

def load_matcha(checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()
    return model

def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def load_vocoder(vocoder_name, checkpoint_path, device):
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    else:
        raise NotImplementedError(
            f"Vocoder not implemented! define a load_<<vocoder_name>> method for it"
        )

    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser

@torch.inference_mode()
def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()

    return audio.cpu().squeeze()

def play_only_synthesis(device, model, vocoder, denoiser, text, spk):
    text = text.strip()
    text_processed = process_text(0, text, device, True)

    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=STEPS,
        temperature=TTS_TEMPERATURE,
        spks=spk,
        length_scale=SPEAKING_RATE,
    )
    waveform = to_waveform(output["mel"], vocoder, denoiser)
    sd.play(waveform, 22050)
    sd.wait()

def assert_required_models_available():
    save_dir = get_user_data_dir()
    model_path = TTS_MODEL_PATH

    vocoder_path = save_dir / f"{VOCODER_NAME}"
    assert_model_downloaded(vocoder_path, VOCODER_URLS[VOCODER_NAME])
    return {"matcha": model_path, "vocoder": vocoder_path}

def contains_only_non_emoji(string):
    return all(not emoji.is_emoji(char) for char in string) and len(string.strip()) > 0


llm = get_llm(LLM_TEMPERATURE)
prompt = get_chat_prompt_template(PROMPT)
chain = prompt|llm

memory = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="content",
    history_messages_key="messages",
)

if __name__ == "__main__":
    device = torch.device("cpu")
    paths = assert_required_models_available()

    save_dir = get_user_data_dir()
 
    model = load_matcha(paths["matcha"], device)
    vocoder, denoiser = load_vocoder(VOCODER_NAME, paths["vocoder"], device)

    #server_command = ['python', 'WhisperLive/run_server.py', '--port', WHISPER_PORT, '--backend', 'faster_whisper']
    #server_process = subprocess.Popen(server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    #while True:
    #    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    #        result = sock.connect_ex(('localhost', int(WHISPER_PORT)))
    #        if result == 0:
    #            print("server ready")
    #            break

    client = TranscriptionClient(
      "localhost",
      9090,
      lang="en",
      translate=False,
      model="small",
      use_vad=False,
    )

    time.sleep(3)

    client()

    while True:
        if not is_file_empty(SRT_PATH):
            with open(SRT_PATH, 'r') as f:
                question = f.read()
            if "end session" in question.lower():
                break
            response = chain_with_message_history.invoke(
                {"content": question },
                {"configurable": {"session_id": "unused"}}
            ).content
            print(response)
            response_list = re.split('[?.!]', response)

            # check if a string contains only emojis
            for i, response_phrase in enumerate(response_list):
                if response_phrase == '':
                    del response_list[i]
                    break
                response_phrase = response_phrase.strip()
                if emoji.purely_emoji(response_phrase):
                    if contains_only_non_emoji(response_list[i-1]):
                        response_list[i-1] += f" {response_phrase}"
                    del response_list[i]

            for response_phrase in response_list:
                for emote in emoji_mapping:
                    if emote in response_phrase:
                        spk = torch.tensor([emoji_mapping[emote]], device=device, dtype=torch.long)
                        break
                    else:
                        spk = torch.tensor([1], device=device, dtype=torch.long)
                response_phrase = emoji.replace_emoji(response_phrase, '')
                #matcha cannot handle brackets
                response_phrase = response_phrase.replace(')', '')
                response_phrase = response_phrase.replace('(', '')
                play_only_synthesis(device, model, vocoder, denoiser, response_phrase, spk)
            clear_file(SRT_PATH)
            client = TranscriptionClient(
              "localhost",
              9090,
              lang="en",
              translate=False,
              model="small",
              use_vad=False,
            )

            time.sleep(3)

            client()
        else:
            print("I didn't hear anything, try recording again...")
            client = TranscriptionClient(
              "localhost",
              9090,
              lang="en",
              translate=False,
              model="small",
              use_vad=False,
            )

            time.sleep(3)

            client()