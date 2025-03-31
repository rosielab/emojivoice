import threading
import numpy as np
import wavio
import sounddevice as sd
import sys
import tty
import termios

class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False
        self.stream = None

    def start_recording(self, filename, fs=44100, channels=1):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(callback=self.callback, channels=channels, samplerate=fs)
        self.stream.start()
        print("Recording... Press any key but Enter to stop recording.")

        # Start a thread to wait for a key press
        stop_thread = threading.Thread(target=self.wait_for_stop)
        stop_thread.start()

        # Wait for the recording to stop
        stop_thread.join()

        self.stream.stop()
        self.stream.close()
        print("Recording stopped.")

        # Check if frames are collected
        if len(self.frames) > 0:
            # Convert frames to a NumPy array
            audio_data = np.concatenate(self.frames, axis=0)
            # Normalize audio data to fit within int16 range
            audio_data = np.clip(audio_data * 32767, -32768, 32767)
            audio_data = audio_data.astype(np.int16)  # Convert to int16

            wavio.write(filename, audio_data, fs, sampwidth=2)
        else:
            print("No audio data recorded.")

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def wait_for_stop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        self.recording = False

if __name__ == "__main__":
    import time
    import whisper
    import torch
    from matcha.utils.utils import get_user_data_dir, assert_model_downloaded, intersperse
    from matcha.models.matcha_tts import MatchaTTS
    from matcha.hifigan.models import Generator as HiFiGAN
    from matcha.hifigan.denoiser import Denoiser
    from matcha.hifigan.env import AttrDict
    from matcha.hifigan.config import v1
    from matcha.text import sequence_to_text, text_to_sequence
    from langchain_ollama import ChatOllama
    from langchain.prompts import (
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
    )
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    import emoji

    import numpy as np
    from pathlib import Path
    import soundfile as sf
    import sounddevice as sd

    VOICE = 'emoji'
    WAV_PATH = "./outputs"
    TTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SRT_PATH = "output.srt"
    ASR_MODEL = "small.en"
    LLM_MODEL = "llama3.2:1b"
    PROMPT = """
                You are a robot designed to help humans

                Interaction Guidelines:
                - You are a robot who is playing a build a story game with a human. You will go back and forth each saying one sentence to build a story.
                - Provide expressive responses with only the following emotions : 😎🤔😍🤣🙂😮🙄😅😭😡😁.
                - Use short and simple responses to build the story.
                - Answers should be limited to one sentence.

                Emotions and Emojis:
                - At the end of each response add one of these emojis: 😎🤔😍🤣🙂😮🙄😅😭😡😁 that reflects the emotion of the the entire response.
                - Add only one emoji per response, at the end of the response.
                - If the phrase is neutral do not include an emoji
                - all other phrases must be chosen to reflect one of these emojis: 😎🤔😍🤣🙂😮🙄😅😭😡😁.
                - Do not use any emojis other than these: 😎🤔😍🤣🙂😮🙄😅😭😡😁

                Error Handling:
                - Avoid giving medical, legal, political, or financial advice. Recommend the user consult a professional instead. You can still talk about historic figures.
                
                Do not include in the response:
                - do not add robot sounds
                - do not use symbols such as () * % & - _
                - do not use new lines
                - do not add emojis other than: 😎🤔😍🤣🙂😮🙄😅😭😡😁
            """
    LLM_TEMPERATURE = 0.6
    if VOICE == 'base' :
        TTS_MODEL_PATH = "../Matcha-TTS/matcha_vctk.ckpt"
        SPEAKING_RATE = 0.8
        STEPS = 10
    else:
        TTS_MODEL_PATH = "../../Matcha-TTS/emoji-hri-paige.ckpt"
        SPEAKING_RATE = 0.8
        STEPS = 10
    VOCODER_NAME= "hifigan_univ_v1"
    TTS_TEMPERATURE = 0.667
    VOCODER_URLS = {
        "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
        "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
    }
    emoji_mapping = {
        '😍' : 107,
        '😡' : 58,
        '😎' : 79,
        '😭' : 103,
        '🙄' : 66,
        '😁' : 18,
        '🙂' : 12,
        '🤣' : 15,
        '😮' : 54,
        '😅' : 22,
        '🤔' : 17
    }
    #male voice mapping
    #emoji_mapping = {
    #    '😍' : 4,
    #    '😡' : 5,
    #    '😎' : 6,
    #    '😭' : 13,
    #    '🙄' : 16,
    #    '😁' : 26,
    #    '🙂' : 30,
    #    '🤣' : 38,
    #    '😮' : 60,
    #    '😅' : 82,
    #    '🤔' : 97
    #}

    def get_llm(temperature):
        return ChatOllama(model=LLM_MODEL, temperature=temperature)

    def get_chat_prompt_template(prompt):
        return ChatPromptTemplate(
            input_variables=["content", "messages"],
            messages=[
                SystemMessagePromptTemplate.from_template(prompt),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessagePromptTemplate.from_template("{content}"),
            ],
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

    def save_to_folder(filename: str, output: dict, folder: str):
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        filename = filename.replace(" ", "-")
        # sf.write(folder / f"to_play-{filename}.wav", output["waveform"], 22050, "PCM_24")
        sf.write(folder / f"output.wav", output["waveform"], 22050, "PCM_24")

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
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

        output["waveform"] = np.clip(output["waveform"], -1.0, 1.0)

        save_to_folder(text, output, WAV_PATH)

    def assert_required_models_available():
        save_dir = get_user_data_dir()
        model_path = TTS_MODEL_PATH

        vocoder_path = save_dir / f"{VOCODER_NAME}"
        assert_model_downloaded(vocoder_path, VOCODER_URLS[VOCODER_NAME])
        return {"matcha": model_path, "vocoder": vocoder_path}

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

        asr_model = whisper.load_model(ASR_MODEL)

        tts_device = torch.device(TTS_DEVICE)
        paths = assert_required_models_available()

        save_dir = get_user_data_dir()
    
        tts_model = load_matcha(paths["matcha"], tts_device)
        vocoder, denoiser = load_vocoder(VOCODER_NAME, paths["vocoder"], tts_device)

        input(f"Press Enter when you're ready to record 🎙️ ")

        recorder = Recorder()
        recorder.start_recording("output.wav")

        result = asr_model.transcribe("output.wav")
        result = result['text']

        print(f'speaker said: {result}')
        
        while True:
            if result != '':
                if "end session" in result.lower():
                    exit(0)
                print("LLM reading")
                response = chain_with_message_history.invoke(
                    {"content": result },
                    {"configurable": {"session_id": "unused"}}
                ).content
                print(response)
                # Get the last emoji (there should only be one but the LLM does not always behave)
                emoji_list = []
                for char in response:
                    if emoji.is_emoji(char):
                        emoji_list.append(char)
                # incase the last emoji is not in the emoji list
                if VOICE == 'base':
                    spk = torch.tensor([1], device=tts_device, dtype=torch.long)
                if VOICE == 'default':
                    spk = torch.tensor([7], device=tts_device, dtype=torch.long)
                if VOICE == 'emoji':
                    spk = torch.tensor([7], device=tts_device, dtype=torch.long)
                    for emote in emoji_list:
                        if emote in emoji_mapping:
                            spk = torch.tensor([emoji_mapping[emote]], device=tts_device, dtype=torch.long)
                            break
                response = emoji.replace_emoji(response, '')
                #matcha cannot handle brackets
                response = response.replace(')', '')
                response = response.replace('(', '')
                if response != '':
                    play_only_synthesis(tts_device, tts_model, vocoder, denoiser, response, spk)
                # sometimes it does just an emoji... just say nice
                else:
                    play_only_synthesis(tts_device, tts_model, vocoder, denoiser, 'nice', spk)

                input(f"Press Enter when you're ready to record 🎙️ ")
                recorder = Recorder()
                recorder.start_recording("output.wav")

                result = asr_model.transcribe("output.wav")
                result = result['text']

                print(f'speaker said: {result}')
            else:
                print("I didn't hear anything, try recording again...")
                input(f"Press Enter when you're ready to record 🎙️ ")

                recorder = Recorder()
                recorder.start_recording("output.wav")

                result = asr_model.transcribe("output.wav")
                result = result['text']

                print(f'speaker said: {result}')