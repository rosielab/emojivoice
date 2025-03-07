# %%
import numpy as np
from pathlib import Path
import soundfile as sf
import torch


from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import get_user_data_dir, intersperse, assert_model_downloaded

import emoji

VOICE = 'emoji'
SCRIPT_PATH = "fairytale_script.txt"
WAV_PATH = "outputs"
#CLARITY = False
############################ TTS PARAMETERS ############################################################################
if VOICE == 'base' :
    TTS_MODEL_PATH = "../../Matcha-TTS/matcha_vctk.ckpt"
    SPEAKING_RATE = 0.8
    STEPS = 10
else:
    TTS_MODEL_PATH = "../../Matcha-TTS/emoji-hri-paige-inference.ckpt"
    SPEAKING_RATE = 0.8
    STEPS = 10
# hifigan_univ_v1 is suggested, unless the custom model is trained on LJ Speech
VOCODER_NAME= "hifigan_univ_v1"
TTS_TEMPERATURE = 0.667
VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",  # Old url: https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=drive_link
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",  # Old url: https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link
}

#maps the emojis used by the LLM to the speaker numbers from the Matcha-TTS checkpoint
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

########################################################################################################################

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
    sf.write(folder / f"to_play-{filename}.wav", output["waveform"], 22050, "PCM_24")

def play_only_synthesis(i, device, model, vocoder, denoiser, text, spk):
    text = text.strip()
    text_processed = process_text(0, text, device, True)

    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=STEPS,
        temperature=TTS_TEMPERATURE,
        spks=spk,
        length_scale=SPEAKING_RATE,
        #clarity = CLARITY,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

    output["waveform"] = np.clip(output["waveform"], -1.0, 1.0)

    save_to_folder(i, output, WAV_PATH)

def assert_required_models_available():
    save_dir = get_user_data_dir()
    model_path = TTS_MODEL_PATH

    vocoder_path = save_dir / f"{VOCODER_NAME}"
    assert_model_downloaded(vocoder_path, VOCODER_URLS[VOCODER_NAME])
    return {"matcha": model_path, "vocoder": vocoder_path}

def contains_only_non_emoji(string):
    return all(not emoji.is_emoji(char) for char in string) and len(string.strip()) > 0

if __name__ == "__main__":

    tts_device = "cuda" if torch.cuda.is_available() else "cpu"
    #tts_device = "cpu"
    paths = assert_required_models_available()

    save_dir = get_user_data_dir()
 
    tts_model = load_matcha(paths["matcha"], tts_device)
    vocoder, denoiser = load_vocoder(VOCODER_NAME, paths["vocoder"], tts_device)

    with open(SCRIPT_PATH, 'r') as file:
        for i, line in enumerate(file):
            # Strip any extra whitespace (like newlines)
            clean_line = line.strip()
            if VOICE == 'emoji':
                spk = torch.tensor([12], device=tts_device, dtype=torch.long)
                for emote in emoji_mapping:
                    if emote in clean_line:
                        spk = torch.tensor([emoji_mapping[emote]], device=tts_device, dtype=torch.long)
                        break
            elif VOICE == 'base':
                spk = torch.tensor([1], device=tts_device, dtype=torch.long)
            elif VOICE == 'default':
                spk = torch.tensor([12], device=tts_device, dtype=torch.long)
            else:
                print("hmmm wrong voice")
            clean_line = emoji.replace_emoji(clean_line, '')
            #matcha cannot handle brackets
            clean_line = clean_line.replace(')', '')
            clean_line = clean_line.replace('(', '')
            play_only_synthesis(i, tts_device, tts_model, vocoder, denoiser, clean_line, spk)


