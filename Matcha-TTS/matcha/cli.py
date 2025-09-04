import argparse
import datetime as dt
import os
import warnings
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir, intersperse

VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",  # Old url: https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=drive_link
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",  # Old url: https://drive.google.com/file/d/1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW/view?usp=drive_link
}


def plot_spectrogram_to_numpy(spectrogram, filename):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.savefig(filename)

def process_text(i: int, text: str, device: torch.device, language: str, play: bool):
    cleaners = {
        "en": "english_cleaners2",
        "fr": "french_cleaners",
        "ja": "japanese_cleaners",
        "es": "spanish_cleaners",
        "de": "german_cleaners",
    }
    if language not in cleaners:
        print("Invalid language. Current supported languages: en (English), fr (French), ja (Japanese), de (German).")
        sys.exit(1)

    if not play:
        print(f"[{i}] - Input text: {text}")

    x = torch.tensor(
        intersperse(text_to_sequence(text, [cleaners[language]])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())

    if not play:
        print(f"[{i}] - Phonetised text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    save_dir = get_user_data_dir()
    model_path = args.checkpoint_path
    vocoder_path = save_dir / f"{args.vocoder}"
    assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    return {"matcha": model_path, "vocoder": vocoder_path}


def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def load_vocoder(vocoder_name, checkpoint_path, device, play):
    if not play:
        print(f"[!] Loading {vocoder_name}!")
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    else:
        raise NotImplementedError(
            f"Vocoder {vocoder_name} not implemented! define a load_<<vocoder_name>> method for it"
        )

    denoiser = Denoiser(vocoder, mode="zeros")
    if not play:
        print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser


def load_matcha(model_name, checkpoint_path, device, play):
    if not play:
        print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()

    if not play:
        print(f"[+] {model_name} loaded!")
    return model


def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()

    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    plot_spectrogram_to_numpy(np.array(output["mel"].squeeze().float().cpu()), f"{filename}.png")
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")
    return folder.resolve() / f"{filename}.wav"


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need some text to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.vocoder == "hifigan_T2_v1":
        warn_ = "[-] I would suggest passing --vocoder hifigan_univ_v1, unless the custom model is trained on LJ Speech."
        warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = "hifigan_univ_v1"

    if args.speaking_rate is None:
        args.speaking_rate = 1.0

    if args.batched:
        assert args.batch_size > 0, "Batch size must be greater than 0"
    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args

@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the custom model checkpoint",
        required=True
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default=None,
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
        choices=VOCODER_URLS.keys(),
    )
    parser.add_argument("--play", action='store_true', help="Play the synthesized text without saving")
    parser.add_argument("--language", type=str, default="en", help="Synthesis language: English default")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--spk", type=int, default=None, help="Speaker ID")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of ODE steps  (default: 10)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default= 0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument("--batched", action="store_true", help="Batched inference (default: False)")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size only useful when --batched (default: 32)"
    )

    args = parser.parse_args()

    play = args.play
    args = validate_args(args)
    device = get_device(args, play)
    language = args.language
    if not play:
        print_config(args)
    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        paths["matcha"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_matcha(args.model, paths["matcha"], device, play)
    vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device, play)

    texts = get_texts(args)

    if args.spk != None:
        spk = torch.tensor([args.spk], device=device, dtype=torch.long)
    else:
        warn_ = "[-] No speaker provided, using speaker number 0."
        warnings.warn(warn_, UserWarning)
        spk = torch.tensor([0], device=device, dtype=torch.long)
    if args.play:
        if not args.file:
            play_only_synthesis(args, device, model, vocoder, denoiser, texts, spk, language)
        else:
            file_synthesis_play_only(args, device, model, vocoder, denoiser, texts, spk, language)
    elif len(texts) == 1 or not args.batched:
        unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk, language)
    else:
        batched_synthesis(args, device, model, vocoder, denoiser, texts, spk, language)


class BatchedSynthesisDataset(torch.utils.data.Dataset):
    def __init__(self, processed_texts):
        self.processed_texts = processed_texts

    def __len__(self):
        return len(self.processed_texts)

    def __getitem__(self, idx):
        return self.processed_texts[idx]


def batched_collate_fn(batch):
    x = []
    x_lengths = []

    for b in batch:
        x.append(b["x"].squeeze(0))
        x_lengths.append(b["x_lengths"])

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x_lengths = torch.concat(x_lengths, dim=0)
    return {"x": x, "x_lengths": x_lengths}


def batched_synthesis(args, device, model, vocoder, denoiser, texts, spk, language):
    total_rtf = []
    total_rtf_w = []
    processed_text = [process_text(i, text, "cpu", language, None) for i, text in enumerate(texts)]
    dataloader = torch.utils.data.DataLoader(
        BatchedSynthesisDataset(processed_text),
        batch_size=args.batch_size,
        collate_fn=batched_collate_fn,
        num_workers=8,
    )
    for i, batch in enumerate(dataloader):
        i = i + 1
        start_t = dt.datetime.now()
        b = batch["x"].shape[0]
        output = model.synthesise(
            batch["x"].to(device),
            batch["x_lengths"].to(device),
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk.expand(b) if spk is not None else spk,
            length_scale=args.speaking_rate,
        )

        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-Batch: {i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-Batch: {i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)
        for j in range(output["mel"].shape[0]):
            base_name = f"utterance_{j:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{j:03d}"
            length = output["mel_lengths"][j]
            new_dict = {"mel": output["mel"][j][:, :length], "waveform": output["waveform"][j][: length * 256]}
            location = save_to_folder(base_name, new_dict, args.output_folder)
            print(f"[🍵-{j}] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")

def file_synthesis_play_only(args, device, model, vocoder, denoiser, texts, spk, language):
    i = 0
    print("Press enter to play each line 🍵")
    while i < len(texts):
      next = input("")
      if next == "":

        split_txt = texts[i].split("|")
        text = split_txt[0]
        text = text.strip()
        spk = int(split_txt[1])
        spk = torch.tensor([spk], device=device, dtype=torch.long)
        text_processed = process_text(i, text, device, language, True)

        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            length_scale=args.speaking_rate,
        )
        waveform = to_waveform(output["mel"], vocoder, denoiser)
        sd.play(waveform, 22050)
        sd.wait()

        i = i + 1

def play_only_synthesis(args, device, model, vocoder, denoiser, text, spk, language):
    play = True
    text = text[0].strip()
    text_processed = process_text(0, text, device, language, True)

    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=args.steps,
        temperature=args.temperature,
        spks=spk,
        length_scale=args.speaking_rate,
    )
    waveform = to_waveform(output["mel"], vocoder, denoiser)
    sd.play(waveform, 22050)
    sd.wait()
    while play:
        text = input("Enter the next text (type Xnow to exit):")
        if text == "Xnow":
            print("Thank you for stirring up some Matcha-TTS🍵")
            play = False
            break
        else:
            spk = int(input("Enter speaker number:"))
            spk = torch.tensor([spk], device=device, dtype=torch.long)
            text = text.strip()
            text_processed = process_text(0, text, device, language, True)

            output = model.synthesise(
                text_processed["x"],
                text_processed["x_lengths"],
                n_timesteps=args.steps,
                temperature=args.temperature,
                spks=spk,
                length_scale=args.speaking_rate,
            )
            waveform = to_waveform(output["mel"], vocoder, denoiser)
            sd.play(waveform, 22050)
            sd.wait()



def unbatched_synthesis(args, device, model, vocoder, denoiser, texts, spk, language):
    total_rtf = []
    total_rtf_w = []
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{args.spk:03d}" if args.spk is not None else f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        text_processed = process_text(i, text, device, language, None)

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk,
            length_scale=args.speaking_rate,
        )
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
        # RTF with HiFiGAN
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])
        print(f"[🍵-{i}] Matcha-TTS RTF: {output['rtf']:.4f}")
        print(f"[🍵-{i}] Matcha-TTS + VOCODER RTF: {rtf_w:.4f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        location = save_to_folder(base_name, output, args.output_folder)
        print(f"[+] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.checkpoint_path}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- Speaker: {args.spk}")


def get_device(args, play):
    if torch.cuda.is_available() and not args.cpu:
        if not play:
            print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        if not play:
            print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    cli()
