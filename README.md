# EmojiVoice 🎉 Towards long-term controllable expressivity in robot speech

An expressive pseudo Speech-to-Speech system 🗣️ for HRI experiments 🤖, a part of *Do You Feel Me?*

### [Paige Tuttösí](https://chocobearz.github.io/), [Shivam Mehta](https://www.kth.se/profile/smehta), Zachary Syvenky, Bermet Burkanova, [Gustav Eje Henter](https://people.kth.se/~ghe/), and [Angelica Lim](https://www.rosielab.ca/)

> This is the official code implementation of EmojiVoice for [RO-MAN 2025].

We have created a wrapper for [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) to aid HRI researchers in training custom light-weight, expressive voices

We have added:
* Training files setup: examples, raw data, and 3 checkpoints (with and without optimizers)
* Additional information on the amount of data needed to fine-tune
* Scripts to record the data
* Wrappers to parse emojis in text to prompt the voices in generation time
* A conversational agent chaining ASR -> LLM -> EmojiVoice

Read the paper [here](https://arxiv.org/abs/2506.15085)

See our demo page [here](https://rosielab.github.io/emojivoice/)

## v1.0.0 updates
Emojivoice is now supports multilingual for
* French
* German
* Japanese - with an updated phonemizer

## Coming soon
Your updates! Please reach out and make PRs for any issues or needed updates

Also contact if you are interested in different languages

## Structure

The system is structured as follows:

ASR -> LLM -> TTS

### ASR
Modified version of:
* [Whisper](https://github.com/openai/whisper?tab=readme-ov-file)

### LLM
Ollama and langchain chatbot implementation of:
* [Llama3](https://ollama.com/library/llama3)

### TTS
Fine tuned:
* [Matcha TTS](https://github.com/shivammehta25/Matcha-TTS)

We currently have 3 available emoji checkpoints:
* Paige - Female, intense emotions
* Olivia - Female, subtle emotions
* Zach - Male

Current checkpoints and data can be found [here](https://drive.google.com/drive/folders/1E_YTAaQxQfFdZYAKs547bgd4epkUbz_5?usp=sharing)
We have left an empty folder (`Matcha-TTS/models`) where we suggest storing them and where they must be stored to
directly run our case-studies

Too see per model (WhisperLive and Matcha-TTS) information and make edits within the pipeline see internal READMEs in
the respective folders

## Useage

Clone this repo

```
git clone git@github.com:rosielab/do_you_feel_me.git
```

Create conda environment or virtualenv and install the requirements

```
conda create -n emojivoice python=3.11 -y
conda activate emojivoice
```

Note this repo has been tested with python 3.11.9

```
cd emojivoice/Matcha-TTS
pip install -e .
```

### Example implementations

Example implementations for case studies can be found in [case_studies](https://github.com/rosielab/emojivoice/tree/main/case_studies)

Example implementations with Pepper robot can be found in [hri-demo](https://github.com/rosielab/emojivoice/tree/main/hri-demo)

### Speech-to-Speech system:

You will need to pull the llama 3 model - This model is best for English may need to change for other languages or use cases

*If you are using Japanese it seems that this model is not very good at Japanese and we suggest trying another*

```
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```

If not already running ollama, you may need to run this before run llama3
```
ollama serve
```

You will need espeak to run Matcha-tts

```
sudo apt-get install espeak-ng
```

You will find the code for the conversational agent in `feel_me.py`

At the top you will find many possible customizations (see below) but also some variables to be set to your environment.
Specifically the path to your model checkpoints, the language (the whisper model will also need to be changed), and the 
emoji to speaker mapping this is under `TTS PARAMETERS`.

Then run:

```
python feel_me.py
```

You can end the session by saying '*end session*'

### Customize

It is possible to customize the pipeline. You can perform the following modifications:

* Modify the LLM prompt and emojis
* Change to a different LLM available from Ollama
* Change the Whisper model
* Change the temperature of the TTS and LLM
* Use a different Matcha-TTS checkpoint
* Modify the speaking rate
* Change the number of steps in the ODE solver for the TTS
* Change the TTS vocoder

All of these changes can be found at the top of the [feel_me.py](feel_me.py)

Currently the system contains 11 emoji voices: 😎🤔😍🤣🙂😮🙄😅😭😡😁
If you wish to change the personality of the chatbot or the emojis used by the chatbot edit the `PROMPT` parameter

If you wish to use a different voice or add new emojis you can quickly and easily fine tune Matcha-TTS to create
your own voice

## Fine tune TTS

Matcha TTS can be fine tuned for your own emojis within as little as 2 minutes of data per emoji.
The new checkpoint can be trained directly from the base Matcha-tts checkpoint (see [README](/Matcha-TTS/README.md)
for links) or from our provided checkpoints.

You can use our script [record_audio.py](/Matcha-TTS/record_audio.py) to easily record your data and
[get_duration.ipynb](/Matcha-TTS/get_duration.ipynb) to check the duration of all of your recordings.
If fine tuning from a checkpoint the sampling rate for the audio files must be 22050.

To record audio create a `<emoji_name>.txt` where each line is a script to read, then set the emoji and emoji name (file name), with the `EMOJI_MAPPING` parameter in `record_audio.py`

When fine tuning you will be overwriting the current voices, in general, we have produced better quality voices when
selecting a voice to overwrite that is more similar to the target voice, e.g. same accent and gender. To easily hear all the voices
along with their speaker numbers use this [hugging face space](https://huggingface.co/spaces/shivammehta25/Matcha-TTS).

Follow the information in [README](/Matcha-TTS/README.md) for fine tuning on the *vctk* checkpoint where each speaker number is an emoji number. You may see our data
and transcription set up in `emojis-hri-clean.zip` 
[here](https://drive.google.com/drive/folders/1E_YTAaQxQfFdZYAKs547bgd4epkUbz_5?usp=sharing) as an example.

#### With the multilingual update we have trained a cleaner and more robust English baseline we suggest fine tuning off of `english-emoji-base.ckpt`

We provide other base voices for other languages, however, we do not guarantee how successfully they can be fine tuned

*FOR MULTILINGUAL FINE TUNING THE CLEANERS MUST BE SET IN THE CONFIGS* see your corresponding cleaner in [cleaners](https://github.com/rosielab/emojivoice/blob/main/Matcha-TTS/matcha/text/cleaners.py))

#### Hints: for fine tuning

You want to have very clean, high quality audio for the best results

First create your own experiment and data configs following the [examples](https://github.com/rosielab/emojivoice/tree/main/Matcha-TTS/configs) mapping to your trascription
file location. The two primary configs to create (and check out the paths to the data) are one in [data](https://github.com/rosielab/emojivoice/blob/main/Matcha-TTS/configs/data/emoji_multi.yaml) and
one in [experiments](https://github.com/rosielab/emojivoice/blob/main/Matcha-TTS/configs/experiment/emoji_multi.yaml). The paths here should point to where your train and validation files are stored,
and your train and validation files should point to your audio file locations. You can test that all these files are pointing the right way before training when you run: `matcha-data-stats -i ljspeech.yaml`
as per the matcha repo training steps.

Then follow the orginal Matcha-TTS instructions

To train from a checkpoint run:
```bash
python matcha/train.py experiment=<YOUR EXPERIMENT> ckpt_path=<PATH TO CHECKPOINT>
```

You can train off of the matcha base release checkpoints or the emojivoice checkpoints.

To run multi-speaker synthesis:

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT> --spk <SPEAKER NUMBER> --vocoder hifigan_univ_v1 --speaking_rate <SPEECH RATE>
```

If you are having issues, sometimes cuda will make the error messages convoluted, run training in [cpu](https://github.com/shivammehta25/Matcha-TTS/blob/main/configs/trainer/default.yaml)(set accelerator to cpu and remove devices)
mode to get more clear error outputs.

### Command line synthesis

## Installation

1. Create an environment (suggested but optional)

```
conda create -n emojivoice python=3.11 -y
conda activate emojivoice
```

2. Install Matcha TTS from source

```bash
cd emojivoice/Matcha-TTS
pip install -e .
```

3. Run CLI

We have added a play only option, which is used in the emojivoice experiment set ups. Here the audio is played and no .wav file is saved

*The default language is English, please ensure you provide the correct language to match your checkpoint*

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT> --play
```

Language other than English
```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT> --play --language fr
```

To save the audio file

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT>
```

### CLI Arguments

- To synthesise from a file, run:

```bash
matcha-tts --file <PATH TO FILE> --checkpoint_path <PATH TO CHECKPOINT> --play 
```

- To batch synthesise from a file, run:

```bash
matcha-tts --file <PATH TO FILE> --checkpoint_path <PATH TO CHECKPOINT> --batched --play
```

Additional arguments

- Speaking rate

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT> --speaking_rate 1.0 --play
```

- Sampling temperature

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT> --temperature 0.667 --play
```

- Euler ODE solver steps

```bash
matcha-tts --text "<INPUT TEXT>" --checkpoint_path <PATH TO CHECKPOINT> --steps 10 --play
```
