# EmojiVoice 🎉

An expressive pseudo Speech-to-Speech system 🗣️ for HRI experiments 🤖, a part of *Do You Feel Me?*

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

Too see per model (WhisperLive and Matcha-TTS) information and make edits within the pipeline see internal READMEs in
the respective folders

## Useage

Clone this repo

```
git clone git@github.com:rosielab/do_you_feel_me.git
```

Create conda environment or virtualenv and install the requirements
Note this repo has been tested with python 3.11.9

```
pip install requirements.txt
```

Speech-to-Speech system:

You will need to pull the llama 3 model

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

Currently the system contains 11 emoji voices: 😎🤔😍🤣🙂😮🙄😅🥲😭😡😁
If you wish to change the personality of the chatbot or the emojis used by the chatbot edit the `PROMPT` parameter

If you wish to use a different voice or add new emojis you can quickly and easily fine tune Matcha-TTS to create
your own voice

## Fine tune TTS

Matcha TTS can be fine tuned for your own emojis within as little as 2 minutes of data per emoji.
The new checkpoint can be trained directly from the base Matcha-tts checkpoint (see [README](/Matcha-TTS/README.md)
for links) or from our provided checkpoints.

You can use our script [record_audio.py](/Matcha-TTS/record_audio.py) to easily record your data and
[get_duration.ipynb](/Matcha-TTS/get_duration.ipynb) to check the duration of all of your recordings.

To record audio create a `<emoji_name>.txt` where each line is a script to read, then set the emoji and emoji name (file name), with the `EMOJI_MAPPING` parameter in `record_audio.py`

When fine tuning you will be overwriting the current voices, in general, we have produced better quality voices when
selecting a voice to overwrite that is more similar to the target voice, e.g. same accent and gender. To easily hear all the voices
along with their speaker numbers use this [hugging face space](https://huggingface.co/spaces/shivammehta25/Matcha-TTS).

Follow the information in [README](/Matcha-TTS/README.md) for fine tuning on the *vctk* checkpoint where each speaker number is an emoji number. You may see our data
and transcription set up in `emojis-hri-clean.zip` 
[here](https://drive.google.com/drive/folders/1E_YTAaQxQfFdZYAKs547bgd4epkUbz_5?usp=sharing) as an example.

Hints: for fine tuning

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

