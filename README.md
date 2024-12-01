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
for links) or from our provided checkpoint.

Follow the information in [README](/Matcha-TTS/README.md) for fine tuning on the *vctk* checkpoint where each speaker
is an emoji. You may see our data
[here](https://drive.google.com/drive/folders/1E_YTAaQxQfFdZYAKs547bgd4epkUbz_5?usp=sharing) as an example.

You can use our script [record_audio.py](/Matcha-TTS/record_audio.py) to easily record your data and
[get_duration.ipynb](/Matcha-TTS/get_duration.ipynb) to check the duration of all of your recordings.

To record audio create a `script.txt` where each line is a script to read, then set the emoji and emoji name in
`record_audio.py`
