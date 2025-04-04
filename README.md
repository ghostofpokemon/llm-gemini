# llm-gemini

[![PyPI](https://img.shields.io/pypi/v/llm-gemini.svg)](https://pypi.org/project/llm-gemini/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-gemini?include_prereleases&label=changelog)](https://github.com/simonw/llm-gemini/releases)
[![Tests](https://github.com/simonw/llm-gemini/workflows/Test/badge.svg)](https://github.com/simonw/llm-gemini/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-gemini/blob/main/LICENSE)

API access to Google's Gemini models

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-gemini
```
## Usage

Configure the model by setting a key called "gemini" to your [API key](https://aistudio.google.com/app/apikey):
```bash
llm keys set gemini
```
```
<paste key here>
```
You can also set the API key by assigning it to the environment variable `LLM_GEMINI_KEY`.

Now run the model using `-m gemini-1.5-pro-latest`, for example:

```bash
llm -m gemini-1.5-pro-latest "A joke about a pelican and a walrus"
```

> A pelican walks into a seafood restaurant with a huge fish hanging out of its beak.  The walrus, sitting at the bar, eyes it enviously.
>
> "Hey," the walrus says, "That looks delicious! What kind of fish is that?"
>
> The pelican taps its beak thoughtfully. "I believe," it says, "it's a billfish."

Other models are:

- `gemini-1.5-flash-latest`
- `gemini-1.5-flash-8b-latest` - the least expensive
- `gemini-exp-1114` - recent experimental #1
- `gemini-exp-1121` - recent experimental #2
- `gemini-exp-1206` - recent experimental #3
- `gemini-2.0-flash-exp` - [Gemini 2.0 Flash](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/#gemini-2-0-flash)
- `learnlm-1.5-pro-experimental` - "an experimental task-specific model that has been trained to align with learning science principles" - [more details here](https://ai.google.dev/gemini-api/docs/learnlm).
- `gemini-2.0-flash-thinking-exp-1219` - experimental "thinking" model from December 2024
- `gemini-2.0-flash-thinking-exp-01-21` - experimental "thinking" model from January 2025
- `gemini-2.0-flash` - Gemini 2.0 Flash
- `gemini-2.0-flash-lite` - Gemini 2.0 Flash-Lite
- `gemini-2.0-pro-exp-02-05` - experimental release of Gemini 2.0 Pro

### Images, audio and video

Gemini models are multi-modal. You can provide images, audio or video files as input like this:

```bash
llm -m gemini-1.5-flash-latest 'extract text' -a image.jpg
```
Or with a URL:
```bash
llm -m gemini-1.5-flash-8b-latest 'describe image' \
  -a https://static.simonwillison.net/static/2024/pelicans.jpg
```
Audio works too:

```bash
llm -m gemini-1.5-pro-latest 'transcribe audio' -a audio.mp3
```

And video:

```bash
llm -m gemini-1.5-pro-latest 'describe what happens' -a video.mp4
```
The Gemini prompting guide includes [extensive advice](https://ai.google.dev/gemini-api/docs/file-prompting-strategies) on multi-modal prompting.

### JSON output

Use `-o json_object 1` to force the output to be JSON:

```bash
llm -m gemini-1.5-flash-latest -o json_object 1 \
  '3 largest cities in California, list of {"name": "..."}'
```
Outputs:
```json
{"cities": [{"name": "Los Angeles"}, {"name": "San Diego"}, {"name": "San Jose"}]}
```

### Code execution

Gemini models can [write and execute code](https://ai.google.dev/gemini-api/docs/code-execution) - they can decide to write Python code, execute it in a secure sandbox and use the result as part of their response.

To enable this feature, use `-o code_execution 1`:

```bash
llm -m gemini-1.5-pro-latest -o code_execution 1 \
'use python to calculate (factorial of 13) * 3'
```
### Google search

Some Gemini models support [Grounding with Google Search](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#web-ground-gemini), where the model can run a Google search and use the results as part of answering a prompt.

Using this feature may incur additional requirements in terms of how you use the results. Consult [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#web-ground-gemini) for more details.

To run a prompt with Google search enabled, use `-o google_search 1`:

```bash
llm -m gemini-1.5-pro-latest -o google_search 1 \
  'What happened in Ireland today?'
```

Use `llm logs -c --json` after running a prompt to see the full JSON response, which includes [additional information](https://github.com/simonw/llm-gemini/pull/29#issuecomment-2606201877) about grounded results.

### Chat

To chat interactively with the model, run `llm chat`:

```bash
llm chat -m gemini-1.5-pro-latest
```

## Embeddings

The plugin also adds support for the `gemini-embedding-exp-03-07` and `text-embedding-004` embedding models.

Run that against a single string like this:
```bash
llm embed -m text-embedding-004 -c 'hello world'
```
This returns a JSON array of 768 numbers.

The `gemini-embedding-exp-03-07` model is larger, returning 3072 numbers. You can also use variants of it that are truncated down to smaller sizes:

- `gemini-embedding-exp-03-07` - 3072 numbers
- `gemini-embedding-exp-03-07-2048` - 2048 numbers
- `gemini-embedding-exp-03-07-1024` - 1024 numbers
- `gemini-embedding-exp-03-07-512` - 512 numbers
- `gemini-embedding-exp-03-07-256` - 256 numbers
- `gemini-embedding-exp-03-07-128` - 128 numbers

This command will embed every `README.md` file in child directories of the current directory and store the results in a SQLite database called `embed.db` in a collection called `readmes`:

```bash
llm embed-multi readmes -d embed.db -m gemini-embedding-exp-03-07-128 \
  --files . '*/README.md'
```
You can then run similarity searches against that collection like this:
```bash
llm similar readmes -c 'upload csvs to stuff' -d embed.db
```

See the [LLM embeddings documentation](https://llm.datasette.io/en/stable/embeddings/cli.html) for further details.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-gemini
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record Gemini API responses for the tests.

If you add a new test that calls the API you can capture the API response like this:
```bash
PYTEST_GEMINI_API_KEY="$(llm keys get gemini)" pytest --record-mode once
```
You will need to have stored a valid Gemini API key using this command first:
```bash
llm keys set gemini
# Paste key here
```

## Image Generation

This plugin supports image generation using both Gemini 2.0 and Imagen 3.0 models:

### Gemini 2.0 Image Generation

Gemini 2.0 models (like `gemini-2.0-flash-exp`) automatically include image generation capabilities. Simply prompt the model to generate images and it will create images along with text, saving them to your current working directory.

```bash
llm -m gemini-2.0-flash-exp "Generate an image of a futuristic city with flying cars"
```

The model will decide when to include images based on your prompt. Images are automatically saved to your current working directory.

**Note:** Sometimes in longer conversations, the model may start outputting text descriptions instead of generating actual images. If this happens, you can redirect it back to image generation by saying something like: "Instead of describing the image format, please generate an actual image like you did in your first response."

### Imagen 3.0 Image Generation

For high-quality standalone images, use the dedicated Imagen model, which provides superior image quality:

```bash
llm -m imagen-3.0-generate-002 "A photorealistic image of a cat wearing a top hat"
```

You can control various parameters using the `-o` option format:

```bash
llm -m imagen-3.0-generate-002 -o number_of_images 4 -o aspect_ratio "16:9" "A beautiful sunset over the mountains"
```

Available parameters:

- `number_of_images`: Generate between 1-4 images (default: 1)
- `aspect_ratio`: Choose from "1:1", "3:4", "4:3", "9:16", "16:9" (default: "1:1")

All images are saved to your current working directory and the model's response will include links to the saved images.