import time
import json
from pathlib import Path
import httpx
import ijson
import llm
from pydantic import Field
from typing import Optional

import urllib.parse

# We disable all of these to avoid random unexpected errors
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
]

@llm.hookimpl
def register_models(register):
    # Only do this if the Gemini key is set
    key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
    if not key:
        return

    latest_models = {}
    for model in get_available_gemini_models():
        model_name = model["name"].replace("models/", "")  # Remove "models/" prefix
        base_name = model_name.split("-latest")[0]  # Extract base name

        # Store only the latest version for each base name
        if base_name not in latest_models or model_name.endswith("-latest"):
            latest_models[base_name] = model_name

    for model_name in latest_models.values():
        register(GeminiPro(model_name))

def get_available_gemini_models():
    """Fetches the available Gemini models from the Google AI API with caching."""
    return fetch_cached_json(
        url="https://generativelanguage.googleapis.com/v1beta/models",
        path=llm.user_dir() / "gemini_models.json",
        cache_timeout=3600,  # Cache for 1 hour
        params={"key": llm.get_key("", "gemini", "LLM_GEMINI_KEY")}
    )["models"]


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout, params=None):
    path = Path(path)

    # Create directories if not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        try:
            # Get the file's modification time
            mod_time = path.stat().st_mtime
            # Check if it's more than the cache_timeout old
            if time.time() - mod_time < cache_timeout:
                # If not, load the file
                with open(path, "r") as file:
                    content = file.read()
                    if content.strip():  # Check if file is not empty
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            # If cache is corrupted, delete it
                            path.unlink()
        except (OSError, json.JSONDecodeError):
            # If there's any error reading the cache, ignore it
            pass

    # Try to download the data
    try:
        response = httpx.get(url, params=params, follow_redirects=True)
        response.raise_for_status()

        data = response.json()

        # If successful, write to the file
        try:
            with open(path, "w") as file:
                json.dump(data, file)
        except OSError:
            # If we can't write the cache, just return the data
            pass

        return data
    except httpx.HTTPError:
        # If there's an existing file, try one last time to load it
        if path.is_file():
            try:
                with open(path, "r") as file:
                    content = file.read()
                    if content.strip():  # Check if file is not empty
                        return json.loads(content)
            except (OSError, json.JSONDecodeError):
                pass

        # If all else fails, raise an error
        raise DownloadError(
            f"Failed to download data and no valid cache is available at {path}"
        )

def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    return mime_type

class GeminiPro(llm.Model):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True

    attachment_types = (
        # PDF
        "application/pdf",
        # Images
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
        # Audio
        "audio/wav",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/mpeg",  # Treated as audio/mp3
        # Video
        "video/mp4",
        "video/mpeg",
        "video/mov",
        "video/avi",
        "video/x-flv",
        "video/mpg",
        "video/webm",
        "video/wmv",
        "video/3gpp",
    )

    class Options(llm.Options):
        code_execution: Optional[bool] = Field(
            description="Enables the model to generate and run Python code",
            default=None,
        )
        temperature: Optional[float] = Field(
            description="Controls the randomness of the output. Use higher values for more creative responses, and lower values for more deterministic responses.",
            default=None,
            ge=0.0,
            le=2.0,
        )
        max_output_tokens: Optional[int] = Field(
            description="Sets the maximum number of tokens to include in a candidate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description="Changes how the model selects tokens for output. Tokens are selected from the most to least probable until the sum of their probabilities equals the topP value.",
            default=None,
            ge=0.0,
            le=1.0,
        )
        top_k: Optional[int] = Field(
            description="Changes how the model selects tokens for output. A topK of 1 means the selected token is the most probable among all the tokens in the model's vocabulary, while a topK of 3 means that the next token is selected from among the 3 most probable using the temperature.",
            default=None,
            ge=1,
        )

    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                parts = []
                for attachment in response.attachments:
                    mime_type = resolve_type(attachment)
                    parts.append(
                        {
                            "inlineData": {
                                "data": attachment.base64_content(),
                                "mimeType": mime_type,
                            }
                        }
                    )
                parts.append({"text": response.prompt.prompt})
                messages.append({"role": "user", "parts": parts})
                messages.append({"role": "model", "parts": [{"text": response.text()}]})

        parts = [{"text": prompt.prompt}]
        for attachment in prompt.attachments:
            mime_type = resolve_type(attachment)
            parts.append(
                {
                    "inlineData": {
                        "data": attachment.base64_content(),
                        "mimeType": mime_type,
                    }
                }
            )

        messages.append({"role": "user", "parts": parts})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?".format(
            self.model_id
        ) + urllib.parse.urlencode(
            {"key": key}
        )
        gathered = []
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.options and prompt.options.code_execution:
            body["tools"] = [{"codeExecution": {}}]
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        # If any of those are set in prompt.options...
        if any(
            getattr(prompt.options, key, None) is not None for key in config_map.keys()
        ):
            generation_config = {}
            for key, other_key in config_map.items():
                config_value = getattr(prompt.options, key, None)
                if config_value is not None:
                    generation_config[other_key] = config_value
            body["generationConfig"] = generation_config

        with httpx.stream(
            "POST",
            url,
            timeout=None,
            json=body,
        ) as http_response:
            events = ijson.sendable_list()
            coro = ijson.items_coro(events, "item")
            for chunk in http_response.iter_bytes():
                coro.send(chunk)
                if events:
                    event = events[0]
                    if isinstance(event, dict) and "error" in event:
                        raise llm.ModelError(event["error"]["message"])
                    try:
                        part = event["candidates"][0]["content"]["parts"][0]
                        if "text" in part:
                            yield part["text"]
                        elif "executableCode" in part:
                            # For code_execution
                            yield f'```{part["executableCode"]["language"].lower()}\n{part["executableCode"]["code"].strip()}\n```\n'
                        elif "codeExecutionResult" in part:
                            # For code_execution
                            yield f'```\n{part["codeExecutionResult"]["output"].strip()}\n```\n'
                    except KeyError:
                        yield ""
                    gathered.append(event)
                    events.clear()
        response.response_json = gathered


@llm.hookimpl
def register_embedding_models(register):
    register(
        GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"),
    )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    batch_size = 20

    def __init__(self, model_id, gemini_model_id):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id

    def embed_batch(self, items):
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "requests": [
                {
                    "model": "models/" + self.gemini_model_id,
                    "content": {"parts": [{"text": item}]},
                }
                for item in items
            ]
        }

        with httpx.Client() as client:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:batchEmbedContents?key={self.get_key()}",
                headers=headers,
                json=data,
                timeout=None,
            )

        response.raise_for_status()
        return [item["values"] for item in response.json()["embeddings"]]
