import click
import copy
import httpx
import ijson
import json
import llm
from pydantic import Field
from typing import Optional
from pathlib import Path
import time

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

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#supported_models_2
GOOGLE_SEARCH_MODELS = {
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
}

# List of hardcoded models
HARDCODED_MODELS = [
    "gemini-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b-latest",
    "gemini-1.5-flash-8b-001",
    "gemini-exp-1114",
    "gemini-exp-1121",
    "gemini-exp-1206",
    "gemini-2.0-flash-exp",
    "learnlm-1.5-pro-experimental",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.0-flash-thinking-exp-01-21",
    # Released 5th Feb 2025:
    "gemini-2.0-flash",
    "gemini-2.0-pro-exp-02-05",
    # Released 25th Feb 2025:
    "gemini-2.0-flash-lite",
    # Released 12th March 2025:
    "gemma-3-27b-it",
    # Image generation model registered as GeminiPro
    "imagen-3.0-generate-002",
]

def fetch_cached_json(url, path, cache_timeout, params=None):
    """Fetch JSON from URL with caching and fallback to cached data if request fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)

    try:
        response = httpx.get(url, params=params, follow_redirects=True, timeout=10.0)
        response.raise_for_status()
        
        with open(path, "w") as file:
            json.dump(response.json(), file)
            
        return response.json()
    except Exception:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        # If no cache available, return an empty structure that won't break anything
        return {"models": []}

# Function to get additional models not in our hardcoded list
def get_additional_gemini_models():
    """Fetches additional Gemini models not in our hardcoded list."""
    key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
    
    # Early return if no key available
    if not key:
        return []
        
    params = {"key": key}
    cache_path = llm.user_dir() / "gemini_models.json"
    cache_timeout = 3600 * 24  # Cache for 24 hours
    
    # Get known model IDs as a set for efficient lookups
    known_models = set(HARDCODED_MODELS)
    
    # Exclude embedding models and other non-chat models
    excluded_patterns = [
        "text-embedding", "embedding-gecko", "embedding-001",
        "text-bison", "chat-bison", 
    ]
    
    try:
        response_data = fetch_cached_json(
            url="https://generativelanguage.googleapis.com/v1beta/models",
            path=cache_path,
            cache_timeout=cache_timeout,
            params=params
        )
        
        # Process API response for new models
        new_models = []
        for model in response_data.get("models", []):
            model_name = model["name"].replace("models/", "")
            
            # Skip if already in hardcoded models
            if model_name in known_models:
                continue
                
            # Skip if matches exclusion patterns
            if any(pattern in model_name for pattern in excluded_patterns):
                continue
                
            # Add the model
            new_models.append(model_name)
        
        return new_models
    except Exception:
        # Return empty list if anything fails
        return []

@llm.hookimpl
def register_models(register):
    # Register all our hardcoded models with their proper capabilities
    for model_id in HARDCODED_MODELS:
        # Skip imagen model as it needs special handling
        if model_id == "imagen-3.0-generate-002":
            continue
            
        can_google_search = model_id in GOOGLE_SEARCH_MODELS
        can_generate_images = model_id == "gemini-2.0-flash-exp"
        register(
            GeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_schema="flash-thinking" not in model_id,
                can_generate_images=can_generate_images,
            ),
            AsyncGeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_schema="flash-thinking" not in model_id,
                can_generate_images=can_generate_images,
            ),
        )
    
    # Register Imagen model (disguised as GeminiPro)
    register(
        ImagenModel("imagen-3.0-generate-002"),
        AsyncImagenModel("imagen-3.0-generate-002"),
    )
    
    # Then register any additional models discovered from the API
    for model_id in get_additional_gemini_models():
        # Conservative defaults for dynamically discovered models
        can_google_search = False  # Only enable for known models
        can_generate_images = False  # Only enable for known models
        register(
            GeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_schema=True,
                can_generate_images=can_generate_images,
            ),
            AsyncGeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_schema=True,
                can_generate_images=can_generate_images,
            ),
        )


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    if mime_type == "application/ogg":
        mime_type = "audio/ogg"
    return mime_type


def cleanup_schema(schema, in_properties=False):
    "Gemini supports only a subset of JSON schema"
    keys_to_remove = ("$schema", "additionalProperties", "title")

    if isinstance(schema, dict):
        # Only remove keys if we're not inside a 'properties' block.
        if not in_properties:
            for key in keys_to_remove:
                schema.pop(key, None)
        for key, value in list(schema.items()):
            # If the key is 'properties', set the flag for its value.
            if key == "properties" and isinstance(value, dict):
                cleanup_schema(value, in_properties=True)
            else:
                cleanup_schema(value, in_properties=in_properties)
    elif isinstance(schema, list):
        for item in schema:
            cleanup_schema(item, in_properties=in_properties)
    return schema


class _SharedGemini:
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True
    supports_schema = True

    attachment_types = (
        # Text
        "text/plain",
        "text/csv",
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
        "application/ogg",
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
        "video/quicktime",
    )

    class Options(llm.Options):
        code_execution: Optional[bool] = Field(
            description="Enables the model to generate and run Python code",
            default=None,
        )
        temperature: Optional[float] = Field(
            description=(
                "Controls the randomness of the output. Use higher values for "
                "more creative responses, and lower values for more "
                "deterministic responses."
            ),
            default=None,
            ge=0.0,
            le=2.0,
        )
        max_output_tokens: Optional[int] = Field(
            description="Sets the maximum number of tokens to include in a candidate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description=(
                "Changes how the model selects tokens for output. Tokens are "
                "selected from the most to least probable until the sum of "
                "their probabilities equals the topP value."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        )
        top_k: Optional[int] = Field(
            description=(
                "Changes how the model selects tokens for output. A topK of 1 "
                "means the selected token is the most probable among all the "
                "tokens in the model's vocabulary, while a topK of 3 means "
                "that the next token is selected from among the 3 most "
                "probable using the temperature."
            ),
            default=None,
            ge=1,
        )
        json_object: Optional[bool] = Field(
            description="Output a valid JSON object {...}",
            default=None,
        )
        number_of_images: Optional[int] = Field(
            description="Number of images to generate (1-4, for image generation models)",
            default=None,
        )
        aspect_ratio: Optional[str] = Field(
            description="Aspect ratio for generated images (1:1, 3:4, 4:3, 9:16, 16:9)",
            default=None,
        )

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )

    def __init__(self, model_id, can_google_search=False, can_schema=False, can_generate_images=False):
        self.model_id = model_id
        self.can_google_search = can_google_search
        self.supports_schema = can_schema
        self.can_generate_images = can_generate_images
        if can_google_search:
            self.Options = self.OptionsWithGoogleSearch

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
                if response.prompt.prompt:
                    parts.append({"text": response.prompt.prompt})
                messages.append({"role": "user", "parts": parts})
                messages.append(
                    {"role": "model", "parts": [{"text": response.text_or_raise()}]}
                )

        parts = []
        if prompt.prompt:
            parts.append({"text": prompt.prompt})
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

    def build_request_body(self, prompt, conversation):
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.options and prompt.options.code_execution:
            body["tools"] = [{"codeExecution": {}}]
        if prompt.options and self.can_google_search and prompt.options.google_search:
            body["tools"] = [{"google_search": {}}]
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        # Set up generation config
        generation_config = {}

        # Handle schema requests
        if prompt.schema:
            generation_config["response_mime_type"] = "application/json"
            generation_config["response_schema"] = cleanup_schema(copy.deepcopy(prompt.schema))
        
        # Handle JSON object requests
        if prompt.options and prompt.options.json_object:
            generation_config["response_mime_type"] = "application/json"
            
        # Enable image generation for compatible models
        if self.can_generate_images:
            generation_config["response_modalities"] = ["Text", "Image"]
            
            # Add image generation options if specified
            if prompt.options:
                if getattr(prompt.options, "number_of_images", None):
                    generation_config["numberOfImages"] = prompt.options.number_of_images
                if getattr(prompt.options, "aspect_ratio", None):
                    generation_config["aspectRatio"] = prompt.options.aspect_ratio
        
        # Add other standard parameters
        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        
        if prompt.options:
            for key, other_key in config_map.items():
                config_value = getattr(prompt.options, key, None)
                if config_value is not None:
                    generation_config[other_key] = config_value
        
        # Add the generation config if we have any settings
        if generation_config:
            body["generationConfig"] = generation_config

        return body

    def display_image_in_terminal(self, filename):
        """
        Attempts to display an image in the terminal, focusing on iTerm imgcat functionality.
        Suppress errors but ensure the image displays correctly.
        """
        import os
        import subprocess
        
        # Simple approach - try imgcat first, then fall back to open on macOS
        try:
            # For imgcat - we need to NOT redirect stdout for the image to display
            # but we DO want to suppress stderr where errors appear
            with open(os.devnull, 'w') as devnull:
                subprocess.run(["imgcat", filename], stderr=devnull, check=False)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            # Fall back to open on macOS
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.run(["open", filename], stderr=devnull, check=False)
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return False

    def save_image(self, image_data, prompt_text, index=0):
        """Save a generated image to the current working directory with a descriptive name."""
        import base64
        from PIL import Image
        from io import BytesIO
        import os
        
        # Generate a unique filename based on timestamp and prompt
        timestamp = int(time.time())
        # Clean the prompt to use in filename (first 20 chars)
        clean_prompt = "".join(c for c in prompt_text[:20] if c.isalnum() or c in (' ', '_')).strip()
        clean_prompt = clean_prompt.replace(' ', '_')
        
        # Create filename
        filename = f"gemini_image_{clean_prompt}_{timestamp}_{index}.png"
        
        # Save the image
        try:
            # Try to decode if it's base64
            try:
                image_bytes = base64.b64decode(image_data)
            except:
                # If not base64, assume it's already binary
                image_bytes = image_data
                
            image = Image.open(BytesIO(image_bytes))
            image.save(filename)
            
            # Try to display the image in terminal
            self.display_image_in_terminal(filename)
                    
            return filename
        except Exception as e:
            return f"Error saving image: {str(e)}"

    def process_part(self, part, prompt_text=""):
        if "text" in part:
            return part["text"]
        elif "executableCode" in part:
            return f'```{part["executableCode"]["language"].lower()}\n{part["executableCode"]["code"].strip()}\n```\n'
        elif "codeExecutionResult" in part:
            return f'```\n{part["codeExecutionResult"]["output"].strip()}\n```\n'
        elif "inlineData" in part:
            mime_type = part["inlineData"].get("mimeType", "")
            if mime_type.startswith("image/"):
                # Save the image and return its path with new elegant format
                image_path = self.save_image(part["inlineData"]["data"], prompt_text)
                return f"\n\n⬚ ⇴ {image_path}\n\n"
        return ""

    def process_candidates(self, candidates, prompt_text=""):
        # We only use the first candidate
        for part in candidates[0]["content"]["parts"]:
            yield self.process_part(part, prompt_text)

    def set_usage(self, response):
        try:
            # Don't record the "content" key from that last candidate
            for candidate in response.response_json["candidates"]:
                candidate.pop("content", None)
            usage = response.response_json.pop("usageMetadata")
            input_tokens = usage.pop("promptTokenCount", None)
            output_tokens = usage.pop("candidatesTokenCount", None)
            usage.pop("totalTokenCount", None)
            if input_tokens is not None:
                response.set_usage(
                    input=input_tokens, output=output_tokens, details=usage or None
                )
        except (IndexError, KeyError):
            pass


class GeminiPro(_SharedGemini, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation)

        with httpx.stream(
            "POST",
            url,
            timeout=None,
            headers={"x-goog-api-key": self.get_key(key)},
            json=body,
        ) as http_response:
            events = ijson.sendable_list()
            coro = ijson.items_coro(events, "item")
            for chunk in http_response.iter_bytes():
                coro.send(chunk)
                if events:
                    for event in events:
                        if isinstance(event, dict) and "error" in event:
                            raise llm.ModelError(event["error"]["message"])
                        try:
                            yield from self.process_candidates(event["candidates"], prompt.prompt)
                        except KeyError:
                            yield ""
                        gathered.append(event)
                    events.clear()
        response.response_json = gathered[-1]
        self.set_usage(response)


class AsyncGeminiPro(_SharedGemini, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                timeout=None,
                headers={"x-goog-api-key": self.get_key(key)},
                json=body,
            ) as http_response:
                events = ijson.sendable_list()
                coro = ijson.items_coro(events, "item")
                async for chunk in http_response.aiter_bytes():
                    coro.send(chunk)
                    if events:
                        for event in events:
                            if isinstance(event, dict) and "error" in event:
                                raise llm.ModelError(event["error"]["message"])
                            try:
                                for chunk in self.process_candidates(
                                    event["candidates"], prompt.prompt
                                ):
                                    yield chunk
                            except KeyError:
                                yield ""
                            gathered.append(event)
                        events.clear()
        response.response_json = gathered[-1]
        self.set_usage(response)


@llm.hookimpl
def register_embedding_models(register):
    register(GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"))
    # gemini-embedding-exp-03-07 in different truncation sizes
    register(
        GeminiEmbeddingModel(
            "gemini-embedding-exp-03-07", "gemini-embedding-exp-03-07"
        ),
    )
    for i in (128, 256, 512, 1024, 2048):
        register(
            GeminiEmbeddingModel(
                f"gemini-embedding-exp-03-07-{i}", f"gemini-embedding-exp-03-07", i
            ),
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def gemini():
        "Commands relating to the llm-gemini plugin"

    @gemini.command()
    @click.option("--key", help="API key to use")
    def models(key):
        "List of Gemini models pulled from their API"
        key = llm.get_key(key, "gemini", "LLM_GEMINI_KEY")
        try:
            cache_path = llm.user_dir() / "gemini_models.json"
            response_data = fetch_cached_json(
                url="https://generativelanguage.googleapis.com/v1beta/models",
                path=cache_path,
                cache_timeout=3600 * 24,
                params={"key": key}
            )
            models = response_data.get("models", [])
            click.echo(json.dumps(models, indent=2))
        except Exception as e:
            click.echo(f"Error fetching models: {e}")


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    batch_size = 20

    def __init__(self, model_id, gemini_model_id, truncate=None):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id
        self.truncate = truncate

    def embed_batch(self, items):
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.get_key(),
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
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:batchEmbedContents",
                headers=headers,
                json=data,
                timeout=None,
            )
        response.raise_for_status()
        values = [item["values"] for item in response.json()["embeddings"]]
        if self.truncate:
            values = [value[: self.truncate] for value in values]
        return values


# Add back the ImagenModel classes
class ImagenModel(llm.KeyModel):
    """Model for Imagen image generation."""
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = False
    
    class Options(llm.Options):
        number_of_images: Optional[int] = Field(
            description="Number of images to generate (1-4)",
            default=1,
        )
        aspect_ratio: Optional[str] = Field(
            description="Aspect ratio of generated images (1:1, 3:4, 4:3, 9:16, 16:9)",
            default="1:1",
        )
        
    def __init__(self, model_id):
        self.model_id = model_id

    def __str__(self):
        # Make it appear as GeminiPro in model listings
        return f"GeminiPro: {self.model_id}"
    
    def display_image_in_terminal(self, filename):
        """
        Attempts to display an image in the terminal.
        """
        import os
        import subprocess
        
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.run(["imgcat", filename], stderr=devnull, check=False)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.run(["open", filename], stderr=devnull, check=False)
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return False
    
    def execute(self, prompt, stream, response, conversation, key):
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        import time
        
        # Get configuration
        number_of_images = getattr(prompt.options, "number_of_images", 1)
        aspect_ratio = getattr(prompt.options, "aspect_ratio", "1:1")
        
        try:
            # Initialize client with the key from LLM's key management
            client = genai.Client(api_key=self.get_key(key))
            
            # Make request using the correct method for Imagen
            # Use permissive safety settings (not user-configurable)
            generation_response = client.models.generate_image(
                model=self.model_id,
                prompt=prompt.prompt,
                config=types.GenerateImageConfig(
                    number_of_images=number_of_images,
                    aspect_ratio=aspect_ratio,
                    # safety_filter_level="BLOCK_ONLY_HIGH",  # Most permissive option
                    # person_generation="ALLOW_ADULT"  # Most permissive option
                )
            )
            
            # Check if any images were returned
            if not hasattr(generation_response, 'generated_images') or generation_response.generated_images is None:
                message = "No images were returned. Try a different prompt."
                response._text = message
                return message
            
            # Process the images
            image_paths = []
            for i, generated_image in enumerate(generation_response.generated_images):
                # Generate a unique filename 
                timestamp = int(time.time())
                clean_prompt = "".join(c for c in prompt.prompt[:20] if c.isalnum() or c in (' ', '_')).strip()
                clean_prompt = clean_prompt.replace(' ', '_')
                
                # Create filename
                filename = f"imagen_{clean_prompt}_{timestamp}_{i+1}.png"
                
                # Save the image
                try:
                    image = Image.open(BytesIO(generated_image.image.image_bytes))
                    image.save(filename)
                    image_paths.append(filename)
                    
                    # Display the image in terminal if possible
                    self.display_image_in_terminal(filename)
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
            
            # Construct response text with image paths - elegant format
            if image_paths:
                paths_text = "\n".join([f"⬚ ⇴ {path}" for path in image_paths])
                response_text = f"\n\n{paths_text}"
            else:
                response_text = "No images were returned. Try a different prompt."
            
            response._text = response_text
            return response_text
            
        except Exception as e:
            # Simplified error handling
            response._text = "No images were returned. Try a different prompt."
            return response._text


class AsyncImagenModel(llm.AsyncKeyModel):
    """Async Model for Imagen image generation."""
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = False
    
    class Options(llm.Options):
        number_of_images: Optional[int] = Field(
            description="Number of images to generate (1-4)",
            default=1,
        )
        aspect_ratio: Optional[str] = Field(
            description="Aspect ratio of generated images (1:1, 3:4, 4:3, 9:16, 16:9)",
            default="1:1",
        )
        
    def __init__(self, model_id):
        self.model_id = model_id

    def __str__(self):
        # Make it appear as GeminiPro in model listings
        return f"GeminiPro: {self.model_id}"
        
    def display_image_in_terminal(self, filename):
        """
        Attempts to display an image in the terminal.
        """
        import os
        import subprocess
        
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.run(["imgcat", filename], stderr=devnull, check=False)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                with open(os.devnull, 'w') as devnull:
                    subprocess.run(["open", filename], stderr=devnull, check=False)
                return True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return False
    
    async def execute(self, prompt, stream, response, conversation, key):
        import asyncio
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        import time
        
        # Get configuration
        number_of_images = getattr(prompt.options, "number_of_images", 1)
        aspect_ratio = getattr(prompt.options, "aspect_ratio", "1:1")
        
        try:
            # Since the SDK might not have async support, we run it in a thread pool
            loop = asyncio.get_event_loop()
            
            # Define a function to create client and make the request
            def generate_images():
                client = genai.Client(api_key=self.get_key(key))
                return client.models.generate_image(
                    model=self.model_id,
                    prompt=prompt.prompt,
                    config=types.GenerateImageConfig(
                        number_of_images=number_of_images,
                        aspect_ratio=aspect_ratio,
                        # safety_filter_level="BLOCK_ONLY_HIGH",  # Most permissive option
                        # person_generation="ALLOW_ADULT"  # Most permissive option
                    )
                )
            
            # Run the synchronous SDK code in an executor
            generation_response = await loop.run_in_executor(None, generate_images)
            
            # Check if any images were returned
            if not hasattr(generation_response, 'generated_images') or generation_response.generated_images is None:
                message = "No images were returned. Try a different prompt."
                response._text = message
                return message
            
            # Process the images
            image_paths = []
            for i, generated_image in enumerate(generation_response.generated_images):
                # Generate a unique filename
                timestamp = int(time.time())
                clean_prompt = "".join(c for c in prompt.prompt[:20] if c.isalnum() or c in (' ', '_')).strip()
                clean_prompt = clean_prompt.replace(' ', '_')
                
                # Create filename
                filename = f"imagen_{clean_prompt}_{timestamp}_{i+1}.png"
                
                # Save the image - run file operations in executor to avoid blocking
                try:
                    def save_image():
                        image = Image.open(BytesIO(generated_image.image.image_bytes))
                        image.save(filename)
                        
                        # Display the image in terminal if possible
                        self.display_image_in_terminal(filename)
                        
                        return filename
                    
                    saved_filename = await loop.run_in_executor(None, save_image)
                    image_paths.append(saved_filename)
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
            
            # Construct response text with image paths - elegant format
            if image_paths:
                paths_text = "\n".join([f"⬚ ⇴ {path}" for path in image_paths])
                response_text = f"\n\n{paths_text}"
            else:
                response_text = "No images were returned. Try a different prompt."
            
            response._text = response_text
            return response_text
            
        except Exception as e:
            # Simplified error handling
            response._text = "No images were returned. Try a different prompt."
            return response._text
