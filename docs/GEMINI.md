========================
CODE SNIPPETS
========================
TITLE: File Get Example
DESCRIPTION: Retrieves information about an uploaded file using its name. This example first uploads a file and then fetches its details.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)
```

----------------------------------------

TITLE: Install Google Gen AI SDK
DESCRIPTION: Installs the google-genai package using pip. This is the initial step to use the SDK.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: shell
CODE:
```
pip install google-genai
```

----------------------------------------

TITLE: LiveClientSetup Type
DESCRIPTION: Represents the setup configuration for a live client.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientSetup:
  (No specific attributes listed in the provided text, implies a configuration object.)
```

----------------------------------------

TITLE: Model Tuning Example
DESCRIPTION: Demonstrates the setup for supervised fine-tuning of a model. It specifies the model and the training dataset, which is expected to be in GCS for Vertex AI.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

model = 'gemini-2.0-flash-001'
training_dataset = types.TuningDataset(

```

----------------------------------------

TITLE: Gemini Preference Example Types
DESCRIPTION: Structures for providing example preferences for Gemini, including completions and content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GeminiPreferenceExample:
  completions: List[GeminiPreferenceExampleCompletion]
    A list of example completions for the prompt.
  contents: List[Content]
    A list of example content turns.

GeminiPreferenceExampleDict:
  completions: List[GeminiPreferenceExampleCompletionDict]
    A list of example completions for the prompt.
  contents: List[ContentDict]
    A list of example content turns.
```

----------------------------------------

TITLE: LiveMusicClientSetup Attributes
DESCRIPTION: Details the 'model' attribute for LiveMusicClientSetup, used for configuring the client setup for live music.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicClientSetup:
  model: Specifies the model to be used for live music client setup.
```

----------------------------------------

TITLE: Live Music Server Setup Complete Types
DESCRIPTION: Defines types for indicating the completion of live music server setup.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicServerSetupComplete, LiveMusicServerSetupCompleteDict

# Example usage for LiveMusicServerSetupComplete:
setup_complete = LiveMusicServerSetupComplete()

# Example usage for LiveMusicServerSetupCompleteDict:
setup_complete_dict = LiveMusicServerSetupCompleteDict()

```

----------------------------------------

TITLE: TuningExample Fields
DESCRIPTION: Details the fields within the TuningExample, representing a single example for model tuning.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningExample:
  output: The desired output for the given input.
  text_input: The text input for the tuning example.
```

----------------------------------------

TITLE: File Upload Example
DESCRIPTION: Demonstrates uploading files to the Gemini Developer API. It shows how to upload multiple PDF files and print their information.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: console
CODE:
```
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

----------------------------------------

TITLE: Gemini Preference Example Completion Types
DESCRIPTION: Defines the structure for example completions within Gemini preferences.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GeminiPreferenceExampleCompletion:
  completion: str
    The text of the example completion.
  score: float
    A score indicating the quality of the completion.

GeminiPreferenceExampleCompletionDict:
  completion: str
    The text of the example completion.
  score: float
    A score indicating the quality of the completion.
```

----------------------------------------

TITLE: LiveMusicClientSetupDict Attributes
DESCRIPTION: Details the 'model' attribute for LiveMusicClientSetupDict, used for configuring the client setup for live music, similar to LiveMusicClientSetup.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicClientSetupDict:
  model: Specifies the model to be used for live music client setup.
```

----------------------------------------

TITLE: Cache Creation Example
DESCRIPTION: Creates a cache for content, potentially from specified file URIs. It configures the cache with contents, system instructions, a display name, and a time-to-live (TTL).

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

----------------------------------------

TITLE: Cache Get Example
DESCRIPTION: Retrieves information about a previously created cache using its name.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
cached_content = client.caches.get(name=cached_content.name)
```

----------------------------------------

TITLE: StartSensitivity Enumeration
DESCRIPTION: Defines sensitivity levels for starting a process. Includes high, low, and unspecified sensitivity options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
StartSensitivity:
  START_SENSITIVITY_HIGH: int
  START_SENSITIVITY_LOW: int
  START_SENSITIVITY_UNSPECIFIED: int
```

----------------------------------------

TITLE: TuningExampleDict Fields
DESCRIPTION: Details the fields within the TuningExampleDict, a dictionary representation of a tuning example.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningExampleDict:
  output: The desired output for the given input.
  text_input: The text input for the tuning example.
```

----------------------------------------

TITLE: LiveClientMessage Attributes
DESCRIPTION: Outlines the attributes for LiveClientMessage, including client_content, realtime_input, setup, and tool_response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientMessage:
  client_content: The content provided by the client.
  realtime_input: Real-time input data from the client.
  setup: Setup information for the client message.
  tool_response: The response from a tool used by the client.
```

----------------------------------------

TITLE: LiveServerSetupComplete Types
DESCRIPTION: Information related to the completion of the server setup process, including the session ID.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerSetupComplete:
  session_id: str | None
    The unique identifier for the current session.

LiveServerSetupCompleteDict:
  session_id: str | None
    The unique identifier for the current session.
```

----------------------------------------

TITLE: List Tuning Jobs (Synchronous)
DESCRIPTION: Lists all available tuning jobs with support for pagination. This example shows how to set the page size and retrieve jobs, including fetching the next page of results.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: File Delete Example
DESCRIPTION: Demonstrates deleting a file that has been uploaded. The example uploads a file and then removes it.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

----------------------------------------

TITLE: List Base Models (Synchronous)
DESCRIPTION: Shows how to list available base models using the synchronous client. It includes a basic loop to print model information and an example of using pagination with `page_size`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
for model in client.models.list():
    print(model)
```

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: List Tuned Models (Synchronous)
DESCRIPTION: Lists available tuned models with optional pagination and filtering. The example shows how to set the page size and iterate through the results.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}}):
    print(model)
```

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False}})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: List Models with Pagination
DESCRIPTION: Shows how to use pagination when listing models, specifying the page size and accessing models page by page. Includes examples for both synchronous and asynchronous iteration.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: SupervisedTuningDataStats API Documentation
DESCRIPTION: Contains statistics related to supervised tuning data, including dropped example reasons and token counts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDataStats:
  dropped_example_reasons: Reasons why examples were dropped during tuning.
  total_billable_character_count: Total billable character count in the dataset.
  total_billable_token_count: Total billable token count in the dataset.
```

----------------------------------------

TITLE: List Base Models (Asynchronous)
DESCRIPTION: Demonstrates listing available base models using the asynchronous client. It includes an async for loop and examples of handling asynchronous pagers with `page_size` and `next_page`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list():
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Configure SOCKS5 Proxy for httpx
DESCRIPTION: Sets HttpOptions to use a SOCKS5 proxy for both synchronous and asynchronous httpx clients, requiring httpx[socks] installation.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

http_options = types.HttpOptions(
    client_args={'proxy': 'socks5://user:pass@host:port'},
    async_client_args={'proxy': 'socks5://user:pass@host:port'},
)

client=genai.Client(..., http_options=http_options)
```

----------------------------------------

TITLE: Configure Socks5 Proxy
DESCRIPTION: Configures a socks5 proxy for both synchronous and asynchronous clients by passing proxy arguments via http_options. Requires httpx[socks] to be installed.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
http_options = types.HttpOptions(
    client_args={'proxy': 'socks5://user:pass@host:port'},
    async_client_args={'proxy': 'socks5://user:pass@host:port'},
)

client=genai.Client(..., http_options=http_options)
```

----------------------------------------

TITLE: Dataset Statistics
DESCRIPTION: Provides statistics related to dataset tuning and user data. Includes counts for characters, examples, and token distributions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DatasetStats:
  total_tuning_character_count: int
  tuning_dataset_example_count: int
  tuning_step_count: int
  user_dataset_examples: int
  user_input_token_distribution: dict
  user_message_per_example_distribution: dict
  user_output_token_distribution: dict

DatasetStatsDict:
  total_billable_character_count: int
  total_tuning_character_count: int
  tuning_dataset_example_count: int
  tuning_step_count: int
  user_dataset_examples: int
  user_input_token_distribution: dict
  user_message_per_example_distribution: dict
  user_output_token_distribution: dict
```

----------------------------------------

TITLE: Count Tokens
DESCRIPTION: Provides an example of how to count tokens for a given text input using the `client.models.count_tokens` method. This is useful for understanding token usage and costs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)

```

----------------------------------------

TITLE: LiveMusicClientMessage Attributes
DESCRIPTION: Details the attributes for LiveMusicClientMessage, including 'client_content', 'music_generation_config', 'playback_control', and 'setup'. These are used for controlling live music playback and generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicClientMessage:
  client_content: The content provided by the client for music generation.
  music_generation_config: Configuration for the music generation process.
  playback_control: Controls for playback of the generated music.
  setup: Setup parameters for the live music client.
```

----------------------------------------

TITLE: LiveClientSetup Configuration Options
DESCRIPTION: Details the configuration parameters available for LiveClientSetup, including context window compression, generation configuration, audio transcription settings, model selection, proactivity, session resumption, system instructions, and tools.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientSetup:
  context_window_compression: Configures context window compression.
  generation_config: Specifies generation configuration parameters.
  input_audio_transcription: Settings for input audio transcription.
  model: The model to be used for generation.
  output_audio_transcription: Settings for output audio transcription.
  proactivity: Controls the proactivity of the client.
  session_resumption: Configuration for session resumption.
  system_instruction: Sets the system instruction for the model.
  tools: Defines the tools available for the model.
```

----------------------------------------

TITLE: Interval Type Documentation
DESCRIPTION: Documentation for the Interval type, which represents a time interval with start and end times.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Interval:
  Represents a time interval.
  Attributes:
    start_time: The start time of the interval.
    end_time: The end time of the interval.
```

----------------------------------------

TITLE: LiveMusicClientMessageDict Attributes
DESCRIPTION: Details the attributes for LiveMusicClientMessageDict, including 'client_content', 'music_generation_config', 'playback_control', and 'setup'. These are used for controlling live music playback and generation, similar to LiveMusicClientMessage.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicClientMessageDict:
  client_content: The content provided by the client for music generation.
  music_generation_config: Configuration for the music generation process.
  playback_control: Controls for playback of the generated music.
  setup: Setup parameters for the live music client.
```

----------------------------------------

TITLE: LiveClientSetupDict Configuration Options
DESCRIPTION: Details the configuration parameters available for LiveClientSetupDict, mirroring LiveClientSetup for dictionary-based configurations, including context window compression, generation configuration, audio transcription settings, model selection, proactivity, session resumption, system instructions, and tools.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientSetupDict:
  context_window_compression: Configures context window compression.
  generation_config: Specifies generation configuration parameters.
  input_audio_transcription: Settings for input audio transcription.
  model: The model to be used for generation.
  output_audio_transcription: Settings for output audio transcription.
  proactivity: Controls the proactivity of the client.
  session_resumption: Configuration for session resumption.
  system_instruction: Sets the system instruction for the model.
  tools: Defines the tools available for the model.
```

----------------------------------------

TITLE: SupervisedTuningDataStats Attributes
DESCRIPTION: Details the attributes of the `SupervisedTuningDataStats` class, which provides statistics for supervised tuning datasets. This includes counts of examples, characters, and token distributions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDataStats:
  total_truncated_example_count: int
    The total number of examples that were truncated.
  total_tuning_character_count: int
    The total number of characters in the tuning dataset.
  truncated_example_indices: list[int]
    A list of indices for the truncated examples.
  tuning_dataset_example_count: int
    The total number of examples in the tuning dataset.
  tuning_step_count: int
    The number of tuning steps performed.
  user_dataset_examples: list[dict]
    A list of examples provided by the user.
  user_input_token_distribution: dict
    The distribution of token counts for user inputs.
  user_message_per_example_distribution: dict
    The distribution of user messages per example.
  user_output_token_distribution: dict
    The distribution of token counts for user outputs.
```

----------------------------------------

TITLE: CreateTuningJob Configuration and Parameters
DESCRIPTION: Details the configuration and parameters for creating tuning jobs. Covers various settings such as adapter size, learning rate, epoch count, and dataset specifications.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateTuningJobConfig:
  adapter_size: int
    The size of the adapter.
  batch_size: int
    The batch size for training.
  description: str
    A description for the tuning job.
  epoch_count: int
    The number of epochs for training.
  export_last_checkpoint_only: bool
    Whether to export only the last checkpoint.
  http_options: dict
    Additional HTTP options for the request.
  learning_rate: float
    The learning rate for training.
  learning_rate_multiplier: float
    A multiplier for the learning rate.
  tuned_model_display_name: str
    The display name for the tuned model.
  validation_dataset: str
    The dataset to use for validation.

CreateTuningJobConfigDict:
  adapter_size: int
    The size of the adapter.
  batch_size: int
    The batch size for training.
  description: str
    A description for the tuning job.
  epoch_count: int
    The number of epochs for training.
  export_last_checkpoint_only: bool
    Whether to export only the last checkpoint.
  http_options: dict
    Additional HTTP options for the request.
  learning_rate: float
    The learning rate for training.
  learning_rate_multiplier: float
    A multiplier for the learning rate.
  tuned_model_display_name: str
    The display name for the tuned model.
  validation_dataset: str
    The dataset to use for validation.

CreateTuningJobParameters:
  base_model: str
    The base model to use for tuning.
```

----------------------------------------

TITLE: Live Music Server Message Types
DESCRIPTION: Defines types for server messages in live music generation, including filtered prompts and setup status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicServerMessage, LiveMusicServerMessageDict

# Example usage for LiveMusicServerMessage:
server_message = LiveMusicServerMessage(
    filtered_prompt='prompt',
    server_content=LiveMusicServerContent(audio_chunks=[b'audio_data']),
    setup_complete=True
)

# Example usage for LiveMusicServerMessageDict:
server_message_dict = LiveMusicServerMessageDict(
    filtered_prompt='prompt',
    server_content={'audio_chunks': [b'audio_data']},
    setup_complete=True
)

```

----------------------------------------

TITLE: Live Send Realtime Input Parameters Types
DESCRIPTION: Defines parameter types for sending real-time input in live music scenarios, including activity start and end times.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveSendRealtimeInputParameters
import datetime

# Example usage:
realtime_input_params = LiveSendRealtimeInputParameters(
    activity_start=datetime.datetime.now(),
    activity_end=datetime.datetime.now() + datetime.timedelta(seconds=10)
)

```

----------------------------------------

TITLE: Get Cached Content
DESCRIPTION: Retrieves existing cached content using its name. This is useful for accessing previously created cached content for further use.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
cached_content = client.caches.get(name=cached_content.name)
```

----------------------------------------

TITLE: Initialize Tuning Dataset
DESCRIPTION: Prepares a tuning dataset for supervised fine-tuning (SFT) by specifying the Google Cloud Storage (GCS) URI of the dataset. This is a prerequisite for the `tune` operation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

model = 'gemini-2.0-flash-001'
training_dataset = types.TuningDataset(
    gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
)
```

----------------------------------------

TITLE: Get Batch Job Status
DESCRIPTION: Retrieves the status of a previously created batch job by its name. This is useful for monitoring the progress of a batch operation.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
# Get a job by name
job = client.batches.get(name=job.name)

job.state
```

----------------------------------------

TITLE: Get File Information
DESCRIPTION: Shows how to retrieve information about an uploaded file using its name. This is useful for verifying upload status or accessing file metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)
```

----------------------------------------

TITLE: Create Client using Environment Variables
DESCRIPTION: Creates a client instance by automatically detecting and using configured environment variables.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai

client = genai.Client()
```

----------------------------------------

TITLE: SupervisedTuningDataStatsDict Attributes
DESCRIPTION: Details the attributes of the `SupervisedTuningDataStatsDict` type, which is a dictionary representation of supervised tuning data statistics. It includes billable counts and reasons for dropped examples.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDataStatsDict:
  dropped_example_reasons: list[str]
    Reasons why examples were dropped during tuning.
  total_billable_character_count: int
    The total number of billable characters.
  total_billable_token_count: int
    The total number of billable tokens.
  total_truncated_example_count: int
    The total number of examples that were truncated.
  total_tuning_character_count: int
    The total number of characters in the tuning dataset.
  truncated_example_indices: list[int]
    A list of indices for the truncated examples.
  tuning_dataset_example_count: int
    The total number of examples in the tuning dataset.
  tuning_step_count: int
    The number of tuning steps performed.
  user_dataset_examples: list[dict]
    A list of examples provided by the user.
  user_input_token_distribution: dict
    The distribution of token counts for user inputs.
  user_message_per_example_distribution: dict
    The distribution of user messages per example.
  user_output_token_distribution: dict
    The distribution of token counts for user outputs.
```

----------------------------------------

TITLE: Download File for Upload
DESCRIPTION: Shows how to download a file from a given URL using `wget`. This file can then be uploaded to the service for content generation.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: console
CODE:
```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

----------------------------------------

TITLE: Google Generative AI Files API
DESCRIPTION: Manages file operations, including uploading, downloading, deleting, getting, and listing files. Supports both asynchronous and synchronous methods.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.files.AsyncFiles:
  delete(file_id: str): Deletes a file by its ID asynchronously.
  download(file_id: str, destination: str):
    Downloads a file by its ID to a specified destination asynchronously.
  get(file_id: str): Retrieves file metadata by its ID asynchronously.
  list(): Lists all files asynchronously.
  upload(file_path: str, display_name: str):
    Uploads a file from a given path asynchronously.

genai.files.Files:
  delete(file_id: str): Deletes a file by its ID.
  download(file_id: str, destination: str):
    Downloads a file by its ID to a specified destination.
  get(file_id: str): Retrieves file metadata by its ID.
  list(): Lists all files.
  upload(file_path: str, display_name: str):
    Uploads a file from a given path.
```

----------------------------------------

TITLE: Provide Content as a types.Content Instance
DESCRIPTION: Shows how to provide content as a single types.Content instance. The SDK wraps this instance in a list.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.generativeai import types

contents = types.Content(
    role='user',
    parts=[types.Part.from_text(text='Why is the sky blue?')]
)

# SDK converts this to:
# [
# types.Content(
#     role='user',
#     parts=[types.Part.from_text(text='Why is the sky blue?')]
# )
# ]
```

----------------------------------------

TITLE: Segment Properties
DESCRIPTION: Details the properties available for the Segment type in the Python GenAI library. These properties define segments of text with their start and end indices.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Segment:
  end_index: The ending index of the segment.
  part_index: The index of the part within a larger sequence.
  start_index: The starting index of the segment.
  text: The text content of the segment.
```

----------------------------------------

TITLE: Models Methods
DESCRIPTION: Provides synchronous operations for interacting with generative AI models, including token computation, content generation (text, images, video), content embedding, and model management (get, list, delete, update).

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Models:
  compute_tokens(content):
    Computes tokens for the given content.
  count_tokens(content):
    Counts the tokens in the provided content.
  delete(model_name):
    Deletes a specified model.
  edit_image(prompt, image):
    Edits an image based on a prompt.
  embed_content(content):
    Embeds the given content into a vector representation.
  generate_content(prompt, **kwargs):
    Generates content based on the provided prompt and optional arguments.
  generate_content_stream(prompt, **kwargs):
    Generates content as a stream.
  generate_images(prompt, **kwargs):
    Generates images based on the prompt.
  generate_videos(prompt, **kwargs):
    Generates videos based on the prompt.
  get(model_name):
    Retrieves a specific model by its name.
  list():
    Lists available models.
  recontext_image(prompt, image):
    Re-contextualizes an image with a given prompt.
  update(model_name, **kwargs):
    Updates a specified model with new parameters.
  upscale_image(image):
    Upscales the provided image.
```

----------------------------------------

TITLE: Google Generative AI Caches API
DESCRIPTION: Provides asynchronous and synchronous methods for managing caches. Supports creating, deleting, getting, listing, and updating cache entries.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.caches.AsyncCaches:
  create(): Creates a new cache entry asynchronously.
  delete(cache_id: str): Deletes a cache entry by its ID asynchronously.
  get(cache_id: str): Retrieves a cache entry by its ID asynchronously.
  list(): Lists all cache entries asynchronously.
  update(cache_id: str, data: dict): Updates a cache entry by its ID asynchronously.

genai.caches.Caches:
  create(): Creates a new cache entry.
  delete(cache_id: str): Deletes a cache entry by its ID.
  get(cache_id: str): Retrieves a cache entry by its ID.
  list(): Lists all cache entries.
  update(cache_id: str, data: dict): Updates a cache entry by its ID.
```

----------------------------------------

TITLE: Batch Prediction Request Format (JSONL)
DESCRIPTION: Example format for a JSONL file used as a source for batch prediction jobs. Each line represents a request with an optional key and the request payload, including content and generation configuration.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: json
CODE:
```
{"key":"request_1", "request": {"contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}], "generation_config": {"response_modalities": ["TEXT"]}}}
{"key":"request_2", "request": {"contents": [{"parts": [{"text": "Explain how Crypto works in a few words"}]}]}}
```

----------------------------------------

TITLE: Create Client using Environment Variables
DESCRIPTION: Creates a genai client instance, automatically using environment variables for configuration. This is a convenient way to initialize the client without explicit parameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from googleimport genai

client = genai.Client()
```

----------------------------------------

TITLE: Get Tuned Model Details
DESCRIPTION: Retrieves detailed information about a specific tuned model using its identifier. This allows inspection of the model's configuration and status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)
```

----------------------------------------

TITLE: Provide a list of function call parts
DESCRIPTION: Illustrates creating multiple function call parts and how the SDK groups them into a single `ModelContent`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
fromgoogle.genai import types

contents = [
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
    ),
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
    ),
]

# SDK representation:
# [
# types.ModelContent(
#     parts=[
#     types.Part.from_function_call(
#         name='get_weather_by_location',
#         args={'location': 'Boston'}
#     ),
#     types.Part.from_function_call(
#         name='get_weather_by_location',
#         args={'location': 'New York'}
#     )
#     ]
# )
# ]
```

----------------------------------------

TITLE: AsyncModels Methods
DESCRIPTION: Enables asynchronous operations for interacting with generative AI models, including token computation, content generation (text, images, video), content embedding, and model management (get, list, delete, update).

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AsyncModels:
  compute_tokens(content):
    Computes tokens for the given content.
  count_tokens(content):
    Counts the tokens in the provided content.
  delete(model_name):
    Deletes a specified model.
  edit_image(prompt, image):
    Edits an image based on a prompt.
  embed_content(content):
    Embeds the given content into a vector representation.
  generate_content(prompt, **kwargs):
    Generates content based on the provided prompt and optional arguments.
  generate_content_stream(prompt, **kwargs):
    Generates content as a stream.
  generate_images(prompt, **kwargs):
    Generates images based on the prompt.
  generate_videos(prompt, **kwargs):
    Generates videos based on the prompt.
  get(model_name):
    Retrieves a specific model by its name.
  list():
    Lists available models.
  recontext_image(prompt, image):
    Re-contextualizes an image with a given prompt.
  update(model_name, **kwargs):
    Updates a specified model with new parameters.
  upscale_image(image):
    Upscales the provided image.
```

----------------------------------------

TITLE: TuningJob Attributes
DESCRIPTION: Provides access to various attributes of a TuningJob, including its service account, start time, state, tuning specifications, and more. These attributes offer detailed information about the progress and configuration of a model tuning job.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningJob:
  service_account: The service account used for the tuning job.
  start_time: The timestamp when the tuning job started.
  state: The current state of the tuning job (e.g., RUNNING, SUCCEEDED, FAILED).
  supervised_tuning_spec: Configuration for supervised tuning.
  tuned_model: The name of the tuned model.
  tuned_model_display_name: A user-friendly display name for the tuned model.
  tuning_data_stats: Statistics related to the tuning data used.
  update_time: The timestamp when the tuning job was last updated.
  veo_tuning_spec: Configuration for VEO (Video/Audio) tuning.
  has_ended: Boolean indicating if the tuning job has completed.
  has_succeeded: Boolean indicating if the tuning job completed successfully.
```

----------------------------------------

TITLE: Get Tuned Model Details
DESCRIPTION: Retrieves detailed information about a specific tuned model using its model identifier. The output includes the model's configuration and metadata.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)
```

----------------------------------------

TITLE: Generate Content with System Instruction and Config
DESCRIPTION: Shows how to configure content generation with a system instruction and various parameters like max output tokens and temperature. This allows for fine-tuning the model's behavior and response length.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Provide a function call part
DESCRIPTION: Shows how to create a function call part using `from_function_call` and how the SDK represents it as a `ModelContent`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
fromgoogle.genai import types

contents = types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'Boston'}
)

# SDK representation:
# [
# types.ModelContent(
#     parts=[
#     types.Part.from_function_call(
#         name='get_weather_by_location',
#         args={'location': 'Boston'}
#     )
#     ]
# )
# ]
```

----------------------------------------

TITLE: ContextWindowCompressionConfig API
DESCRIPTION: API documentation for ContextWindowCompressionConfig, detailing sliding window and trigger token configurations, along with their dictionary representations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ContextWindowCompressionConfig:
  sliding_window: int
    The size of the sliding window for compression.
  trigger_tokens: int
    The number of tokens that trigger compression.

ContextWindowCompressionConfigDict:
  sliding_window: int
    The size of the sliding window for compression.
  trigger_tokens: int
    The number of tokens that trigger compression.
```

----------------------------------------

TITLE: Create Vertex AI Client
DESCRIPTION: Creates a client instance for the Vertex AI API, specifying project ID and location.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)
```

----------------------------------------

TITLE: File Upload using gsutil
DESCRIPTION: Provides the command-line instructions to copy files from Google Cloud Storage to the local directory, preparing them for upload to the Generative AI API.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: bash
CODE:
```
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .
```

----------------------------------------

TITLE: VideoMetadata Types
DESCRIPTION: Defines types for video metadata, including start and end offsets, and frames per second (fps). These types are used to structure video-related information within the SDK.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VideoMetadata:
  start_offset: Offset in milliseconds from the beginning of the video.
  end_offset: Offset in milliseconds from the end of the video.

VideoMetadataDict:
  start_offset: Offset in milliseconds from the beginning of the video.
  end_offset: Offset in milliseconds from the end of the video.
  fps: Frames per second of the video.
```

----------------------------------------

TITLE: Disabling Automatic Function Calling
DESCRIPTION: Provides an example of how to disable the automatic function calling feature when passing Python functions as tools. This is useful when you want to manually handle function calls.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='This is a test.',
    # Additional config to disable automatic calling would go here if available
)
```

----------------------------------------

TITLE: Upload File for Content Generation
DESCRIPTION: Downloads a file using wget and then uploads it to be used in content generation. The file content is passed as part of the contents argument.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: bash
CODE:
```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

----------------------------------------

TITLE: Get Tuning Job Status
DESCRIPTION: Retrieves the current status of a tuning job. It includes a loop that periodically checks the job status until it reaches a completed state (SUCCEEDED, FAILED, or CANCELLED).

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import time

completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
    ]
)

while tuning_job.state not in completed_states:
    print(tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)
```

----------------------------------------

TITLE: Get Tuning Job Status
DESCRIPTION: Retrieves the status of a previously created tuning job using its unique name. This is often used in a loop to monitor the job's progress until completion.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
tuning_job = client.tunings.get(name=tuning_job.name)
print(tuning_job)
```

----------------------------------------

TITLE: Create Gemini Developer API Client
DESCRIPTION: Creates a client instance for the Gemini Developer API using an API key.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')
```

----------------------------------------

TITLE: Citation Type Attributes
DESCRIPTION: Details the attributes for the Citation type, which represents a citation within the generated text. This includes start and end indices, license, publication date, title, and URI.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Citation:
  end_index: The end index of the cited text within the content.
  license: The license associated with the citation.
  publication_date: The publication date of the cited source.
  start_index: The start index of the cited text within the content.
  title: The title of the cited source.
  uri: The Uniform Resource Identifier (URI) of the cited source.
```

----------------------------------------

TITLE: API Documentation for CreateAuthTokenParameters
DESCRIPTION: Details the configuration parameters for creating an authentication token. This includes settings related to the model's configuration.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateAuthTokenParameters:
  config: dict
    Configuration for the authentication token. This may include model-specific settings.
```

----------------------------------------

TITLE: Provide a list of string parts
DESCRIPTION: Demonstrates how to provide a list of text parts, which the SDK converts into a single content object with a user role.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
fromgoogle.genai import types

contents = [
    types.UserContent(
        parts=[
            types.Part.from_text(text='Why is the sky blue?'),
            types.Part.from_text(text='Why is the cloud white?'),
        ]
    )
]
```

----------------------------------------

TITLE: BatchJobDict Attributes
DESCRIPTION: Defines the attributes for the BatchJobDict data structure, representing a batch job. Includes creation time, destination, display name, end time, error status, model used, job name, source, start time, state, and update time.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchJobDict:
  create_time: Timestamp of job creation.
  dest: Destination for the batch job output.
  display_name: User-friendly name for the batch job.
  end_time: Timestamp when the job finished.
  error: Details about any errors encountered during job execution.
  model: The model used for the batch job.
  name: Unique identifier for the batch job.
  src: Source of the batch job data.
  start_time: Timestamp when the job started.
  state: Current state of the batch job (e.g., running, completed, failed).
  update_time: Timestamp of the last update to the job.
```

----------------------------------------

TITLE: LiveConnectConfig Parameters
DESCRIPTION: Details the various parameters available for configuring LiveConnect sessions, including real-time input, response modalities, generation settings, and system instructions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectConfig:
  realtime_input_config: Configuration for real-time input.
  response_modalities: Specifies the modalities for the response.
  seed: Seed for reproducibility.
  session_resumption: Configuration for session resumption.
  speech_config: Configuration for speech processing.
  system_instruction: System-level instructions for the model.
  temperature: Controls the randomness of predictions.
  tools: List of tools the model can use.
  top_k: Top-K sampling parameter.
  top_p: Top-P (nucleus) sampling parameter.
```

----------------------------------------

TITLE: LiveConnectConfigDict Parameters
DESCRIPTION: Details the dictionary-based configuration options for LiveConnect, mirroring LiveConnectConfig but intended for dictionary usage.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectConfigDict:
  context_window_compression: Enables context window compression.
  enable_affective_dialog: Enables affective dialog features.
  generation_config: Configuration for response generation.
  http_options: HTTP-related options.
  input_audio_transcription: Configuration for input audio transcription.
  max_output_tokens: Maximum number of tokens in the output.
  media_resolution: Resolution for media processing.
  output_audio_transcription: Configuration for output audio transcription.
  proactivity: Controls model proactivity.
  realtime_input_config: Configuration for real-time input.
  response_modalities: Specifies the modalities for the response.
  seed: Seed for reproducibility.
  session_resumption: Configuration for session resumption.
  speech_config: Configuration for speech processing.
  system_instruction: System-level instructions for the model.
  temperature: Controls the randomness of predictions.
  tools: List of tools the model can use.
  top_k: Top-K sampling parameter.
  top_p: Top-P (nucleus) sampling parameter.
```

----------------------------------------

TITLE: Get and Monitor Batch Prediction Job State
DESCRIPTION: Retrieves a batch prediction job by its name and monitors its state until it reaches a completed status (SUCCEEDED, FAILED, CANCELLED, or PAUSED). Includes a loop with a delay to periodically check the job status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
# Get a job by name
job = client.batches.get(name=job.name)

job.state

completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_PAUSED',
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)

job
```

----------------------------------------

TITLE: Mix types in contents and configuration
DESCRIPTION: Demonstrates mixing different content types and configuring generation parameters like `system_instruction`, `max_output_tokens`, and `temperature`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
fromgoogle.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: API Documentation for CreateFileConfig
DESCRIPTION: Defines the configuration for creating a file, including optional HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateFileConfig:
  http_options: dict
    Optional HTTP client options for the file creation request.
```

----------------------------------------

TITLE: Create Vertex AI Client
DESCRIPTION: Creates a client instance for the Vertex AI API, specifying the project ID and location. This client is used for interacting with Google Cloud's Vertex AI services.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from googleimport genai

# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)
```

----------------------------------------

TITLE: LiveConnectConfig Configuration Options
DESCRIPTION: Details the configuration parameters for LiveConnectConfig, covering context window compression, affective dialog enablement, generation configuration, HTTP options, audio transcription settings, maximum output tokens, media resolution, proactivity, and output audio transcription.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectConfig:
  context_window_compression: Configures context window compression.
  enable_affective_dialog: Enables or disables affective dialog features.
  generation_config: Specifies generation configuration parameters.
  http_options: Configuration for HTTP requests.
  input_audio_transcription: Settings for input audio transcription.
  max_output_tokens: Sets the maximum number of output tokens.
  media_resolution: Specifies the resolution for media processing.
  output_audio_transcription: Settings for output audio transcription.
  proactivity: Controls the proactivity of the connection.
```

----------------------------------------

TITLE: Generate Image with Imagen
DESCRIPTION: Demonstrates image generation using the Imagen model via `client.models.generate_images`. This includes specifying the prompt, model, and configuration options like the number of images and output format. Support is behind an allowlist.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()

```

----------------------------------------

TITLE: PrebuiltVoiceConfigDict Attributes
DESCRIPTION: A dictionary representation of PrebuiltVoiceConfig, used for specifying pre-built voice configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PrebuiltVoiceConfigDict:
  voice_name: The name of the pre-built voice.
```

----------------------------------------

TITLE: Provide a list of function call parts
DESCRIPTION: Shows how to provide multiple function calls as content. The SDK groups these into a single `types.ModelContent` object.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

contents = [
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'Boston'}
    ),
    types.Part.from_function_call(
        name='get_weather_by_location',
        args={'location': 'New York'}
    ),
]

# The SDK converts a list of function call parts to the a content with a `model` role:
# 
# [
# types.ModelContent(
#     parts=[
#     types.Part.from_function_call(
#         name='get_weather_by_location',
#         args={'location': 'Boston'}
#     ),
#     types.Part.from_function_call(
#         name='get_weather_by_location',
#         args={'location': 'New York'}
#     )
#     ]
# )
# ]
# 
# Where a `types.ModelContent` is a subclass of `types.Content`, the
# `role` field in `types.ModelContent` is fixed to be `model`.
```

----------------------------------------

TITLE: Python GenAI Client API
DESCRIPTION: Documentation for the `genai.client` module, detailing the `AsyncClient` and `Client` classes and their attributes and methods. This includes authentication tokens, batch operations, file management, chat interactions, model access, and tuning capabilities.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AsyncClient:
  Attributes:
    auth_tokens: Authentication tokens for the client.
    batches: Interface for managing batches.
    caches: Interface for managing caches.
    chats: Interface for managing chat sessions.
    files: Interface for managing files.
    live: Interface for live operations.
    models: Interface for accessing models.
    operations: Interface for managing operations.
    tunings: Interface for managing model tunings.

Client:
  Attributes:
    api_key: The API key for authentication.
    vertexai: Vertex AI specific configurations.
    credentials: User credentials.
    project: The Google Cloud project ID.
    location: The Google Cloud location.
    debug_config: Configuration for debugging.
    http_options: HTTP request options.
    aio: Asynchronous client instance.
    auth_tokens: Authentication tokens for the client.
    batches: Interface for managing batches.
    caches: Interface for managing caches.
    chats: Interface for managing chat sessions.
    files: Interface for managing files.
    models: Interface for accessing models.
    operations: Interface for managing operations.
    tunings: Interface for managing model tunings.

DebugConfig:
  Attributes:
    client_mode: The client mode for debugging.
    replay_id: The replay ID for debugging.
    replays_directory: The directory for storing replays.
```

----------------------------------------

TITLE: Configure Async Client with Aiohttp
DESCRIPTION: Sets up HttpOptions to use aiohttp for faster asynchronous client performance, allowing custom client arguments.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

http_options = types.HttpOptions(
    async_client_args={'cookies': ..., 'ssl': ...},
)

client=genai.Client(..., http_options=http_options)
```

----------------------------------------

TITLE: GenerateVideosConfigDict Parameters
DESCRIPTION: Defines the configuration parameters for generating videos, including negative prompts, output URIs, resolution, and safety settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateVideosConfigDict:
  negative_prompt: str | None
    A prompt that the model should avoid generating.
  number_of_videos: int | None
    The number of videos to generate.
  output_gcs_uri: str | None
    The Cloud Storage URI where the generated videos will be saved.
  person_generation: bool | None
    Whether to enable person generation.
  pubsub_topic: str | None
    The Pub/Sub topic to publish notifications to.
  resolution: str | None
    The desired resolution of the generated videos (e.g., '1080p').
  seed: int | None
    A seed for the random number generator to ensure reproducibility.
```

----------------------------------------

TITLE: Provide a `list[types.Content]`
DESCRIPTION: Demonstrates the canonical way to provide content to the `generate_content` method using a list of `types.Content` objects. The SDK performs no conversion in this case.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Content(
role='user',
parts=[types.Part.from_text(text='Why is the sky blue?')]
)

# SDK converts this to
# [
# types.Content(
#     role='user',
#     parts=[types.Part.from_text(text='Why is the sky blue?')]
# )
# ]
```

----------------------------------------

TITLE: API Documentation for CreateBatchJobConfig
DESCRIPTION: Defines the configuration for creating a batch job, including destination, display name, and HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateBatchJobConfig:
  dest: str
    The destination for the batch job output.
  display_name: str
    A user-friendly name for the batch job.
  http_options: dict
    Optional HTTP client options for the batch job request.
```

----------------------------------------

TITLE: GenAI Client Module Documentation
DESCRIPTION: Documentation for the genai.client module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.client
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: Create Gemini Developer API Client
DESCRIPTION: Creates a client instance for the Gemini Developer API by providing an API key. This client is used to interact with the Gemini Developer API services.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from googleimport genai

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')
```

----------------------------------------

TITLE: Provide a list of non-function call parts
DESCRIPTION: Shows how to combine text and URI parts in a list, which the SDK groups into a single `UserContent`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
fromgoogle.genai import types

contents = [
    types.Part.from_text('What is this image about?'),
    types.Part.from_uri(
        file_uri='gs://generativeai-downloads/images/scones.jpg',
        mime_type='image/jpeg',
    )
]

# SDK representation:
# [
# types.UserContent(
#     parts=[
#     types.Part.from_text('What is this image about?'),
#     types.Part.from_uri(
#         file_uri: 'gs://generativeai-downloads/images/scones.jpg',
#         mime_type: 'image/jpeg',
#     )
#     ]
# )
# ]
```

----------------------------------------

TITLE: CreateFileConfig and CreateFileResponse Types
DESCRIPTION: Defines the configuration and response types for creating files. Includes options for HTTP response handling and SDK HTTP response retrieval.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateFileConfig:
  should_return_http_response: bool
    Whether to return the HTTP response object.

CreateFileConfigDict:
  http_options: dict
    Additional HTTP options for the request.
  should_return_http_response: bool
    Whether to return the HTTP response object.

CreateFileResponse:
  sdk_http_response: google.api_core.http_response.HttpResponse
    The SDK HTTP response object.

CreateFileResponseDict:
  sdk_http_response: google.api_core.http_response.HttpResponse
    The SDK HTTP response object.
```

----------------------------------------

TITLE: Provide Content as a String
DESCRIPTION: Illustrates providing content as a simple string. The SDK assumes this is a text part and converts it into a types.UserContent instance within a list.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
contents='Why is the sky blue?'

# The SDK will assume this is a text part, and it converts this into the following:
# [
# types.UserContent(
#     parts=[
#     types.Part.from_text(text='Why is the sky blue?')
#     ]
# )
# ]
# Where a types.UserContent is a subclass of types.Content, it sets the role field to be user.
```

----------------------------------------

TITLE: PrebuiltVoiceConfig Attributes
DESCRIPTION: Configuration for pre-built voice settings, specifying the name of the voice to be used.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PrebuiltVoiceConfig:
  voice_name: The name of the pre-built voice.
```

----------------------------------------

TITLE: List All Available Models
DESCRIPTION: Demonstrates how to iterate through all available models using the `client.models.list()` method. This is useful for discovering which models are available for use.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
for model in client.models.list():
    print(model)
```

----------------------------------------

TITLE: DownloadFileConfig and DownloadFileConfigDict Configuration
DESCRIPTION: Specifies configuration for downloading files, including HTTP options. Both DownloadFileConfig and DownloadFileConfigDict are covered.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DownloadFileConfig:
  http_options: HTTP options for the download.

DownloadFileConfigDict:
  http_options: HTTP options for the download.
```

----------------------------------------

TITLE: VertexAISearchDict API Documentation
DESCRIPTION: Documentation for VertexAISearchDict, a dictionary representation of Vertex AI Search configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexAISearchDict:
  data_store_specs: Specifications for data stores.
  datastore: The datastore to use.
  engine: The search engine.
  filter: Filter criteria for search results.
```

----------------------------------------

TITLE: List Batch Jobs with Pager
DESCRIPTION: Demonstrates using the pager object to navigate through lists of batch jobs. It shows how to access page size, individual jobs, and move to the next page.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: ListTuningJobsConfig and ListTuningJobsConfigDict API Documentation
DESCRIPTION: Defines the configuration options for listing tuning jobs, including filtering, HTTP options, page size, and page token. Also includes the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListTuningJobsConfig:
  filter: str | None
    Optional. A filter to apply to the list of tuning jobs.
  http_options: google.api_core.retry.RetryOptions | dict | None
    Optional. HTTP options for the request.
  page_size: int | None
    Optional. The maximum number of tuning jobs to return in a single page.
  page_token: str | None
    Optional. The page token to use for pagination.

ListTuningJobsConfigDict:
  filter: str | None
    Optional. A filter to apply to the list of tuning jobs.
  http_options: google.api_core.retry.RetryOptions | dict | None
    Optional. HTTP options for the request.
  page_size: int | None
    Optional. The maximum number of tuning jobs to return in a single page.
  page_token: str | None
    Optional. The page token to use for pagination.
```

----------------------------------------

TITLE: Generate Videos
DESCRIPTION: Demonstrates how to generate videos using a specified model and prompt. It includes polling the operation status and displaying the generated video.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
operation = client.models.generate_videos(
        model='veo-2.0-generate-001',
        prompt='A neon hologram of a cat driving at top speed',
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            fps=24,
            duration_seconds=5,
            enhance_prompt=True,
        ),
    )

    # Poll operation
    while not operation.done:
        time.sleep(20)
        operation = client.operations.get(operation)

    video = operation.result.generated_videos[0].video
    video.show()
```

----------------------------------------

TITLE: Create Tuning Job Parameters
DESCRIPTION: Defines parameters for creating a tuning job, including configuration and training dataset details. Supports both object and dictionary representations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateTuningJobParameters:
  config: Configuration for the tuning job.
  training_dataset: The dataset to be used for training.

CreateTuningJobParametersDict:
  base_model: The base model for tuning.
  config: Configuration for the tuning job.
  training_dataset: The dataset to be used for training.
```

----------------------------------------

TITLE: Initialize Gemini Client
DESCRIPTION: Initializes the Gemini client for the Gemini Developer API with a specific API version.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai
from google.generativeai import types

client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)
```

----------------------------------------

TITLE: GenerationConfig Parameters
DESCRIPTION: Configuration options for controlling the generation process, such as temperature, top_k, and top_p.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfig:
  temperature: float | None
    Controls the randomness of predictions. Lower values make the model more deterministic.
  top_k: int | None
    Top-k sampling. The model considers only the top_k most likely tokens for each step.
  top_p: float | None
    Top-p (nucleus) sampling. The model considers tokens that make up the top_p probability mass.
```

----------------------------------------

TITLE: ListBatchJobsConfig and ListBatchJobsConfigDict
DESCRIPTION: Configuration options for listing batch jobs. These structures allow filtering, pagination, and specifying HTTP options for the request.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListBatchJobsConfig:
  filter: String used to filter the results. Supports various query parameters.
  http_options: Optional HTTP options for the request.
  page_size: The number of batch jobs to return per page (int).
  page_token: A token for fetching the next page of results.

ListBatchJobsConfigDict:
  filter: String used to filter the results. Supports various query parameters.
  http_options: Optional HTTP options for the request.
  page_size: The number of batch jobs to return per page (int).
  page_token: A token for fetching the next page of results.
```

----------------------------------------

TITLE: StyleReferenceImage and StyleReferenceImageDict
DESCRIPTION: Defines how to reference an image for style guidance. Includes configuration, reference ID, the image itself, reference type, and style image configuration.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
StyleReferenceImage:
  config: dict
  reference_id: str
  reference_image: bytes
  reference_type: str
  style_image_config: dict

StyleReferenceImageDict:
  config: dict
  reference_id: str
  reference_image: bytes
  reference_type: str
```

----------------------------------------

TITLE: Provide Content as a List of types.Content
DESCRIPTION: Demonstrates the canonical way to provide content to the generate_content method using a list of types.Content objects. The SDK performs no conversion in this case.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.generativeai import types

contents = types.Content(
    role='user',
    parts=[types.Part.from_text(text='Why is the sky blue?')]
)

# SDK converts this to:
# [
# types.Content(
#     role='user',
#     parts=[types.Part.from_text(text='Why is the sky blue?')]
# )
# ]
```

----------------------------------------

TITLE: Generate Videos with Veo Model
DESCRIPTION: Shows how to generate videos using the Veo model. This involves creating a generation operation, polling for its completion, and then displaying the generated video. Note that this feature is behind an allowlist.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types
import time

# Create operation
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()
```

----------------------------------------

TITLE: GenAI Live Module Documentation
DESCRIPTION: Documentation for the genai.live module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.live
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: Live Music Set Configuration Parameters Types
DESCRIPTION: Defines parameter types for setting music generation configurations in live music scenarios.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicSetConfigParameters, LiveMusicSetConfigParametersDict

# Example usage for LiveMusicSetConfigParameters:
config_params = LiveMusicSetConfigParameters(music_generation_config={'tempo': 120})

# Example usage for LiveMusicSetConfigParametersDict:
config_params_dict = LiveMusicSetConfigParametersDict(music_generation_config={'tempo': 120})

```

----------------------------------------

TITLE: LiveClientRealtimeInputDict Attributes
DESCRIPTION: Describes the attributes for LiveClientRealtimeInputDict, mirroring LiveClientRealtimeInput.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientRealtimeInputDict:
  activity_end: Timestamp for the end of an activity.
  activity_start: Timestamp for the start of an activity.
  audio: Audio data from the client.
  audio_stream_end: Indicates the end of an audio stream.
  media_chunks: Chunks of media data.
  text: Textual input from the client.
  video: Video data from the client.
```

----------------------------------------

TITLE: Provide a function call part
DESCRIPTION: Demonstrates how to provide a function call as a part of the content. The SDK converts this into a `types.ModelContent` with the function call part.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Part.from_function_call(
    name='get_weather_by_location',
    args={'location': 'Boston'}
)

# The SDK converts a function call part to a content with a `model` role:
# 
# [
# types.ModelContent(
#     parts=[
#     types.Part.from_function_call(
#         name='get_weather_by_location',
#         args={'location': 'Boston'}
#     )
#     ]
# )
# ]
# 
# Where a `types.ModelContent` is a subclass of `types.Content`, the
# `role` field in `types.ModelContent` is fixed to be `model`.
```

----------------------------------------

TITLE: GenerateImagesConfigDict Parameters
DESCRIPTION: Dictionary-based configuration for image generation, mirroring GenerateImagesConfig with additional options like aspect ratio and guidance scale.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateImagesConfigDict:
  add_watermark: bool
    Whether to add a watermark to the generated image.
  aspect_ratio: str
    The aspect ratio of the generated image (e.g., '16:9').
  enhance_prompt: bool
    Whether to enhance the prompt for better results.
  guidance_scale: float
    Controls how much the generation follows the prompt.
  http_options: dict
    Additional HTTP options for the request.
  image_size: str
    The desired size of the generated image (e.g., '1024x1024').
  include_rai_reason: bool
    Whether to include RAI reason in the response.
  include_safety_attributes: bool
    Whether to include safety attributes in the response.
  language: str
    The language for the generated content.
  negative_prompt: str
    A prompt that describes what to avoid in the generated image.
  number_of_images: int
    The number of images to generate.
  output_compression_quality: int
    The compression quality for the output image (0-100).
  output_gcs_uri: str
    The Google Cloud Storage URI for the output image.
  output_mime_type: str
    The MIME type for the output image (e.g., 'image/png').
  person_generation: str
    Specifies the type of person generation (e.g., 'photorealistic').
  safety_filter_level: str
    The safety filter level to apply (e.g., 'high').
  seed: int
    A seed for reproducible image generation.
```

----------------------------------------

TITLE: Set Proxy Environment Variables
DESCRIPTION: Sets environment variables for proxy and SSL certificate file to configure proxy connections for both httpx and aiohttp libraries.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: bash
CODE:
```
export HTTPS_PROXY='http://username:password@proxy_uri:port'
export SSL_CERT_FILE='client.pem'
```

----------------------------------------

TITLE: LiveMusicGenerationConfig and LiveMusicGenerationConfigDict
DESCRIPTION: Defines configuration parameters for live music generation, including BPM, brightness, density, guidance, scale, seed, temperature, and top_k. It also includes options for music generation mode and muting specific instrument tracks.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicGenerationConfig:
  bpm: int
    Beats per minute for the music.
  brightness: float
    Controls the overall brightness or clarity of the music.
  density: float
    Determines the density or complexity of the musical elements.
  guidance: float
    Guidance scale for the music generation model.
  music_generation_mode: str
    Specifies the mode for music generation (e.g., 'melody', 'harmony').
  mute_bass: bool
    If true, the bass track will be muted.
  mute_drums: bool
    If true, the drums track will be muted.
  only_bass_and_drums: bool
    If true, only bass and drums will be generated.
  scale: str
    The musical scale to use for generation (e.g., 'major', 'minor').
  seed: int
    Seed for reproducible music generation.
  temperature: float
    Controls the randomness of the music generation.
  top_k: int
    Top-k sampling parameter for music generation.

LiveMusicGenerationConfigDict:
  bpm: int
    Beats per minute for the music.
  brightness: float
    Controls the overall brightness or clarity of the music.
  density: float
    Determines the density or complexity of the musical elements.
  guidance: float
    Guidance scale for the music generation model.
  music_generation_mode: str
    Specifies the mode for music generation (e.g., 'melody', 'harmony').
  mute_bass: bool
    If true, the bass track will be muted.
  mute_drums: bool
    If true, the drums track will be muted.
  only_bass_and_drums: bool
    If true, only bass and drums will be generated.
  scale: str
    The musical scale to use for generation (e.g., 'major', 'minor').
  seed: int
    Seed for reproducible music generation.
  temperature: float
    Controls the randomness of the music generation.
  top_k: int
    Top-k sampling parameter for music generation.
```

----------------------------------------

TITLE: DynamicRetrievalConfig and DynamicRetrievalConfigDict Configuration
DESCRIPTION: Configuration for dynamic retrieval, including dynamic threshold and mode. Covers both DynamicRetrievalConfig and DynamicRetrievalConfigDict.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DynamicRetrievalConfig:
  dynamic_threshold: The dynamic threshold for retrieval.
  mode: The mode for dynamic retrieval (e.g., MODE_DYNAMIC, MODE_UNSPECIFIED).

DynamicRetrievalConfigDict:
  dynamic_threshold: The dynamic threshold for retrieval.
  mode: The mode for dynamic retrieval (e.g., MODE_DYNAMIC, MODE_UNSPECIFIED).
```

----------------------------------------

TITLE: Configure Faster Async Client with Aiohttp
DESCRIPTION: Configures the SDK to use aiohttp for faster asynchronous client operations. Additional arguments for aiohttp.ClientSession.request can be passed through http_options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
http_options = types.HttpOptions(
    async_client_args={'cookies': ..., 'ssl': ...},
)

client=genai.Client(..., http_options=http_options)
```

----------------------------------------

TITLE: ListModelsConfig and ListModelsConfigDict API Documentation
DESCRIPTION: Defines the configuration options for listing models, including filtering, HTTP options, page size, and page token. Also includes the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListModelsConfig:
  filter: str | None
    Optional. A filter to apply to the list of models.
  http_options: google.api_core.retry.RetryOptions | dict | None
    Optional. HTTP options for the request.
  page_size: int | None
    Optional. The maximum number of models to return in a single page.
  page_token: str | None
    Optional. The page token to use for pagination.
  query_base: str | None
    Optional. The base query for the request.

ListModelsConfigDict:
  filter: str | None
    Optional. A filter to apply to the list of models.
  http_options: google.api_core.retry.RetryOptions | dict | None
    Optional. HTTP options for the request.
  page_size: int | None
    Optional. The maximum number of models to return in a single page.
  page_token: str | None
    Optional. The page token to use for pagination.
  query_base: str | None
    Optional. The base query for the request.
```

----------------------------------------

TITLE: LiveClientRealtimeInput Attributes
DESCRIPTION: Details the attributes for LiveClientRealtimeInput, covering various input types like audio, video, and text.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientRealtimeInput:
  activity_end: Timestamp for the end of an activity.
  activity_start: Timestamp for the start of an activity.
  audio: Audio data from the client.
  audio_stream_end: Indicates the end of an audio stream.
  media_chunks: Chunks of media data.
  text: Textual input from the client.
  video: Video data from the client.
```

----------------------------------------

TITLE: Generate Content with Various Input Types
DESCRIPTION: Demonstrates how to structure the `contents` argument for the `generate_content` method using different input formats. This includes single strings, lists of strings, `Content` instances, and function calls.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
generate_content(contents: Union[str, list[str], types.Content, list[types.Content], dict, list[dict]])

  - Provide a list[types.Content]: `contents=[types.Content(parts=[types.TextPart('Hello')])]`
  - Provide a types.Content instance: `contents=types.Content(parts=[types.TextPart('Hello')])`
  - Provide a string: `contents='Hello'`
  - Provide a list of string: `contents=['Hello', 'World']`
  - Provide a function call part: `contents=[types.Content(parts=[types.FunctionCallPart({'name': 'my_func', 'args': {'arg1': 'value1'}})])]`
  - Provide a list of function call parts: `contents=[types.Content(parts=[types.FunctionCallPart({'name': 'my_func', 'args': {'arg1': 'value1'}}), types.FunctionCallPart({'name': 'my_func2', 'args': {'arg2': 'value2'}})])]`
  - Provide a non function call part: `contents=[types.Content(parts=[types.TextPart('Hello')])]
  - Provide a list of non function call parts: `contents=[types.Content(parts=[types.TextPart('Hello')]), types.Content(parts=[types.TextPart('World')])]
  - Mix types in contents: `contents=[types.Content(parts=[types.TextPart('Hello')]), 'World', types.Content(parts=[types.FunctionCallPart({'name': 'my_func', 'args': {'arg1': 'value1'}})])]`
```

----------------------------------------

TITLE: Provide a list of strings as content
DESCRIPTION: Explains how a list of strings is handled by the SDK when passed to the `contents` argument. Each string becomes a separate text part within a single `types.UserContent` object.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
contents=['Why is the sky blue?', 'Why is the cloud white?']

# The SDK assumes these are 2 text parts, it converts this into a single content,
# like the following:
# 
# [
# types.UserContent(
#     parts=[
#     types.Part.from_text(text='Why is the sky blue?'),
#     types.Part.from_text(text='Why is the cloud white?'),
#     ]
# )
# ]
# 
# Where a `types.UserContent` is a subclass of `types.Content`, the
# `role` field in `types.UserContent` is fixed to be `user`.
```

----------------------------------------

TITLE: StyleReferenceConfig and StyleReferenceConfigDict
DESCRIPTION: Configuration for referencing a style, allowing a description to be provided. The dictionary version offers a direct mapping.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
StyleReferenceConfig:
  style_description: str

StyleReferenceConfigDict:
  style_description: str
```

----------------------------------------

TITLE: ListTuningJobsResponse and ListTuningJobsResponseDict API Documentation
DESCRIPTION: Defines the response structure for listing tuning jobs, including the next page token, SDK HTTP response, and the list of tuning jobs. Also includes the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListTuningJobsResponse:
  next_page_token: str | None
    The page token to use for the next page of results.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response from the SDK.
  tuning_jobs: list[genai.types.TuningJob]
    The list of tuning jobs.

ListTuningJobsResponseDict:
  next_page_token: str | None
    The page token to use for the next page of results.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response from the SDK.
```

----------------------------------------

TITLE: RealtimeInputConfig and RealtimeInputConfigDict
DESCRIPTION: Defines configurations for real-time input, including activity handling and detection. These types are used to manage how the system processes real-time data streams.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RealtimeInputConfig:
  activity_handling: Specifies how to handle activities.
  automatic_activity_detection: Enables or disables automatic activity detection.
  turn_coverage: Defines the coverage for turns.

RealtimeInputConfigDict:
  activity_handling: Specifies how to handle activities.
  automatic_activity_detection: Enables or disables automatic activity detection.
  turn_coverage: Defines the coverage for turns.
```

----------------------------------------

TITLE: UploadFileConfigDict
DESCRIPTION: Dictionary-based configuration for uploading a file, mirroring UploadFileConfig.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UploadFileConfigDict:
  Dictionary-based configuration for uploading a file.
  Attributes:
    display_name: The display name for the uploaded file.
    http_options: HTTP options for the file upload.
    mime_type: The MIME type of the file.
    name: The name of the file.
```

----------------------------------------

TITLE: Generate Content with Text and URI Parts
DESCRIPTION: Demonstrates how to create content for the generative model using a combination of text and a URI pointing to an image. The SDK converts these parts into a user content object.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        types.UserContent(
            parts=[
            types.Part.from_text('What is this image about?'),
            types.Part.from_uri(
                file_uri: 'gs://generativeai-downloads/images/scones.jpg',
                mime_type: 'image/jpeg',
            )
            ]
        )
        ]
    )
    ]
)
```

----------------------------------------

TITLE: Create Tuning Job
DESCRIPTION: Initiates a supervised fine-tuning (SFT) job for a specified base model using a prepared training dataset. It allows configuration of tuning parameters like epoch count and the display name for the tuned model.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
print(tuning_job)
```

----------------------------------------

TITLE: ProactivityConfigDict Fields
DESCRIPTION: Details the fields available for ProactivityConfigDict, specifically for proactive audio settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ProactivityConfigDict:
  proactive_audio: Configuration for proactive audio.
```

----------------------------------------

TITLE: LiveClientContentDict Attributes
DESCRIPTION: Details the attributes for LiveClientContentDict, mirroring LiveClientContent with turn_complete and turns.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientContentDict:
  turn_complete: Boolean indicating if the turn is complete.
  turns: A list of turns in the conversation.
```

----------------------------------------

TITLE: Generate Content (Synchronous Streaming)
DESCRIPTION: Shows how to generate content using a streaming approach, allowing the model's output to be received in chunks as it's generated, rather than waiting for the complete response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Asynchronous Streaming Content Generation
DESCRIPTION: Demonstrates how to generate content asynchronously with streaming enabled using `client.aio.models.generate_content_stream`. This allows for real-time processing of model responses.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')

```

----------------------------------------

TITLE: Generate Videos (Python)
DESCRIPTION: Initiates video generation using the Veo model. Support for video generation is currently behind an allowlist for both Vertex AI and Gemini Developer API.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

# Create operation
```

----------------------------------------

TITLE: LogprobsResultTopCandidates and LogprobsResultTopCandidatesDict
DESCRIPTION: Holds a list of top candidates for log probability results.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LogprobsResultTopCandidates:
  candidates: List[LogprobsResultCandidate]
    A list of top candidates.

LogprobsResultTopCandidatesDict:
  candidates: List[LogprobsResultCandidateDict]
    A list of top candidates.
```

----------------------------------------

TITLE: Google Generative AI Live API
DESCRIPTION: Facilitates live interactions, including connecting to live sessions and accessing music-related functionalities asynchronously.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.live.AsyncLive:
  connect(): Establishes an asynchronous connection to a live session.
  music(): Accesses music-related functionalities asynchronously.

genai.live.AsyncSession:
  (No methods listed in the provided text, likely represents a session object)
```

----------------------------------------

TITLE: ListBatchJobsResponse and ListBatchJobsResponseDict
DESCRIPTION: Response structure for listing batch jobs. Contains a list of batch jobs, a token for the next page, and the SDK's HTTP response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListBatchJobsResponse:
  batch_jobs: A list of batch job objects.
  next_page_token: Token to retrieve the next page of results.
  sdk_http_response: The raw HTTP response from the SDK.

ListBatchJobsResponseDict:
  batch_jobs: A list of batch job objects.
  next_page_token: Token to retrieve the next page of results.
  sdk_http_response: The raw HTTP response from the SDK.
```

----------------------------------------

TITLE: Authentication Configuration Types
DESCRIPTION: Provides details on various authentication configurations supported by the GenAI library, including API key, service account, and OAuth methods.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AuthConfig:
  api_key_config: Configuration for API key authentication.
  auth_type: The type of authentication being used.
  google_service_account_config: Configuration for Google Service Account authentication.
  http_basic_auth_config: Configuration for HTTP Basic Authentication.
  oauth_config: Configuration for OAuth authentication.
  oidc_config: Configuration for OpenID Connect authentication.

AuthConfigDict:
  api_key_config: Dictionary configuration for API key authentication.
  auth_type: The type of authentication being used.
  google_service_account_config: Dictionary configuration for Google Service Account authentication.
  http_basic_auth_config: Dictionary configuration for HTTP Basic Authentication.
  oauth_config: Dictionary configuration for OAuth authentication.
  oidc_config: Dictionary configuration for OpenID Connect authentication.

AuthConfigGoogleServiceAccountConfig:
  service_account: The service account credentials.

AuthConfigGoogleServiceAccountConfigDict:
  service_account: The service account credentials as a dictionary.

AuthConfigHttpBasicAuthConfig:
  credential_secret: The secret credential for HTTP Basic Authentication.
```

----------------------------------------

TITLE: BatchJobDestinationDict Properties
DESCRIPTION: Specifies the dictionary-based configuration for batch job destinations, covering BigQuery URI, file name, output format, GCS URI, and the option to inline responses.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchJobDestinationDict:
  bigquery_uri: str
    Description: The BigQuery table URI for the output.
  file_name: str
    Description: The name of the output file.
  format: str
    Description: The format of the output file (e.g., 'jsonl', 'csv').
  gcs_uri: str
    Description: The Google Cloud Storage URI for the output.
  inlined_responses: bool
    Description: Whether to inline the responses in the output.
```

----------------------------------------

TITLE: Gemini Developer API Environment Variable Configuration
DESCRIPTION: Configures the GOOGLE_API_KEY environment variable for the Gemini Developer API. This allows the genai.Client() to automatically pick up the API key.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: bash
CODE:
```
export GOOGLE_API_KEY='your-api-key'
```

----------------------------------------

TITLE: Create Batch Job
DESCRIPTION: Creates a new batch job using the specified model and source files. This is the initial step in processing data in batches.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
batch_job = client.batches.create(
    model="gemini-2.0-flash",
    src="files/file_name",
)
```

----------------------------------------

TITLE: VertexAISearch API Documentation
DESCRIPTION: Documentation for VertexAISearch, covering data store specifications, datastore, engine, filter, and max results.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexAISearch:
  data_store_specs: Specifications for data stores.
  datastore: The datastore to use.
  engine: The search engine.
  filter: Filter criteria for search results.
  max_results: Maximum number of results to return.
```

----------------------------------------

TITLE: Generate Content Configuration
DESCRIPTION: Configuration options for generating content, including audio timestamps.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentConfig:
  audio_timestamp: Optional[datetime.datetime]
    Timestamp for audio input, if applicable.
```

----------------------------------------

TITLE: LiveServerContent and LiveServerContentDict
DESCRIPTION: Represents the content received from a live server. This includes transcription of input and output, model turn indicators, grounding metadata, and signals for generation completion or interruption. Both object and dictionary representations are available.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerContent:
  generation_complete: Indicates if the generation process is complete.
  grounding_metadata: Metadata related to grounding the model's response.
  input_transcription: Transcription of the user's input.
  interrupted: Indicates if the process was interrupted.
  model_turn: Indicates if the current content is from the model's turn.
  output_transcription: Transcription of the model's output.
  turn_complete: Indicates if the current turn is complete.
  url_context_metadata: Metadata related to URL context provided to the model.

LiveServerContentDict:
  generation_complete: Indicates if the generation process is complete (dictionary format).
  grounding_metadata: Metadata related to grounding the model's response (dictionary format).
  input_transcription: Transcription of the user's input (dictionary format).
  interrupted: Indicates if the process was interrupted (dictionary format).
  model_turn: Indicates if the current content is from the model's turn (dictionary format).
  output_transcription: Transcription of the model's output (dictionary format).
  turn_complete: Indicates if the current turn is complete (dictionary format).
  url_context_metadata: Metadata related to URL context provided to the model (dictionary format).
```

----------------------------------------

TITLE: VertexAISearchDataStoreSpecDict API Documentation
DESCRIPTION: Documentation for VertexAISearchDataStoreSpecDict, a dictionary representation of data store specifications.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexAISearchDataStoreSpecDict:
  data_store: The data store to use.
  filter: Filter criteria for the data store.
```

----------------------------------------

TITLE: Typed Configuration for Content Generation
DESCRIPTION: Demonstrates using Pydantic types for parameters in the `generate_content` method, including specifying model, content, and generation configuration like temperature, top_p, and max output tokens.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

----------------------------------------

TITLE: SupervisedHyperParameters API Documentation
DESCRIPTION: Details the supervised learning hyperparameters, such as adapter size, batch size, epoch count, and learning rate.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedHyperParameters:
  adapter_size: The size of the adapter for training.
  batch_size: The number of samples in each training batch.
  epoch_count: The total number of training epochs.
  learning_rate: The learning rate for the optimizer.
  learning_rate_multiplier: A multiplier for the learning rate.
```

----------------------------------------

TITLE: Safety Settings for Content Generation
DESCRIPTION: Illustrates how to configure safety settings for content generation, specifically blocking only high-severity hate speech by providing a `SafetySetting` object.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Set Proxy Environment Variables
DESCRIPTION: Configures HTTPS proxy and SSL certificate file paths using environment variables for network requests.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: bash
CODE:
```
export HTTPS_PROXY='http://username:password@proxy_uri:port'
export SSL_CERT_FILE='client.pem'
```

----------------------------------------

TITLE: VeoHyperParameters API Documentation
DESCRIPTION: Documentation for VeoHyperParameters, detailing epoch count, learning rate multiplier, and tuning task.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VeoHyperParameters:
  epoch_count: The number of epochs for training.
  learning_rate_multiplier: Multiplier for the learning rate.
  tuning_task: The task for tuning.
```

----------------------------------------

TITLE: Provide a non-function call part (URI)
DESCRIPTION: Demonstrates creating a part from a URI (e.g., an image) and how the SDK wraps it in a `UserContent`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
fromgoogle.genai import types

contents = types.Part.from_uri(
    file_uri='gs://generativeai-downloads/images/scones.jpg',
    mime_type='image/jpeg',
)

# SDK representation:
# [
# types.UserContent(parts=[
#     types.Part.from_uri(
#     file_uri: 'gs://generativeai-downloads/images/scones.jpg',
#     mime_type: 'image/jpeg',
#     )
# ])
# ]
```

----------------------------------------

TITLE: ListFilesResponse Fields
DESCRIPTION: Details the fields of ListFilesResponse, including the list of files, next_page_token for pagination, and sdk_http_response for HTTP communication details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListFilesResponse:
    files: list
    next_page_token: str
    sdk_http_response: google.api_core.http_response.HTTPResponse
```

----------------------------------------

TITLE: RecontextImageConfig and RecontextImageConfigDict
DESCRIPTION: Configuration options for recontextualizing images, including base steps, prompt enhancement, HTTP options, number of images, output compression, GCS URI, MIME type, person generation, safety filter level, and seed.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RecontextImageConfig:
  base_steps: The base number of steps for image generation.
  enhance_prompt: Whether to enhance the prompt.
  http_options: HTTP client options.
  number_of_images: The number of images to generate.
  output_compression_quality: The compression quality for the output image.
  output_gcs_uri: The Google Cloud Storage URI for the output image.
  output_mime_type: The MIME type of the output image.
  person_generation: Configuration for person generation.
  safety_filter_level: The safety filter level.
  seed: The seed for random number generation.

RecontextImageConfigDict:
  base_steps: The base number of steps for image generation.
  enhance_prompt: Whether to enhance the prompt.
  http_options: HTTP client options.
  number_of_images: The number of images to generate.
  output_compression_quality: The compression quality for the output image.
  output_gcs_uri: The Google Cloud Storage URI for the output image.
  output_mime_type: The MIME type of the output image.
  person_generation: Configuration for person generation.
  safety_filter_level: The safety filter level.
  seed: The seed for random number generation.
```

----------------------------------------

TITLE: GenerateImagesConfig Parameters
DESCRIPTION: Configuration options for generating images. This includes parameters to control watermarking, aspect ratio, prompt enhancement, guidance scale, HTTP options, and image size.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateImagesConfig:
  add_watermark: Boolean to add a watermark to the generated image.
  aspect_ratio: String specifying the aspect ratio of the image (e.g., '1:1', '16:9').
  enhance_prompt: Boolean to enhance the prompt for better image generation.
  guidance_scale: Float controlling how closely the image generation follows the prompt.
  http_options: Dictionary for custom HTTP request options.
  image_size: String specifying the size of the generated image (e.g., '1024x1024').
```

----------------------------------------

TITLE: Vertex AI Environment Variable Configuration
DESCRIPTION: Configures environment variables for Vertex AI: GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION. This enables the genai.Client() to use Vertex AI services.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: bash
CODE:
```
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

----------------------------------------

TITLE: HttpOptions and HttpOptionsDict
DESCRIPTION: Configuration options for HTTP requests, including API version, base URL, client arguments, headers, and retry options. HttpOptionsDict is the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HttpOptions:
  api_version: str
    The API version to use for requests.
  async_client_args: dict
    Arguments to pass to the asynchronous HTTP client.
  base_url: str
    The base URL for API requests.
  client_args: dict
    Arguments to pass to the synchronous HTTP client.
  extra_body: dict
    Additional data to include in the request body.
  headers: dict
    Custom headers to include in requests.
  retry_options: HttpRetryOptions
    Options for controlling request retry behavior.
  timeout: float
    The timeout in seconds for HTTP requests.

HttpOptionsDict:
  api_version: str
    The API version to use for requests.
  async_client_args: dict
    Arguments to pass to the asynchronous HTTP client.
  base_url: str
    The base URL for API requests.
  client_args: dict
    Arguments to pass to the synchronous HTTP client.
  extra_body: dict
    Additional data to include in the request body.
  headers: dict
    Custom headers to include in requests.
  retry_options: HttpRetryOptionsDict
    Options for controlling request retry behavior.
  timeout: float
    The timeout in seconds for HTTP requests.
```

----------------------------------------

TITLE: Set Vertex AI Environment Variables
DESCRIPTION: Sets environment variables for using the Gemini API in Vertex AI, including project and location.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: bash
CODE:
```
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
```

----------------------------------------

TITLE: Generate Content with Uploaded File
DESCRIPTION: Illustrates generating content by first uploading a file and then referencing it in the `generate_content` method. This is specific to the Gemini Developer API.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

----------------------------------------

TITLE: ListModelsResponse and ListModelsResponseDict API Documentation
DESCRIPTION: Defines the response structure for listing models, including the list of models, the next page token, and the SDK HTTP response. Also includes the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListModelsResponse:
  models: list[genai.types.Model]
    The list of models.
  next_page_token: str | None
    The page token to use for the next page of results.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response from the SDK.

ListModelsResponseDict:
  models: list[genai.types.Model]
    The list of models.
  next_page_token: str | None
    The page token to use for the next page of results.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response from the SDK.
```

----------------------------------------

TITLE: GenerateContentConfigDict Parameters
DESCRIPTION: Details the configurable parameters for the GenerateContentConfigDict object, which is a dictionary representation of content generation configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentConfigDict:
  audio_timestamp: Timestamp information for audio inputs.
  automatic_function_calling: Enables or disables automatic function calling.
  cached_content: Specifies cached content to be used for generation.
```

----------------------------------------

TITLE: CreateAuthTokenConfig and CreateAuthTokenParameters Types
DESCRIPTION: Defines the configuration and parameters for creating authentication tokens. Includes settings for expiration time, HTTP options, and session constraints.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateAuthTokenConfig:
  description: Configuration for creating an authentication token.
  fields:
    expire_time: The expiration time for the token.
    http_options: HTTP options for the request.
    live_connect_constraints: Constraints for live connections.
    lock_additional_fields: Whether to lock additional fields.
    new_session_expire_time: Expiration time for a new session.
    uses: The intended uses of the token.

CreateAuthTokenConfigDict:
  description: Dictionary representation of CreateAuthTokenConfig.
  fields:
    expire_time: The expiration time for the token.
    http_options: HTTP options for the request.
    live_connect_constraints: Constraints for live connections.
    lock_additional_fields: Whether to lock additional fields.
    new_session_expire_time: Expiration time for a new session.
    uses: The intended uses of the token.

CreateAuthTokenParameters:
  description: Parameters for creating an authentication token.
```

----------------------------------------

TITLE: VeoTuningSpecDict API Documentation
DESCRIPTION: Documentation for VeoTuningSpecDict, a dictionary representation of tuning specifications.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VeoTuningSpecDict:
  hyper_parameters: Hyperparameter configuration for tuning.
  training_dataset_uri: URI for the training dataset.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: Set Gemini Developer API Key Environment Variable
DESCRIPTION: Sets the GOOGLE_API_KEY environment variable for authenticating with the Gemini Developer API.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: bash
CODE:
```
export GOOGLE_API_KEY='your-api-key'
```

----------------------------------------

TITLE: LiveSendRealtimeInputParameters and LiveSendRealtimeInputParametersDict
DESCRIPTION: Defines the parameters for sending real-time input data during a live session. This includes audio, text, media, and video streams, as well as stream end indicators. Both object and dictionary representations are available.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveSendRealtimeInputParameters:
  audio: Audio data for the real-time input.
  audio_stream_end: Signal indicating the end of an audio stream.
  media: Media data for the real-time input.
  text: Text data for the real-time input.
  video: Video data for the real-time input.

LiveSendRealtimeInputParametersDict:
  activity_end: Signal indicating the end of an activity.
  activity_start: Signal indicating the start of an activity.
  audio: Audio data for the real-time input (dictionary format).
  audio_stream_end: Signal indicating the end of an audio stream (dictionary format).
  media: Media data for the real-time input (dictionary format).
  text: Text data for the real-time input (dictionary format).
  video: Video data for the real-time input (dictionary format).
```

----------------------------------------

TITLE: VeoTuningSpec API Documentation
DESCRIPTION: Documentation for VeoTuningSpec, including hyper parameters and dataset URIs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VeoTuningSpec:
  hyper_parameters: Hyperparameter configuration for tuning.
  training_dataset_uri: URI for the training dataset.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: Generate Content with Text
DESCRIPTION: Demonstrates how to generate content using the `generate_content` method with a text prompt. This is the most basic usage for text-based generation.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)
```

----------------------------------------

TITLE: GenAI Tunings Module Documentation
DESCRIPTION: Documentation for the genai.tunings module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.tunings
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: UploadFileConfig
DESCRIPTION: Configuration for uploading a file, including display name, HTTP options, MIME type, and name.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UploadFileConfig:
  Configuration for uploading a file.
  Attributes:
    display_name: The display name for the uploaded file.
    http_options: HTTP options for the file upload.
    mime_type: The MIME type of the file.
    name: The name of the file.
```

----------------------------------------

TITLE: Retrieval Configuration and Types
DESCRIPTION: Documentation for Retrieval and RetrievalConfig types, used for configuring retrieval mechanisms. Includes options for disabling attribution, using external APIs, and specifying Vertex AI search or RAG store configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Retrieval:
  disable_attribution: Boolean indicating if attribution should be disabled.
  external_api: Configuration for an external API retrieval.
  vertex_ai_search: Configuration for Vertex AI search retrieval.
  vertex_rag_store: Configuration for Vertex AI RAG store retrieval.

RetrievalConfig:
  language_code: The language code for retrieval configuration.
  lat_lng: Latitude and longitude coordinates for retrieval.
```

----------------------------------------

TITLE: Generate Videos
DESCRIPTION: Generates video content based on textual descriptions using the Veo model. This enables the creation of dynamic video sequences from prompts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('video-generation@001')

prompt = "A drone shot flying over a lush green forest."
response = model.generate_content(prompt)

# The response contains video data, typically a URL
# print(response.videos[0].url)
```

----------------------------------------

TITLE: VertexRagStore Attributes and Methods
DESCRIPTION: Documentation for the VertexRagStore type, including its attributes and methods related to RAG (Retrieval Augmented Generation).

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexRagStore:
  rag_corpora: List of RAG corpora to use.
  rag_resources: List of RAG resources to use.
  rag_retrieval_config: Configuration for RAG retrieval.
  similarity_top_k: The number of similar documents to retrieve.
  store_context: Whether to store context.
  vector_distance_threshold: The threshold for vector distance.
```

----------------------------------------

TITLE: GenerationConfigDict Parameters
DESCRIPTION: Details the various parameters available within the GenerationConfigDict for configuring generative model responses. This includes settings for response format, modality, schema, routing, seeding, speech, stop sequences, temperature, thinking, and token limits.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigDict:
  response_mime_type: The MIME type of the response. (e.g., 'text/plain', 'application/json')
  response_modalities: The modalities of the response. (e.g., ['text'], ['image'])
  response_schema: The schema for the response. (e.g., JSON schema)
  routing_config: Configuration for model routing.
  seed: A seed for reproducible generation.
  speech_config: Configuration for speech synthesis.
  stop_sequences: Sequences that will cause the model to stop generating.
  temperature: Controls randomness. Lower values make the output more deterministic.
  thinking_config: Configuration for the model's thinking process.
  top_k: The model samples from the top K most likely next tokens.
  top_p: The model samples from the smallest set of tokens whose cumulative probability exceeds top_p.
```

----------------------------------------

TITLE: Model Tuning with genai.tunings
DESCRIPTION: Enables asynchronous and synchronous operations for tuning generative models. This includes listing, retrieving, and initiating tuning jobs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.tunings.AsyncTunings.get(tuning_id: str)
  - Asynchronously retrieves a specific tuning job.
  - Parameters:
    - tuning_id: The unique identifier for the tuning job.

genai.tunings.AsyncTunings.list()
  - Asynchronously lists available tuning jobs.

genai.tunings.AsyncTunings.tune(model_name: str, training_data: list, ...)
  - Asynchronously initiates a new model tuning job.
  - Parameters:
    - model_name: The name of the model to tune.
    - training_data: A list of training data examples.
    - ...: Other tuning parameters.

genai.tunings.Tunings.get(tuning_id: str)
  - Synchronously retrieves a specific tuning job.
  - Parameters:
    - tuning_id: The unique identifier for the tuning job.

genai.tunings.Tunings.list()
  - Synchronously lists available tuning jobs.

genai.tunings.Tunings.tune(model_name: str, training_data: list, ...)
  - Synchronously initiates a new model tuning job.
  - Parameters:
    - model_name: The name of the model to tune.
    - training_data: A list of training data examples.
    - ...: Other tuning parameters.
```

----------------------------------------

TITLE: LiveClientContent Attributes
DESCRIPTION: Describes the attributes of the LiveClientContent type, specifically turn_complete and turns.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientContent:
  turn_complete: Boolean indicating if the turn is complete.
  turns: A list of turns in the conversation.
```

----------------------------------------

TITLE: ListFilesConfig Fields
DESCRIPTION: Specifies configuration parameters for listing files, including http_options for request customization, page_size for result limiting, and page_token for pagination.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListFilesConfig:
    http_options: dict
    page_size: int
    page_token: str
```

----------------------------------------

TITLE: LiveConnectConstraints Parameters
DESCRIPTION: Defines the constraints for LiveConnect, including associated configuration and model information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectConstraints:
  config: The LiveConnect configuration.
  model: Information about the model being used.
```

----------------------------------------

TITLE: SpeakerVoiceConfigDict and SpeechConfig
DESCRIPTION: Defines configurations for speaker voice settings and general speech synthesis parameters. Includes language code and multi-speaker voice configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SpeakerVoiceConfigDict:
  speaker: str
  voice_config: dict

SpeechConfig:
  language_code: str
  multi_speaker_voice_config: dict
  voice_config: dict
```

----------------------------------------

TITLE: Tool Configuration and Definitions
DESCRIPTION: Defines types for tools, including their configurations and specific tool implementations like code execution and search.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Tool:
  code_execution: ToolCodeExecution | None
    Configuration for code execution tool.
  enterprise_web_search: dict | None
    Configuration for enterprise web search tool.
  function_declarations: list[dict] | None
    Declarations for functions the model can call.
  google_maps: dict | None
    Configuration for Google Maps tool.
  google_search: dict | None
    Configuration for Google Search tool.
  google_search_retrieval: dict | None
    Configuration for Google Search Retrieval tool.
  retrieval: dict | None
    Configuration for retrieval tool.
  url_context: dict | None
    Configuration for URL context tool.

ToolCodeExecution:
  # No specific fields documented, implies a simple flag or structure.

ToolConfig:
  function_calling_config: dict | None
    Configuration for function calling.
  retrieval_config: dict | None
    Configuration for retrieval.

ToolDict:
  code_execution: dict | None
    Configuration for code execution tool.
  enterprise_web_search: dict | None
    Configuration for enterprise web search tool.
  function_declarations: list[dict] | None
    Declarations for functions the model can call.
  google_maps: dict | None
    Configuration for Google Maps tool.
  google_search: dict | None
    Configuration for Google Search tool.
  google_search_retrieval: dict | None
    Configuration for Google Search Retrieval tool.
  retrieval: dict | None
    Configuration for retrieval tool.
  url_context: dict | None
    Configuration for URL context tool.

ToolConfigDict:
  function_calling_config: dict | None
    Configuration for function calling.
  retrieval_config: dict | None
    Configuration for retrieval.
```

----------------------------------------

TITLE: ControlReferenceConfig API
DESCRIPTION: API documentation for ControlReferenceConfig, covering control type and image computation enablement, along with its dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ControlReferenceConfig:
  control_type: ControlReferenceType
    The type of control to apply.
  enable_control_image_computation: bool
    Whether to enable control image computation.

ControlReferenceConfigDict:
  control_type: ControlReferenceType
    The type of control to apply.
  enable_control_image_computation: bool
    Whether to enable control image computation.
```

----------------------------------------

TITLE: GenAI Batches Module Documentation
DESCRIPTION: Documentation for the genai.batches module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.batches
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: Token Management with genai.tokens
DESCRIPTION: Provides methods for asynchronous and synchronous token creation and management. These are essential for interacting with generative models that require tokenization.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.tokens.AsyncTokens.create()
  - Asynchronously creates a token.

genai.tokens.Tokens.create()
  - Synchronously creates a token.
```

----------------------------------------

TITLE: Video Attributes and Methods
DESCRIPTION: Documentation for the Video type, including its attributes and methods for handling video data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Video:
  mime_type: The MIME type of the video.
  uri: The URI of the video.
  video_bytes: The video data as bytes.

  from_file(file_path: str):
    Creates a Video object from a file.
    Parameters:
      file_path: The path to the video file.
    Returns: A Video object.

  save(file_path: str):
    Saves the video to a file.
    Parameters:
      file_path: The path to save the video file.

  show():
    Displays the video.
```

----------------------------------------

TITLE: BatchJobDestination Properties
DESCRIPTION: Defines the properties for specifying the destination of a batch job, including BigQuery URI, file name, format, GCS URI, and whether responses should be inlined.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchJobDestination:
  bigquery_uri: str
    Description: The BigQuery table URI for the output.
  file_name: str
    Description: The name of the output file.
  format: str
    Description: The format of the output file (e.g., 'jsonl', 'csv').
  gcs_uri: str
    Description: The Google Cloud Storage URI for the output.
  inlined_responses: bool
    Description: Whether to inline the responses in the output.
```

----------------------------------------

TITLE: Authentication Configuration Types
DESCRIPTION: Defines different types of authentication configurations including HTTP Basic, OAuth, and OIDC. Each configuration may have specific parameters like credentials, access tokens, or service accounts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AuthConfigHttpBasicAuthConfigDict:
  credential_secret: str
    The secret credential for HTTP Basic authentication.

AuthConfigOauthConfig:
  access_token: str
    The OAuth access token.
  service_account: str
    The service account identifier for OAuth authentication.

AuthConfigOauthConfigDict:
  access_token: str
    The OAuth access token.
  service_account: str
    The service account identifier for OAuth authentication.

AuthConfigOidcConfig:
  id_token: str
    The OIDC ID token.
  service_account: str
    The service account identifier for OIDC authentication.

AuthConfigOidcConfigDict:
  id_token: str
    The OIDC ID token.
  service_account: str
    The service account identifier for OIDC authentication.
```

----------------------------------------

TITLE: SupervisedHyperParametersDict API Documentation
DESCRIPTION: Provides dictionary-based access to supervised learning hyperparameters, mirroring SupervisedHyperParameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedHyperParametersDict:
  adapter_size: The size of the adapter for training.
  batch_size: The number of samples in each training batch.
  epoch_count: The total number of training epochs.
  learning_rate: The learning rate for the optimizer.
  learning_rate_multiplier: A multiplier for the learning rate.
```

----------------------------------------

TITLE: LiveMusicClientContentDict Attributes
DESCRIPTION: Details the attributes available within the LiveMusicClientContentDict, specifically 'weighted_prompts'. This is used for content related to live music generation, similar to LiveMusicClientContent.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicClientContentDict:
  weighted_prompts: A list of prompts with associated weights for music generation.
```

----------------------------------------

TITLE: Generate Content with Typed Configuration Parameters
DESCRIPTION: Illustrates using Pydantic types from `google.genai.types` for specifying generation configuration parameters such as temperature, top_p, top_k, and seed for deterministic output.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)
```

----------------------------------------

TITLE: List Tuning Jobs (Asynchronous)
DESCRIPTION: Provides an asynchronous method to list tuning jobs, enabling efficient handling of multiple requests without blocking. It demonstrates iterating through jobs and managing pagination asynchronously.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Provide a string as content
DESCRIPTION: Shows how providing a single string to the `contents` argument is interpreted by the SDK. It's converted into a `types.UserContent` with a single text part.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
contents='Why is the sky blue?'

# The SDK will assume this is a text part, and it converts this into the following:
# [
# types.UserContent(
#     parts=[
#     types.Part.from_text(text='Why is the sky blue?')
#     ]
# )
# ]
# 
# Where a `types.UserContent` is a subclass of `types.Content`, it sets the
# `role` field to be `user`.
```

----------------------------------------

TITLE: File Upload
DESCRIPTION: Demonstrates uploading files to the Generative AI API. This involves using the `client.files.upload` method with the local file path.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)
```

----------------------------------------

TITLE: SupervisedTuningSpec Properties
DESCRIPTION: Defines the specifications for supervised tuning, including dataset URIs, tuning mode, and hyper-parameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningSpec:
  export_last_checkpoint_only: bool
    Whether to export only the last checkpoint.
  hyper_parameters: dict
    A dictionary of hyper-parameters for tuning.
  training_dataset_uri: str
    The URI of the training dataset.
  tuning_mode: str
    The mode of tuning (e.g., 'full-data', 'parameter-efficient').
  validation_dataset_uri: str
    The URI of the validation dataset.
```

----------------------------------------

TITLE: LiveMusicClientContent Attributes
DESCRIPTION: Details the attributes available within the LiveMusicClientContent, specifically 'weighted_prompts'. This is used for content related to live music generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicClientContent:
  weighted_prompts: A list of prompts with associated weights for music generation.
```

----------------------------------------

TITLE: List Tuned Models (Asynchronous)
DESCRIPTION: Provides an asynchronous way to list tuned models, allowing for non-blocking iteration over the results. It demonstrates fetching page size and individual models asynchronously.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False}})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Function Calling with Automatic Python Function Support
DESCRIPTION: Demonstrates how to enable automatic function calling by passing a Python function directly to the `tools` parameter in `generate_content`. The SDK handles the function execution and response.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: GenerateContentResponsePromptFeedbackDict Attributes
DESCRIPTION: Outlines the attributes of GenerateContentResponsePromptFeedbackDict, mirroring prompt feedback details such as block reason, block reason message, and safety ratings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponsePromptFeedbackDict:
  block_reason: The reason the response was blocked.
  block_reason_message: A message explaining the block reason.
  safety_ratings: Safety ratings for the response.
```

----------------------------------------

TITLE: API Key Configuration Types
DESCRIPTION: Details the configuration for API key authentication, including the API key string itself.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ApiKeyConfig:
  api_key_string: The string representation of the API key.

ApiKeyConfigDict:
  api_key_string: The API key string as a dictionary entry.
```

----------------------------------------

TITLE: GenerateVideosResponseDict Fields
DESCRIPTION: A dictionary representation of the GenerateVideosResponse.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateVideosResponseDict:
  generated_videos: list[GeneratedVideoDict]
    A list of generated videos.
  rai_media_filtered_count: int
    The number of media items filtered by RAI.
  rai_media_filtered_reasons: list[str]
    The reasons why media items were filtered by RAI.
```

----------------------------------------

TITLE: PartnerModelTuningSpecDict Attributes
DESCRIPTION: A dictionary representation of PartnerModelTuningSpec, used for specifying partner model tuning configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PartnerModelTuningSpecDict:
  hyper_parameters: Configuration for hyper-parameters.
  training_dataset_uri: URI for the training dataset.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: SupervisedTuningDatasetDistributionDatasetBucketDict Properties
DESCRIPTION: A dictionary representation of a dataset distribution bucket, providing count and range information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDatasetDistributionDatasetBucketDict:
  count: int
    The number of items in this bucket.
  left: float
    The left boundary of the bucket.
  right: float
    The right boundary of the bucket.
```

----------------------------------------

TITLE: Live Music Set Weighted Prompts Parameters Types
DESCRIPTION: Defines parameter types for setting weighted prompts in live music generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicSetWeightedPromptsParameters, LiveMusicSetWeightedPromptsParametersDict

# Example usage for LiveMusicSetWeightedPromptsParameters:
weighted_prompts_params = LiveMusicSetWeightedPromptsParameters(weighted_prompts=[('prompt1', 0.8)])

# Example usage for LiveMusicSetWeightedPromptsParametersDict:
weighted_prompts_params_dict = LiveMusicSetWeightedPromptsParametersDict(weighted_prompts=[('prompt1', 0.8)])

```

----------------------------------------

TITLE: API Reference
DESCRIPTION: General API reference for the Google Generative AI Python SDK. This section provides an overview of available classes, methods, and their usage.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerativeModel:
  __init__(model_name: str, **kwargs)
    model_name: The name of the generative model (e.g., 'gemini-pro').
    **kwargs: Additional configuration options.

  generate_content(prompt: str, **kwargs) -> GenerateContentResponse
    prompt: The input prompt for the model.
    **kwargs: Generation configuration parameters.
    Returns: A GenerateContentResponse object containing the model's output.

  count_tokens(prompt: str) -> CountTokensResponse
    prompt: The input prompt for token counting.
    Returns: A CountTokensResponse object with token counts.
```

----------------------------------------

TITLE: List Tuned Models (Synchronous)
DESCRIPTION: Lists available tuned models with optional pagination and filtering. It demonstrates how to iterate through a list of models and access their properties.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
for model in client.models.list(config={'page_size': 10, 'query_base': False}}):
    print(model)
```

LANGUAGE: python
CODE:
```
pager = client.models.list(config={'page_size': 10, 'query_base': False}})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: AsyncSession Methods
DESCRIPTION: Provides methods for managing asynchronous sessions, including closing the session, receiving data, sending content, and managing real-time inputs and streams.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AsyncSession:
  close(): Closes the asynchronous session.
  receive(): Receives data from the session.
  send(content): Sends content through the session.
  send_client_content(content): Sends client-specific content.
  send_realtime_input(input): Sends real-time input.
  send_tool_response(response): Sends a tool response.
  start_stream(): Starts a streaming session.
```

----------------------------------------

TITLE: ProactivityConfig Fields
DESCRIPTION: Details the fields available for ProactivityConfig, specifically for proactive audio settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ProactivityConfig:
  proactive_audio: Configuration for proactive audio.
```

----------------------------------------

TITLE: Use Tuned Model for Content Generation
DESCRIPTION: Generates content using a tuned model by providing the model's endpoint and the input prompt. It then prints the generated text response.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

----------------------------------------

TITLE: GenAI Tokens Module Documentation
DESCRIPTION: Documentation for the genai.tokens module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.tokens
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: GenAI Files Module Documentation
DESCRIPTION: Documentation for the genai.files module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.files
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: GenerationConfig Parameters
DESCRIPTION: Details the various parameters available within the GenerationConfig class for controlling generative model behavior. These include settings for audio timestamps, candidate count, frequency and presence penalties, log probabilities, output tokens, media resolution, response formatting, and more.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfig:
  audio_timestamp: Configuration for audio timestamp generation.
  candidate_count: The number of candidates to generate.
  enable_affective_dialog: Enables affective dialog features.
  frequency_penalty: Controls the frequency penalty for generated text.
  logprobs: Whether to include log probabilities in the output.
  max_output_tokens: The maximum number of tokens to generate.
  media_resolution: Configuration for media resolution.
  model_selection_config: Configuration for model selection.
  presence_penalty: Controls the presence penalty for generated text.
  response_json_schema: A JSON schema for the response.
  response_logprobs: Whether to include log probabilities in the response.
  response_mime_type: The MIME type for the response.
  response_modalities: The modalities for the response.
  response_schema: The schema for the response.
  routing_config: Configuration for routing.
  seed: The seed for random number generation.
  speech_config: Configuration for speech generation.
  stop_sequences: Sequences that stop generation.
  temperature: Controls the randomness of the output.
  thinking_config: Configuration for thinking process.
  top_k: The top-k sampling parameter.
  top_p: The top-p sampling parameter.
```

----------------------------------------

TITLE: LiveServerToolCall and LiveServerToolCallDict
DESCRIPTION: Defines the structure for tool calls in a live server environment. Includes function calls and their parameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerToolCall:
  function_calls: List[FunctionCall]
    A list of function calls to be executed.

LiveServerToolCallDict:
  function_calls: List[FunctionCallDict]
    A list of function calls to be executed.
```

----------------------------------------

TITLE: SupervisedTuningDatasetDistributionDict Properties
DESCRIPTION: A dictionary representation of dataset distribution statistics for supervised tuning, including billable sum, buckets, and various statistical measures.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDatasetDistributionDict:
  billable_sum: float
    The sum of billable values in the dataset.
  buckets: list[SupervisedTuningDatasetDistributionDatasetBucketDict]
    A list of dataset buckets.
  max: float
    The maximum value of the dataset.
  mean: float
    The mean of the dataset.
  median: float
    The median of the dataset.
  min: float
    The minimum value of the dataset.
  p5: float
    The 5th percentile of the dataset.
  p95: float
    The 95th percentile of the dataset.
  sum: float
    The sum of the dataset values.
```

----------------------------------------

TITLE: Embed Content with Configuration
DESCRIPTION: Shows how to embed multiple content pieces with specific configuration, such as setting the output dimensionality, using `client.models.embed_content` and `types.EmbedContentConfig`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)

```

----------------------------------------

TITLE: UpscaleImageConfig Parameters
DESCRIPTION: Defines configuration options for upscaling an image, including RAI reason inclusion, output compression quality, and MIME type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageConfig:
  include_rai_reason: bool
    Whether to include the RAI reason in the response.
  output_compression_quality: int
    The quality of the output compression (0-100).
  output_mime_type: str
    The MIME type of the output image.
```

----------------------------------------

TITLE: VideoCompressionQuality Enum
DESCRIPTION: Documentation for the VideoCompressionQuality enum, defining available compression qualities.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VideoCompressionQuality:
  LOSSLESS: Lossless compression.
  OPTIMIZED: Optimized compression.
```

----------------------------------------

TITLE: API Documentation for CreateCachedContentConfig
DESCRIPTION: Specifies the configuration for creating cached content, including its contents, display name, expiration, and associated tools or system instructions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CreateCachedContentConfig:
  contents: list
    The content to be cached.
  display_name: str
    A user-friendly name for the cached content.
  expire_time: datetime
    The time at which the cached content should expire.
  http_options: dict
    Optional HTTP client options for the request.
  kms_key_name: str
    The Cloud KMS key name to use for encrypting the cached content.
  system_instruction: str
    A system instruction to associate with the cached content.
  tool_config: dict
    Configuration for tools to be used with the cached content.
  tools: list
    A list of tools to associate with the cached content.
  ttl: timedelta
    The time-to-live for the cached content.
```

----------------------------------------

TITLE: GenerationConfigRoutingConfigManualRoutingMode Parameters
DESCRIPTION: Specifies the parameters for manual model routing, allowing direct selection of a model by name.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigRoutingConfigManualRoutingMode:
  model_name: The name of the model to use for generation.
```

----------------------------------------

TITLE: Create Cached Content
DESCRIPTION: Demonstrates how to create cached content using the client library. It supports both Vertex AI and non-Vertex AI environments, allowing for the creation of cached content from specified file URIs with configuration for model, contents, system instructions, display name, and time-to-live.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

if client.vertexai:
    file_uris = [
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=file_uris[0], mime_type='application/pdf'
                    ),
                    types.Part.from_uri(
                        file_uri=file_uris[1],
                        mime_type='application/pdf',
                    ),
                ],
            )
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)
```

----------------------------------------

TITLE: GenerateVideosConfig Attributes
DESCRIPTION: Details the configuration attributes for generating videos using the Python GenAI library. Covers aspects like aspect ratio, duration, FPS, and output options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateVideosConfig:
  aspect_ratio: The aspect ratio for the generated videos.
  compression_quality: The compression quality setting for videos.
  duration_seconds: The duration of the generated videos in seconds.
  enhance_prompt: Whether to enhance the prompt for video generation.
  fps: Frames per second for the generated videos.
  generate_audio: Whether to generate audio for the videos.
  http_options: HTTP client options for the request.
  last_frame: Configuration for the last frame of the video.
  negative_prompt: A prompt to guide the video generation away from certain content.
  number_of_videos: The number of videos to generate.
  output_gcs_uri: Google Cloud Storage URI for the output videos.
  person_generation: Configuration for person generation in videos.
  pubsub_topic: Pub/Sub topic for notifications.
  resolution: The resolution of the generated videos.
  seed: Seed for random number generation.

GenerateVideosConfigDict:
  aspect_ratio: The aspect ratio for the generated videos.
  compression_quality: The compression quality setting for videos.
  duration_seconds: The duration of the generated videos in seconds.
  enhance_prompt: Whether to enhance the prompt for video generation.
  fps: Frames per second for the generated videos.
  generate_audio: Whether to generate audio for the videos.
  http_options: HTTP client options for the request.
  last_frame: Configuration for the last frame of the video.
  negative_prompt: A prompt to guide the video generation away from certain content.
  number_of_videos: The number of videos to generate.
  output_gcs_uri: Google Cloud Storage URI for the output videos.
  person_generation: Configuration for person generation in videos.
  pubsub_topic: Pub/Sub topic for notifications.
  resolution: The resolution of the generated videos.
  seed: Seed for random number generation.
```

----------------------------------------

TITLE: List Tuning Jobs
DESCRIPTION: Lists tuning jobs associated with the client, with support for pagination. It demonstrates how to iterate through jobs and access job details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: VertexRagStoreRagResource Attributes
DESCRIPTION: Documentation for the VertexRagStoreRagResource type, outlining its attributes for specifying RAG resources.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexRagStoreRagResource:
  rag_corpus: The RAG corpus to use.
  rag_file_ids: A list of file IDs to include in the RAG resource.
```

----------------------------------------

TITLE: LiveConnectConstraintsDict Parameters
DESCRIPTION: Defines the dictionary-based representation of LiveConnect constraints.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectConstraintsDict:
  config: The LiveConnect configuration.
  model: Information about the model being used.
```

----------------------------------------

TITLE: API Reference: Google Generative AI
DESCRIPTION: Provides a comprehensive reference for the Google Generative AI API, detailing available models, methods for content generation, chat interactions, safety settings, and more.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GoogleGenerativeAI:
  __init__(model_name: str, **kwargs)
    Initializes the GenerativeModel with a specific model name and optional configurations.
    Parameters:
      model_name: The name of the generative model (e.g., 'gemini-pro').
      **kwargs: Additional configuration options like safety_settings, tools, generation_config, etc.

  generate_content(prompt: str, stream: bool = False, **kwargs)
    Generates content based on a given prompt. Supports streaming responses.
    Parameters:
      prompt: The input text or content for generation.
      stream: If True, returns a streaming response.
      **kwargs: Additional parameters for content generation.
    Returns:
      A GenerateContentResponse object or a streaming iterator.

  generate_content_async(prompt: str, stream: bool = False, **kwargs)
    Asynchronously generates content based on a given prompt. Supports streaming.
    Parameters:
      prompt: The input text or content for generation.
      stream: If True, returns an asynchronous streaming response.
    Returns:
      An awaitable GenerateContentResponse object or an async iterator.

  count_tokens(prompt: str)
    Counts the number of tokens in the provided prompt for the initialized model.
    Parameters:
      prompt: The text to count tokens for.
    Returns:
      A CountTokensResponse object containing the token count.

  start_chat(history: list = None, **kwargs)
    Starts a new chat session with the model, optionally initializing with conversation history.
    Parameters:
      history: A list of previous messages to initialize the chat.
    Returns:
      A ChatSession object for managing the conversation.

  upload_file(file_obj: file, display_name: str = None)
    Uploads a file to be used with the generative models.
    Parameters:
      file_obj: A file-like object to upload.
      display_name: An optional display name for the file.
    Returns:
      An UploadFileResponse object containing the file URI.

  get_file(file_uri: str)
    Retrieves information about an uploaded file.
    Parameters:
      file_uri: The URI of the file to retrieve.
    Returns:
      A GetFileResponse object with file details.

  delete_file(file_uri: str)
    Deletes an uploaded file.
    Parameters:
      file_uri: The URI of the file to delete.

  create_cache(display_name: str = None)
    Creates a new cache for storing model responses.
    Parameters:
      display_name: An optional display name for the cache.
    Returns:
      A CreateCacheResponse object with the cache ID.

  get_cache(cache_id: str)
    Retrieves information about a specific cache.
    Parameters:
      cache_id: The ID of the cache to retrieve.
    Returns:
      A GetCacheResponse object with cache details.

  tune_model(training_data: str, **kwargs)
    Initiates a model tuning job using the provided training data.
    Parameters:
      training_data: The URI of the training dataset (e.g., a JSONL file).
      **kwargs: Additional parameters for the tuning job.
    Returns:
      A TuningJob object representing the tuning process.

  get_tuning_job(tuning_job_name: str)
    Retrieves the status and details of a model tuning job.
    Parameters:
      tuning_job_name: The name of the tuning job.
    Returns:
      A TuningJob object with the job's status and details.

  list_models(**kwargs)
    Lists available generative models.
    Returns:
      A list of Model objects.

  list_tuned_models(**kwargs)
    Lists all available tuned models.
    Returns:
      A list of TunedModel objects.

SafetySettings:
  category: The category of safety setting (e.g., HARM_CATEGORY_DANGEROUS_CONTENT).
  threshold: The blocking threshold for the category (e.g., BLOCK_MEDIUM_AND_ABOVE).

FunctionDeclaration:
  name: The name of the function.
  description: A description of what the function does.
  parameters: A schema defining the function's parameters.

ParameterSchema:
  type: The data type of the parameter (e.g., OBJECT, STRING, INTEGER).
  properties: An object defining the properties of the parameter if type is OBJECT.
  required: A list of required parameter names.

Error Handling:
  - API errors may occur due to network issues, invalid requests, or rate limiting.
  - Specific error codes and messages will be provided in the API response.
  - Implement retry mechanisms for transient errors.
```

----------------------------------------

TITLE: LiveConnectConstraintsDict Attributes
DESCRIPTION: Details the attributes available within the LiveConnectConstraintsDict, including 'config' and 'model'. These are used for configuring live connections.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectConstraintsDict:
  config: Configuration settings for the live connection.
  model: Specifies the model to be used for the live connection.
```

----------------------------------------

TITLE: ListFilesConfigDict Fields
DESCRIPTION: Defines the dictionary structure for ListFilesConfig, containing http_options, page_size, and page_token for file listing operations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListFilesConfigDict:
    http_options: dict
    page_size: int
    page_token: str
```

----------------------------------------

TITLE: ControlReferenceImage API
DESCRIPTION: API documentation for ControlReferenceImage, detailing its configuration, reference ID, image data, and reference type, along with its dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ControlReferenceImage:
  config: ControlReferenceConfig
    Configuration for the control reference image.
  control_image_config: ControlReferenceConfig
    Configuration for the control image.
  reference_id: str
    A unique identifier for the reference image.
  reference_image: bytes
    The reference image data.
  reference_type: ControlReferenceType
    The type of the reference image.

ControlReferenceImageDict:
  config: ControlReferenceConfigDict
    Configuration for the control reference image.
  reference_id: str
    A unique identifier for the reference image.
  reference_image: bytes
    The reference image data.
  reference_type: ControlReferenceType
    The type of the reference image.
```

----------------------------------------

TITLE: Python GenAI Batches API
DESCRIPTION: Documentation for the `genai.batches` module, covering `AsyncBatches` and `Batches` classes. This includes methods for creating, retrieving, listing, canceling, and deleting batches.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AsyncBatches:
  Methods:
    cancel(batch: str):
      Cancels a batch.
    create(batch: dict):
      Creates a new batch.
    delete(batch: str):
      Deletes a batch.
    get(batch: str):
      Retrieves a specific batch.
    list():
      Lists all batches.

Batches:
  Methods:
    cancel(batch: str):
      Cancels a batch.
```

----------------------------------------

TITLE: GeneratedVideoDict Fields
DESCRIPTION: A dictionary representation of the GeneratedVideo.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GeneratedVideoDict:
  video: VideoDict
    A dictionary representation of the generated video.
```

----------------------------------------

TITLE: Create Tuning Job
DESCRIPTION: Creates a new tuning job for a base model with a specified training dataset and configuration. This involves defining the number of epochs and a display name for the tuned model.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    )
)
print(tuning_job)
```

----------------------------------------

TITLE: Image Type and Methods
DESCRIPTION: Documentation for the Image type, including its attributes and methods for handling image data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Image:
  gcs_uri: Google Cloud Storage URI of the image.
  image_bytes: The raw bytes of the image.
  mime_type: The MIME type of the image.
  from_file(path: str):
    Creates an Image object from a local file path.
  model_post_init():
    Internal method for post-initialization of the Image model.
  save(path: str):
    Saves the image to a specified file path.
  show():
    Displays the image.
```

----------------------------------------

TITLE: PreferenceOptimizationSpec Fields
DESCRIPTION: Details the fields available for PreferenceOptimizationSpec, including training and validation dataset URIs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationSpec:
  training_dataset_uri: URI for the training dataset.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: GenAI Models Module Documentation
DESCRIPTION: Documentation for the genai.models module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.models
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: Python GenAI Configuration Types
DESCRIPTION: This section details various configuration types available in the Python GenAI library, including settings for generation, batch jobs, cached content, file operations, model retrieval, operation status, tuning jobs, Google Maps integration, and Google Search.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigThinkingConfigDict:
  thinking_budget: Configuration for thinking budget.

GetBatchJobConfig:
  http_options: HTTP options for batch job configuration.

GetBatchJobConfigDict:
  http_options: HTTP options for batch job configuration.

GetCachedContentConfig:
  http_options: HTTP options for cached content configuration.

GetCachedContentConfigDict:
  http_options: HTTP options for cached content configuration.

GetFileConfig:
  http_options: HTTP options for file configuration.

GetFileConfigDict:
  http_options: HTTP options for file configuration.

GetModelConfig:
  http_options: HTTP options for model configuration.

GetModelConfigDict:
  http_options: HTTP options for model configuration.

GetOperationConfig:
  http_options: HTTP options for operation configuration.

GetOperationConfigDict:
  http_options: HTTP options for operation configuration.

GetTuningJobConfig:
  http_options: HTTP options for tuning job configuration.

GetTuningJobConfigDict:
  http_options: HTTP options for tuning job configuration.

GoogleMaps:
  auth_config: Authentication configuration for Google Maps.

GoogleMapsDict:
  auth_config: Authentication configuration for Google Maps.

GoogleRpcStatus:
  code: The status code.
  details: Additional details about the status.
  message: The status message.

GoogleRpcStatusDict:
  code: The status code.
  details: Additional details about the status.
  message: The status message.

GoogleSearch:
  time_range_filter: Filter for the time range of search results.
```

----------------------------------------

TITLE: GroundingChunkMapsPlaceAnswerSourcesReviewSnippet
DESCRIPTION: Details of a review snippet for place answer sources, including author attribution, content URIs, and review text.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsPlaceAnswerSourcesReviewSnippet:
  author_attribution: GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution
    Information about the author of the review.
  flag_content_uri: str
    The URI of the flagged content within the review.
  google_maps_uri: str
    The URI to the review on Google Maps.
  relative_publish_time_description: str
    A description of when the review was published relative to the current time.
  review: str
    The text content of the review.
```

----------------------------------------

TITLE: PreferenceOptimizationHyperParametersDict Fields
DESCRIPTION: Details the hyperparameters for preference optimization as a dictionary, including adapter size, beta, epoch count, and learning rate multiplier.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationHyperParametersDict:
  adapter_size: int
    The size of the adapter.
  beta: float
    The beta parameter for optimization.
  epoch_count: int
    The number of epochs for tuning.
  learning_rate_multiplier: float
    A multiplier for the learning rate.
```

----------------------------------------

TITLE: PreferenceOptimizationHyperParameters Fields
DESCRIPTION: Details the hyperparameters for preference optimization, including adapter size, beta, epoch count, and learning rate multiplier.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationHyperParameters:
  adapter_size: int
    The size of the adapter.
  beta: float
    The beta parameter for optimization.
  epoch_count: int
    The number of epochs for tuning.
  learning_rate_multiplier: float
    A multiplier for the learning rate.
```

----------------------------------------

TITLE: VertexAISearchDataStoreSpec API Documentation
DESCRIPTION: Documentation for VertexAISearchDataStoreSpec, specifying data store and filter.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexAISearchDataStoreSpec:
  data_store: The data store to use.
  filter: Filter criteria for the data store.
```

----------------------------------------

TITLE: PartnerModelTuningSpec Attributes
DESCRIPTION: Defines specifications for tuning partner models, including hyper-parameters and dataset URIs for training and validation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PartnerModelTuningSpec:
  hyper_parameters: Configuration for hyper-parameters.
  training_dataset_uri: URI for the training dataset.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: UrlContextDict
DESCRIPTION: Dictionary representation of UrlContext.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlContextDict:
  # No specific fields documented in the provided text.
```

----------------------------------------

TITLE: GenerateImagesConfig Parameters
DESCRIPTION: Configuration options for generating images, including safety attributes, language, negative prompts, and output settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateImagesConfig:
  include_rai_reason: bool
    Whether to include RAI reason in the response.
  include_safety_attributes: bool
    Whether to include safety attributes in the response.
  language: str
    The language for the generated content.
  negative_prompt: str
    A prompt that describes what to avoid in the generated image.
  number_of_images: int
    The number of images to generate.
  output_compression_quality: int
    The compression quality for the output image (0-100).
  output_gcs_uri: str
    The Google Cloud Storage URI for the output image.
  output_mime_type: str
    The MIME type for the output image (e.g., 'image/png').
  person_generation: str
    Specifies the type of person generation (e.g., 'photorealistic').
  safety_filter_level: str
    The safety filter level to apply (e.g., 'high').
  seed: int
    A seed for reproducible image generation.
```

----------------------------------------

TITLE: GenerationConfigThinkingConfig Parameters
DESCRIPTION: Configures the model's thinking process, including whether to include thoughts and the thinking budget.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigThinkingConfig:
  include_thoughts: Whether to include the model's thought process in the output.
  thinking_budget: The maximum budget for the model's thinking process.
```

----------------------------------------

TITLE: GenerationConfigRoutingConfig Parameters
DESCRIPTION: Defines the routing configuration for generative models, allowing for automatic or manual model selection.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigRoutingConfig:
  auto_mode: Configuration for automatic model routing.
  manual_mode: Configuration for manual model selection.
```

----------------------------------------

TITLE: GroundingChunkMapsPlaceAnswerSources Attributes
DESCRIPTION: Details the attributes of GroundingChunkMapsPlaceAnswerSources, including flag_content_uri and review_snippets for answer source information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsPlaceAnswerSources:
  flag_content_uri: URI for flagged content.
  review_snippets: Snippets from reviews.
```

----------------------------------------

TITLE: HttpRetryOptions and HttpRetryOptionsDict
DESCRIPTION: Configuration options for retrying HTTP requests, including the number of attempts, base for exponential backoff, specific HTTP status codes to retry, initial delay, jitter, and maximum delay. HttpRetryOptionsDict is the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HttpRetryOptions:
  attempts: int
    The maximum number of retry attempts.
  exp_base: float
    The base for exponential backoff calculation.
  http_status_codes: list[int]
    A list of HTTP status codes that should trigger a retry.
  initial_delay: float
    The initial delay in seconds before the first retry.
  jitter: float
    The factor for adding randomness to the retry delay.
  max_delay: float
    The maximum delay in seconds between retries.

HttpRetryOptionsDict:
  attempts: int
    The maximum number of retry attempts.
  exp_base: float
    The base for exponential backoff calculation.
  http_status_codes: list[int]
    A list of HTTP status codes that should trigger a retry.
  initial_delay: float
    The initial delay in seconds before the first retry.
  jitter: float
    The factor for adding randomness to the retry delay.
  max_delay: float
    The maximum delay in seconds between retries.
```

----------------------------------------

TITLE: Import Gen AI Modules
DESCRIPTION: Imports the necessary genai and types modules from the google library for using the SDK.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types
```

----------------------------------------

TITLE: UpscaleImageConfigDict Parameters
DESCRIPTION: Dictionary representation of UpscaleImageConfig, allowing for image enhancement, HTTP options, image preservation factor, RAI reason, output compression quality, and MIME type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageConfigDict:
  enhance_input_image: bool
    Whether to enhance the input image.
  http_options: dict
    Options for HTTP requests.
  image_preservation_factor: int
    Factor for preserving image quality during upscaling.
  include_rai_reason: bool
    Whether to include the RAI reason in the response.
  output_compression_quality: int
    The quality of the output compression (0-100).
  output_mime_type: str
    The MIME type of the output image.
```

----------------------------------------

TITLE: Generate Content (Async Streaming) (Python)
DESCRIPTION: Demonstrates asynchronous streaming of content generation from a model. This method is useful for receiving output incrementally and uses the 'gemini-2.0-flash-001' model.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Generate Content Stream (Python)
DESCRIPTION: Demonstrates how to generate content from a model using a streaming approach, processing image bytes. This requires the 'gemini-2.0-flash-001' model and provides output chunk by chunk.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: UpscaleImageParametersDict
DESCRIPTION: Dictionary representation of UpscaleImageParameters, containing configuration, image data, model name, and upscale factor.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageParametersDict:
  config: UpscaleImageConfigDict
    Configuration for the upscaling process.
  image: bytes or PIL.Image.Image
    The image data to upscale.
  model: str
    The name of the model to use for upscaling.
  upscale_factor: int
    The factor by which to upscale the image.
```

----------------------------------------

TITLE: GenerationConfigRoutingConfigAutoRoutingMode Parameters
DESCRIPTION: Specifies the parameters for automatic model routing, including preferences for model selection.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigRoutingConfigAutoRoutingMode:
  model_routing_preference: The preference for routing models (e.g., 'best_available', 'fastest_available').
```

----------------------------------------

TITLE: DistillationSpecDict Configuration
DESCRIPTION: Defines the configuration for distillation, including base teacher model, hyperparameters, pipeline root directory, student model, dataset URIs, and tuned teacher model source.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DistillationSpecDict:
  base_teacher_model: The base teacher model to use for distillation.
  hyper_parameters: Hyperparameters for the distillation process.
  pipeline_root_directory: The root directory for the distillation pipeline.
  student_model: The student model to be trained.
  training_dataset_uri: URI for the training dataset.
  tuned_teacher_model_source: Source for the tuned teacher model.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: Generate Content with Pydantic Model Schema
DESCRIPTION: Demonstrates how to generate content and specify a Pydantic model for the response schema. The model's output will conform to the defined Pydantic model.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from google.genai import types


class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: GenAI Caches Module Documentation
DESCRIPTION: Documentation for the genai.caches module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.caches
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: UsageMetadata Structure
DESCRIPTION: Provides detailed information about the usage of a model, including token counts for prompts, responses, and cached content. It also includes details about token usage for tool use and traffic types.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UsageMetadata:
  prompt_token_count: int
    The number of tokens in the prompt.
  response_token_count: int
    The number of tokens in the response.
  cached_content_token_count: int
    The number of tokens from cached content.
  total_token_count: int
    The total number of tokens used.
  prompt_tokens_details: list[TokenCountDetails]
    Details about prompt token usage.
  response_tokens_details: list[TokenCountDetails]
    Details about response token usage.
  cache_tokens_details: list[TokenCountDetails]
    Details about cache token usage.
  thoughts_token_count: int
    The number of tokens used for thoughts.
  tool_use_prompt_token_count: int
    The number of tokens used for tool use prompts.
  tool_use_prompt_tokens_details: list[TokenCountDetails]
    Details about tool use prompt token usage.
  traffic_type: TrafficType
    The type of traffic for this usage.
```

----------------------------------------

TITLE: List Batch Prediction Jobs with Pager
DESCRIPTION: Demonstrates using a pager object to list batch prediction jobs, allowing for efficient retrieval of jobs page by page. It shows how to access page size, individual jobs, and navigate to the next page.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])
```

----------------------------------------

TITLE: Edit Image with Raw and Mask References
DESCRIPTION: Demonstrates how to edit an image using both raw reference images and mask reference images. This allows for precise control over image modifications by specifying background masks.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

----------------------------------------

TITLE: GenerationConfigDict Parameters
DESCRIPTION: Details the various parameters available within the GenerationConfigDict type, which is a dictionary representation of GenerationConfig. These parameters control generative model behavior, including audio timestamps, candidate count, penalties, token limits, media resolution, and response formatting.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerationConfigDict:
  audio_timestamp: Configuration for audio timestamp generation.
  candidate_count: The number of candidates to generate.
  enable_affective_dialog: Enables affective dialog features.
  frequency_penalty: Controls the frequency penalty for generated text.
  logprobs: Whether to include log probabilities in the output.
  max_output_tokens: The maximum number of tokens to generate.
  media_resolution: Configuration for media resolution.
  model_selection_config: Configuration for model selection.
  presence_penalty: Controls the presence penalty for generated text.
  response_json_schema: A JSON schema for the response.
  response_logprobs: Whether to include log probabilities in the response.
```

----------------------------------------

TITLE: SpeechConfigDict
DESCRIPTION: A dictionary-based representation for speech synthesis configuration, mirroring SpeechConfig. It specifies language code and voice configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SpeechConfigDict:
  language_code: str
  multi_speaker_voice_config: dict
  voice_config: dict
```

----------------------------------------

TITLE: Upscale Image with Imagen (Vertex AI Only)
DESCRIPTION: Shows how to upscale a generated image using the `client.models.upscale_image` method. This functionality is only supported in Vertex AI and requires the image object from a previous generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-002',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()

```

----------------------------------------

TITLE: GenerateContentConfig Parameters
DESCRIPTION: Details the configurable parameters for the GenerateContentConfig object, which controls various aspects of content generation, including function calling, caching, token limits, penalties, and safety settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentConfig:
  automatic_function_calling: Enables or disables automatic function calling.
  cached_content: Specifies cached content to be used for generation.
  candidate_count: The number of candidates to generate.
  frequency_penalty: Controls the penalty for repeating tokens.
  http_options: Options for HTTP requests.
  labels: Custom labels for the generation request.
  logprobs: Whether to include log probabilities in the output.
  max_output_tokens: The maximum number of tokens to generate.
  media_resolution: The resolution for media inputs.
  model_selection_config: Configuration for model selection.
  presence_penalty: Controls the penalty for repeating tokens based on presence.
  response_json_schema: A JSON schema for the response.
  response_logprobs: Whether to include log probabilities for the response.
  response_mime_type: The MIME type for the response.
  response_modalities: The modalities expected in the response.
  response_schema: A schema for the response.
  routing_config: Configuration for routing requests.
  safety_settings: Settings for safety filters.
  seed: A seed for reproducible generation.
  speech_config: Configuration for speech generation.
  stop_sequences: Sequences that stop generation.
  system_instruction: System-level instructions for the model.
  temperature: Controls the randomness of the output.
  thinking_config: Configuration for the model's thinking process.
  tool_config: Configuration for tools.
  tools: A list of tools to use for generation.
  top_k: The number of top-k tokens to consider.
  top_p: The cumulative probability of top-p tokens to consider.
```

----------------------------------------

TITLE: PreferenceOptimizationSpecDict Fields
DESCRIPTION: Details the fields available for PreferenceOptimizationSpecDict, including hyperparameters, training, and validation dataset URIs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationSpecDict:
  hyper_parameters: Dictionary of hyperparameters for training.
  training_dataset_uri: URI for the training dataset.
  validation_dataset_uri: URI for the validation dataset.
```

----------------------------------------

TITLE: GroundingChunkMapsPlaceAnswerSourcesDict
DESCRIPTION: A dictionary representing place answer sources within grounding chunks, including content URIs and review snippets.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsPlaceAnswerSourcesDict:
  flag_content_uri: str
    The URI of the flagged content.
  review_snippets: list[GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict]
    A list of review snippets associated with the place answer.
```

----------------------------------------

TITLE: SupervisedTuningDatasetDistribution Properties
DESCRIPTION: Provides access to statistical properties of a dataset distribution for supervised tuning. Includes metrics like mean, median, min, max, and percentiles.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDatasetDistribution:
  mean: float
    The mean of the dataset.
  median: float
    The median of the dataset.
  min: float
    The minimum value of the dataset.
  p5: float
    The 5th percentile of the dataset.
  p95: float
    The 95th percentile of the dataset.
  sum: float
    The sum of the dataset values.
```

----------------------------------------

TITLE: LiveConnectParameters Attributes
DESCRIPTION: Details the attributes available within the LiveConnectParameters, including 'config' and 'model'. These are used for setting parameters for live connections.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectParameters:
  config: Configuration settings for the live connection.
  model: Specifies the model to be used for the live connection.
```

----------------------------------------

TITLE: UpscaleImageParameters
DESCRIPTION: Specifies parameters for image upscaling, including configuration, the image itself, the model to use, and the upscale factor.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageParameters:
  config: UpscaleImageConfig
    Configuration for the upscaling process.
  image: bytes or PIL.Image.Image
    The image data to upscale.
  model: str
    The name of the model to use for upscaling.
  upscale_factor: int
    The factor by which to upscale the image.
```

----------------------------------------

TITLE: VideoDict Attributes
DESCRIPTION: Documentation for the VideoDict type, detailing its attributes for video representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VideoDict:
  mime_type: The MIME type of the video.
  uri: The URI of the video.
  video_bytes: The video data as bytes.
```

----------------------------------------

TITLE: ListTuningJobsResponseDict Attributes
DESCRIPTION: Details the attributes available within the ListTuningJobsResponseDict, including sdk_http_response and tuning_jobs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ListTuningJobsResponseDict:
  sdk_http_response: The HTTP response object from the SDK.
  tuning_jobs: A list of tuning jobs.
```

----------------------------------------

TITLE: GenerateContentConfigDict Parameters
DESCRIPTION: Details the various configuration options available for the GenerateContentConfigDict in the Google Generative AI Python SDK. These parameters control aspects of content generation such as candidate count, penalties, output tokens, and safety settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentConfigDict:
  candidate_count: Number of candidates to generate.
  frequency_penalty: Controls the frequency penalty for generated content.
  http_options: Options for HTTP requests.
  labels: Labels for the generation request.
  logprobs: Whether to include log probabilities in the response.
  max_output_tokens: Maximum number of tokens to generate in the output.
  media_resolution: Resolution for media content.
  model_selection_config: Configuration for model selection.
  presence_penalty: Controls the presence penalty for generated content.
  response_json_schema: JSON schema for the response.
  response_logprobs: Whether to include response log probabilities.
  response_mime_type: MIME type for the response.
  response_modalities: Modalities expected in the response.
  response_schema: Schema for the response.
  routing_config: Configuration for routing requests.
  safety_settings: Settings for safety filters.
  seed: Seed for random number generation.
  speech_config: Configuration for speech generation.
  stop_sequences: Sequences that stop generation.
  system_instruction: System-level instructions for the model.
  temperature: Controls the randomness of the output.
  thinking_config: Configuration for the model's thinking process.
  tool_config: Configuration for tools.
  tools: List of tools to use.
  top_k: Top-K sampling parameter.
  top_p: Top-P sampling parameter.
```

----------------------------------------

TITLE: Generate Image (Python)
DESCRIPTION: Generates images based on a text prompt using the Imagen model. Supports specifying the number of images, RAI reasons, and output format.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()
```

----------------------------------------

TITLE: Function Response Scheduling Options
DESCRIPTION: Defines the possible scheduling options for function responses.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FunctionResponseScheduling:
  INTERRUPT: FunctionResponseScheduling
    Interrupt the current process.
  SCHEDULING_UNSPECIFIED: FunctionResponseScheduling
    Default, unspecified scheduling.
  SILENT: FunctionResponseScheduling
    Respond silently without user interaction.
  WHEN_IDLE: FunctionResponseScheduling
    Respond when the system is idle.
```

----------------------------------------

TITLE: Live Music Source Metadata Types
DESCRIPTION: Defines types for source metadata in live music generation, including client content and generation configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicSourceMetadata, LiveMusicSourceMetadataDict

# Example usage for LiveMusicSourceMetadata:
source_metadata = LiveMusicSourceMetadata(
    client_content=LiveMusicServerContent(audio_chunks=[b'audio_data']),
    music_generation_config={'tempo': 120}
)

# Example usage for LiveMusicSourceMetadataDict:
source_metadata_dict = LiveMusicSourceMetadataDict(
    client_content={'audio_chunks': [b'audio_data']},
    music_generation_config={'tempo': 120}
)

```

----------------------------------------

TITLE: Generate Image
DESCRIPTION: Generates images based on textual descriptions using the Imagen model. This allows for creating visual content from natural language prompts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('image-generation@001')

prompt = "A futuristic cityscape at sunset, digital art"
response = model.generate_content(prompt)

# The response contains image data, typically a URL or bytes
# print(response.images[0].url)
```

----------------------------------------

TITLE: SubjectReferenceConfig and SubjectReferenceConfigDict
DESCRIPTION: Configuration for referencing a subject, allowing a description and type to be specified. The dictionary version provides a direct mapping.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SubjectReferenceConfig:
  subject_description: str
  subject_type: str

SubjectReferenceConfigDict:
  subject_description: str
  subject_type: str
```

----------------------------------------

TITLE: GroundingChunkMapsDict Attributes
DESCRIPTION: Explains the attributes of GroundingChunkMapsDict, mirroring GroundingChunkMaps with place_answer_sources, place_id, text, title, and uri.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsDict:
  place_answer_sources: Sources for place answers.
  place_id: Unique identifier for a place.
  text: Textual content associated with the map.
  title: Title of the map entry.
  uri: Uniform Resource Identifier for the map.
```

----------------------------------------

TITLE: TuningOperation and TuningOperationDict Attributes
DESCRIPTION: Details the attributes for TuningOperation and TuningOperationDict, which include status, error information, metadata, and HTTP response details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningOperation:
  done: Boolean indicating if the operation is complete.
  error: Information about any error that occurred during the operation.
  metadata: Metadata associated with the tuning operation.
  name: The name or identifier of the tuning operation.
  sdk_http_response: The HTTP response from the SDK.

TuningOperationDict:
  done: Boolean indicating if the operation is complete.
  error: Information about any error that occurred during the operation.
  metadata: Metadata associated with the tuning operation.
  name: The name or identifier of the tuning operation.
  sdk_http_response: The HTTP response from the SDK.
```

----------------------------------------

TITLE: GroundingChunk Attributes
DESCRIPTION: Lists the attributes of GroundingChunk, including maps, retrieved_context, and web for grounding information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunk:
  maps: Information related to maps.
  retrieved_context: Context retrieved from external sources.
  web: Information related to web content.
```

----------------------------------------

TITLE: UpscaleImageConfig
DESCRIPTION: Configuration for upscaling an image, including enhancement, HTTP options, and image preservation factor.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageConfig:
  Configuration for upscaling an image.
  Attributes:
    enhance_input_image: Whether to enhance the input image during upscaling.
    http_options: HTTP options for the image upscaling.
    image_preservation_factor: Factor to preserve image quality during upscaling.
```

----------------------------------------

TITLE: BatchJob Properties
DESCRIPTION: Details the various properties associated with a BatchJob, including its creation and update times, destination, display name, model, state, and any associated errors.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchJob:
  create_time: datetime
    Description: The timestamp when the batch job was created.
  dest: BatchJobDestination
    Description: The destination configuration for the batch job output.
  display_name: str
    Description: A user-friendly name for the batch job.
  end_time: datetime
    Description: The timestamp when the batch job finished.
  error: Error
    Description: Information about any errors encountered during the batch job.
  model: str
    Description: The name of the model used for the batch job.
  name: str
    Description: The unique identifier for the batch job.
  src: str
    Description: The source data for the batch job.
  start_time: datetime
    Description: The timestamp when the batch job started.
  state: str
    Description: The current state of the batch job (e.g., PENDING, RUNNING, SUCCEEDED, FAILED).
  update_time: datetime
    Description: The timestamp when the batch job was last updated.
```

----------------------------------------

TITLE: Function Calling with Python Functions
DESCRIPTION: Shows how to enable automatic function calling by passing a Python function directly as a tool. The model can then call this function based on the user's prompt.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Import GenAI and Types
DESCRIPTION: Imports the necessary genai and types modules from the google.genai library for interacting with Google's Generative AI APIs.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from googleimport genai
from google.genaiimport types
```

----------------------------------------

TITLE: List Batch Jobs with Pager (Asynchronous)
DESCRIPTION: Shows how to use the pager object with asynchronous operations for listing batch jobs. This includes accessing page details and fetching subsequent pages asynchronously.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: VeoHyperParametersDict API Documentation
DESCRIPTION: Documentation for VeoHyperParametersDict, a dictionary representation of hyperparameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VeoHyperParametersDict:
  epoch_count: The number of epochs for training.
  learning_rate_multiplier: Multiplier for the learning rate.
  tuning_task: The task for tuning.
```

----------------------------------------

TITLE: RagRetrievalConfig Fields
DESCRIPTION: Details the fields available for RagRetrievalConfig, including filter, hybrid search, ranking, and top_k settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfig:
  filter: Filter configuration for retrieval.
  hybrid_search: Configuration for hybrid search.
  ranking: Ranking configuration for retrieval.
  top_k: The number of top results to retrieve.
```

----------------------------------------

TITLE: List Tuning Jobs (Asynchronous)
DESCRIPTION: Asynchronously lists tuning jobs, allowing for non-blocking operations. It shows how to use async iterators and manage asynchronous pagination.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: RagRetrievalConfigDict Fields
DESCRIPTION: Details the fields available for RagRetrievalConfigDict, including filter, hybrid search, ranking, and top_k settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigDict:
  filter: Filter configuration for retrieval.
  hybrid_search: Configuration for hybrid search.
  ranking: Ranking configuration for retrieval.
  top_k: The number of top results to retrieve.
```

----------------------------------------

TITLE: Live Music Server Content Types
DESCRIPTION: Defines types for server content in live music generation, including audio chunks.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicServerContent, LiveMusicServerContentDict

# Example usage for LiveMusicServerContent:
server_content = LiveMusicServerContent(audio_chunks=[b'audio_data'])

# Example usage for LiveMusicServerContentDict:
server_content_dict = LiveMusicServerContentDict(audio_chunks=[b'audio_data'])

```

----------------------------------------

TITLE: ProductImage Fields
DESCRIPTION: Details the fields available for ProductImage, including the product image itself.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ProductImage:
  product_image: The product image data.
```

----------------------------------------

TITLE: UpdateModelConfigDict
DESCRIPTION: Dictionary-based configuration for updating a model, mirroring UpdateModelConfig.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpdateModelConfigDict:
  Dictionary-based configuration for updating a model.
  Attributes:
    default_checkpoint_id: The ID of the default checkpoint for the model.
    description: A description for the model.
    display_name: The display name of the model.
    http_options: HTTP options for the model update.
```

----------------------------------------

TITLE: ComputeTokensResponse Attributes
DESCRIPTION: Outlines the attributes of ComputeTokensResponse, including sdk_http_response and tokens_info.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ComputeTokensResponse:
  sdk_http_response: object
    The HTTP response from the SDK.
  tokens_info: object
    Information about the computed tokens.
```

----------------------------------------

TITLE: Use Tuned Model for Generation
DESCRIPTION: Generates content using a model that has been fine-tuned. It utilizes the endpoint of the tuned model to send a prompt and print the generated response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)
```

----------------------------------------

TITLE: SupervisedTuningSpecDict Attributes
DESCRIPTION: Details the attributes available within the SupervisedTuningSpecDict for configuring supervised tuning jobs. This includes settings for exporting checkpoints, hyperparameters, dataset URIs, and tuning mode.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import SupervisedTuningSpecDict

# Accessing attributes
export_last_checkpoint_only: bool = SupervisedTuningSpecDict.export_last_checkpoint_only
hyper_parameters: dict = SupervisedTuningSpecDict.hyper_parameters
training_dataset_uri: str = SupervisedTuningSpecDict.training_dataset_uri
tuning_mode: str = SupervisedTuningSpecDict.tuning_mode
validation_dataset_uri: str = SupervisedTuningSpecDict.validation_dataset_uri
```

----------------------------------------

TITLE: UpdateModelConfig
DESCRIPTION: Configuration for updating a model, including default checkpoint, description, display name, and HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpdateModelConfig:
  Configuration for updating a model.
  Attributes:
    default_checkpoint_id: The ID of the default checkpoint for the model.
    description: A description for the model.
    display_name: The display name of the model.
    http_options: HTTP options for the model update.
```

----------------------------------------

TITLE: TuningJobDict Attributes
DESCRIPTION: Provides access to attributes of the TuningJobDict, including tuning data statistics, update time, and VEO tuning specifications.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningJobDict:
  tuning_data_stats: Statistics related to the tuning data.
  update_time: The timestamp when the tuning job was last updated.
  veo_tuning_spec: The VEO tuning specification for the job.
```

----------------------------------------

TITLE: List Batch Jobs
DESCRIPTION: Lists all batch jobs associated with the client, with an option to control the page size for pagination. This allows for efficient retrieval of job lists.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

----------------------------------------

TITLE: GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict
DESCRIPTION: A dictionary representation of a review snippet for place answer sources, containing author details, URIs, and review content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsPlaceAnswerSourcesReviewSnippetDict:
  author_attribution: GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict
    Information about the author of the review.
  flag_content_uri: str
    The URI of the flagged content within the review.
  google_maps_uri: str
    The URI to the review on Google Maps.
  relative_publish_time_description: str
    A description of when the review was published relative to the current time.
  review: str
    The text content of the review.
```

----------------------------------------

TITLE: SubjectReferenceImage API Documentation
DESCRIPTION: Documentation for the SubjectReferenceImage type, including its configuration, reference ID, image, and type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SubjectReferenceImage:
  config: Configuration for the reference image.
  reference_id: The unique identifier for the reference image.
  reference_image: The reference image data.
  reference_type: The type of the reference image.
  subject_image_config: Configuration specific to the subject image.
```

----------------------------------------

TITLE: Pydantic Model Schema Support
DESCRIPTION: Demonstrates how to use Pydantic models to define the schema for JSON responses from the generative model. This allows for structured and validated data retrieval.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from google.generativeai import types


class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int


response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo,
    ),
)
print(response.text)
```

LANGUAGE: python
CODE:
```
from google.generativeai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            'required': [
                'name',
                'population',
                'capital',
                'continent',
                'gdp',
                'official_language',
                'total_area_sq_mi',
            ],
            'properties': {
                'name': {'type': 'STRING'},
                'population': {'type': 'INTEGER'},
                'capital': {'type': 'STRING'},
                'continent': {'type': 'STRING'},
                'gdp': {'type': 'INTEGER'},
                'official_language': {'type': 'STRING'},
                'total_area_sq_mi': {'type': 'INTEGER'},
            },
            'type': 'OBJECT',
        },
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Set Gemini Developer API Version to v1alpha
DESCRIPTION: Configures the client to use the 'v1alpha' API endpoints for the Gemini Developer API, enabling preview features.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

# Only run this block for Gemini Developer API
client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)
```

----------------------------------------

TITLE: AutomaticActivityDetectionDict Properties
DESCRIPTION: Details the properties available within the AutomaticActivityDetectionDict for configuring automatic activity detection. This includes parameters for padding, silence duration, and speech sensitivity.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AutomaticActivityDetectionDict:
  prefix_padding_ms: int
    Description: Padding in milliseconds to add to the beginning of the audio.
  silence_duration_ms: int
    Description: Duration in milliseconds to consider as silence.
  start_of_speech_sensitivity: float
    Description: Sensitivity level for detecting the start of speech (0.0 to 1.0).
```

----------------------------------------

TITLE: Distillation Hyperparameters
DESCRIPTION: Specifies the hyperparameters used for distillation, including adapter size, epoch count, and learning rate multiplier.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DistillationHyperParameters:
  adapter_size: The size of the adapter used in distillation.
  epoch_count: The number of epochs for the distillation training.
  learning_rate_multiplier: Multiplier for the learning rate during distillation.

DistillationHyperParametersDict:
  A dictionary representation of DistillationHyperParameters.
  adapter_size: The size of the adapter used in distillation.
  epoch_count: The number of epochs for the distillation training.
  learning_rate_multiplier: Multiplier for the learning rate during distillation.
```

----------------------------------------

TITLE: MultiSpeakerVoiceConfig Fields
DESCRIPTION: Configuration for multi-speaker voice settings, including individual speaker configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MultiSpeakerVoiceConfig:
  speaker_voice_configs: A list of configurations for each speaker.
```

----------------------------------------

TITLE: EditImageConfig Configuration
DESCRIPTION: Configuration options for editing images, including watermark addition, aspect ratio, steps, edit mode, guidance scale, HTTP options, RAI attributes, safety attributes, language, negative prompt, number of images, and output compression quality.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EditImageConfig:
  add_watermark: Whether to add a watermark to the edited image.
  aspect_ratio: The desired aspect ratio for the edited image.
  base_steps: The base number of steps for the image editing process.
  edit_mode: The mode for editing the image.
  guidance_scale: The guidance scale for the image editing process.
  http_options: HTTP options for the image editing request.
  include_rai_reason: Whether to include RAI (Responsible AI) reasons.
  include_safety_attributes: Whether to include safety attributes in the output.
  language: The language for any text-based image editing operations.
  negative_prompt: A prompt describing what to exclude from the edited image.
  number_of_images: The number of edited images to generate.
  output_compression_quality: The compression quality for the output images.
```

----------------------------------------

TITLE: ProductImageDict Fields
DESCRIPTION: Details the fields available for ProductImageDict, including the product image itself.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ProductImageDict:
  product_image: The product image data.
```

----------------------------------------

TITLE: Generate Content with Uploaded File
DESCRIPTION: Generates content by referencing an uploaded file. The file is uploaded using client.files.upload and then passed to generate_content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)
```

----------------------------------------

TITLE: ComputeTokensConfigDict Attributes
DESCRIPTION: Describes the ComputeTokensConfigDict type, which includes http_options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ComputeTokensConfigDict:
  http_options: dict
    HTTP options for the compute tokens request.
```

----------------------------------------

TITLE: IntervalDict Type Documentation
DESCRIPTION: Documentation for the IntervalDict type, a dictionary-like object for representing time intervals.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
IntervalDict:
  A dictionary-like object for time intervals.
  Attributes:
    start_time: The start time of the interval.
    end_time: The end time of the interval.
```

----------------------------------------

TITLE: LogprobsResult and LogprobsResultDict
DESCRIPTION: Contains log probabilities for generated candidates. Includes chosen and top candidates.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LogprobsResult:
  chosen_candidates: List[LogprobsResultCandidate]
    The candidates chosen by the model.
  top_candidates: List[LogprobsResultCandidate]
    The top candidates ranked by log probability.

LogprobsResultDict:
  chosen_candidates: List[LogprobsResultCandidateDict]
    The candidates chosen by the model.
  top_candidates: List[LogprobsResultCandidateDict]
    The top candidates ranked by log probability.
```

----------------------------------------

TITLE: Stream Image Content from Local File System
DESCRIPTION: Illustrates how to stream image content from the local file system by reading the image as bytes data and using the `from_bytes` class method. This method is suitable for images stored locally.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

YOUR_IMAGE_PATH = 'your_image_path'
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),
    ],
):
    print(chunk.text, end='')

```

----------------------------------------

TITLE: UpscaleImageResponseDict
DESCRIPTION: Dictionary representation of UpscaleImageResponse, containing generated images and SDK HTTP response information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageResponseDict:
  generated_images: list[bytes]
    A list of generated image data.
  sdk_http_response: dict
    The HTTP response details from the SDK.
```

----------------------------------------

TITLE: LiveClientToolResponse and LiveClientToolResponseDict
DESCRIPTION: Describes the structure for tool responses, including function responses. This applies to both object and dictionary representations of the tool response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientToolResponse:
  function_responses: A list of responses from tool functions.

LiveClientToolResponseDict:
  function_responses: A list of responses from tool functions.
```

----------------------------------------

TITLE: LiveConnectParametersDict Attributes
DESCRIPTION: Details the attributes available within the LiveConnectParametersDict, including 'config' and 'model'. These are used for setting parameters for live connections, similar to LiveConnectParameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveConnectParametersDict:
  config: Configuration settings for the live connection.
  model: Specifies the model to be used for the live connection.
```

----------------------------------------

TITLE: Set API Version to v1alpha for Gemini Developer API
DESCRIPTION: Configures the genai client to use the 'v1alpha' API endpoints for the Gemini Developer API. This allows access to preview features.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from googleimport genai
from google.genaiimport types

# To set the API version to v1alpha for the Gemini Developer API:
client = genai.Client(
    http_options=types.HttpOptions(api_version='v1alpha')
)
```

----------------------------------------

TITLE: LiveMusicPlaybackControl Enum
DESCRIPTION: Defines playback control options for live music, including PLAY, PAUSE, RESET_CONTEXT, and an unspecified default.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicPlaybackControl:
  PLAYBACK_CONTROL_UNSPECIFIED: int
    Default value. Should not be used.
  PLAY: int
    Starts or resumes music playback.
  PAUSE: int
    Pauses music playback.
  RESET_CONTEXT: int
    Resets the music generation context.
```

----------------------------------------

TITLE: ReplayFile Types
DESCRIPTION: Details the structure for ReplayFile and its dictionary representation, including interactions and replay ID.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ReplayFile:
  interactions: A list of interactions.
  replay_id: The unique identifier for the replay.

ReplayFileDict:
  interactions: A list of interactions (dictionary format).
  replay_id: The unique identifier for the replay (dictionary format).
```

----------------------------------------

TITLE: ModelSelectionConfig and ModelSelectionConfigDict Fields
DESCRIPTION: Specifies configuration for model selection, including feature preference.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ModelSelectionConfig:
  feature_selection_preference: Preference for feature selection.

ModelSelectionConfigDict:
  feature_selection_preference: Preference for feature selection.
```

----------------------------------------

TITLE: SlidingWindow Properties
DESCRIPTION: Details the properties available for the SlidingWindow type in the Python GenAI library. These properties define a sliding window for token management.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SlidingWindow:
  target_tokens: The target number of tokens for the window.
```

----------------------------------------

TITLE: FetchPredictOperationConfigDict
DESCRIPTION: Dictionary representation for FetchPredictOperationConfig, specifying HTTP options for fetching prediction operations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FetchPredictOperationConfigDict:
  http_options: dict
    HTTP options for fetching the prediction operation.
```

----------------------------------------

TITLE: Model and ModelDict Fields
DESCRIPTION: Details the attributes available for Model and ModelDict objects, including checkpoints, limits, and metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Model:
  checkpoints: List of available checkpoints for the model.
  default_checkpoint_id: The ID of the default checkpoint.
  description: A description of the model.
  display_name: The display name of the model.
  endpoints: List of endpoints for the model.
  input_token_limit: The maximum input tokens the model can handle.
  labels: Labels associated with the model.
  name: The name of the model.
  output_token_limit: The maximum output tokens the model can generate.
  supported_actions: Actions supported by the model.
  tuned_model_info: Information about tuned versions of the model.
  version: The version of the model.

ModelDict:
  checkpoints: List of available checkpoints for the model.
  default_checkpoint_id: The ID of the default checkpoint.
  description: A description of the model.
  display_name: The display name of the model.
  endpoints: List of endpoints for the model.
  input_token_limit: The maximum input tokens the model can handle.
  labels: Labels associated with the model.
  name: The name of the model.
  output_token_limit: The maximum output tokens the model can generate.
  supported_actions: Actions supported by the model.
  tuned_model_info: Information about tuned versions of the model.
  version: The version of the model.
```

----------------------------------------

TITLE: Handle API Errors
DESCRIPTION: Demonstrates how to catch and handle API errors raised by the SDK, specifically `APIError`. It shows how to access error codes and messages for debugging.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
try:
    client.models.generate_content(
        model="invalid-model-name",
        contents="What is your name?",
    )
except errors.APIError as e:
    print(e.code) # 404
    print(e.message)
```

----------------------------------------

TITLE: Set Vertex AI API Version to v1
DESCRIPTION: Configures the client to use the stable 'v1' API endpoints for Vertex AI.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    http_options=types.HttpOptions(api_version='v1')
)
```

----------------------------------------

TITLE: Asynchronous Non-Streaming Content Generation
DESCRIPTION: Shows how to generate content asynchronously without streaming using the `client.aio` interface. This is the asynchronous equivalent of `client.models.generate_content`.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)

```

----------------------------------------

TITLE: GenerateVideosOperation Methods
DESCRIPTION: Represents an ongoing video generation operation, providing access to the response and results.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateVideosOperation:
  response: GenerateVideosResponse
    The response from the video generation API.
  result(timeout: float | None = None) -> GenerateVideosResponse
    Waits for the operation to complete and returns the result.
    Parameters:
      timeout: Maximum time in seconds to wait for the operation.
  from_api_response(api_response: dict) -> GenerateVideosOperation
    Creates a GenerateVideosOperation from an API response dictionary.
```

----------------------------------------

TITLE: Manually Declare and Invoke Function for Function Calling
DESCRIPTION: Demonstrates how to manually declare a function using `types.FunctionDeclaration` and pass it as a tool to the model. It then shows how to receive a function call part in the response, invoke the function with its arguments, and send the function's response back to the model for a final answer.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)
print(response.function_calls[0])
```

LANGUAGE: python
CODE:
```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content


try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}


function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Generate Content (Async Non-Streaming) (Python)
DESCRIPTION: Shows how to asynchronously generate content from a model. This method is suitable for single responses and uses the 'gemini-2.0-flash-001' model.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)
```

----------------------------------------

TITLE: SupervisedTuningDatasetDistributionDatasetBucket Properties
DESCRIPTION: Represents a bucket within a dataset distribution, detailing the count of items and the range (left and right boundaries).

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDatasetDistributionDatasetBucket:
  count: int
    The number of items in this bucket.
  left: float
    The left boundary of the bucket.
  right: float
    The right boundary of the bucket.
```

----------------------------------------

TITLE: RecontextImageSource Types
DESCRIPTION: Defines the structure for RecontextImageSource and its dictionary representation, including person images, product images, and prompts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RecontextImageSource:
  person_image: The image of a person.
  product_images: A list of product images.
  prompt: The prompt associated with the image.

RecontextImageSourceDict:
  person_image: The image of a person (dictionary format).
  product_images: A list of product images (dictionary format).
  prompt: The prompt associated with the image (dictionary format).
```

----------------------------------------

TITLE: RAG Ranking Configuration
DESCRIPTION: Specifies ranking strategies for RAG results, including using an LLM for re-ranking or a dedicated rank service. This allows for more sophisticated ordering of retrieved documents.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigRanking:
  llm_ranker: RagRetrievalConfigRankingLlmRanker | None
    Configuration for LLM-based re-ranking.
  rank_service: RagRetrievalConfigRankingRankService | None
    Configuration for an external rank service.

RagRetrievalConfigRankingDict:
  llm_ranker: RagRetrievalConfigRankingLlmRankerDict | None
    Configuration for LLM-based re-ranking.
  rank_service: RagRetrievalConfigRankingRankServiceDict | None
    Configuration for an external rank service.
```

----------------------------------------

TITLE: FetchPredictOperationConfig
DESCRIPTION: Configuration for fetching prediction operation results, including HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FetchPredictOperationConfig:
  http_options: dict
    HTTP options for fetching the prediction operation.
```

----------------------------------------

TITLE: UserContent API Documentation
DESCRIPTION: Documentation for the UserContent type, including its parts and role attributes.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UserContent:
  parts: List of content parts.
  role: The role associated with the content.
```

----------------------------------------

TITLE: Distillation Specification
DESCRIPTION: Defines the overall specification for a distillation process, including model details, dataset URIs, and training parameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DistillationSpec:
  base_teacher_model: The base teacher model for distillation.
  hyper_parameters: Hyperparameters for the distillation process.
  pipeline_root_directory: The root directory for the distillation pipeline.
  student_model: The student model to be trained.
  training_dataset_uri: The URI of the training dataset.
  tuned_teacher_model_source: The source of the tuned teacher model.
  validation_dataset_uri: The URI of the validation dataset.
```

----------------------------------------

TITLE: PreferenceOptimizationDataStats Fields
DESCRIPTION: Details the fields available within the PreferenceOptimizationDataStats class, including dataset statistics and token distributions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationDataStats:
  tuning_dataset_example_count: int
    The number of examples in the tuning dataset.
  tuning_step_count: int
    The number of tuning steps performed.
  user_dataset_examples: int
    The number of examples provided by the user.
  user_input_token_distribution: dict
    A dictionary representing the distribution of input tokens.
  user_output_token_distribution: dict
    A dictionary representing the distribution of output tokens.
```

----------------------------------------

TITLE: Delete Resource Job Management
DESCRIPTION: Provides API documentation for managing delete resource jobs, including checking job status, retrieving errors, and accessing associated HTTP responses.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DeleteResourceJob:
  done: Boolean indicating if the job is completed.
  error: Information about any error that occurred during the job.
  name: The name or identifier of the delete resource job.
  sdk_http_response: The HTTP response object from the SDK.

DeleteResourceJobDict:
  A dictionary representation of the DeleteResourceJob.
  done: Boolean indicating if the job is completed.
  error: Information about any error that occurred during the job.
  name: The name or identifier of the delete resource job.
  sdk_http_response: The HTTP response object from the SDK.
```

----------------------------------------

TITLE: List Batch Prediction Jobs
DESCRIPTION: Lists all batch prediction jobs associated with the client, with support for pagination. It demonstrates how to iterate through jobs and access their details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)
```

----------------------------------------

TITLE: VertexRagStoreRagResourceDict Attributes
DESCRIPTION: Documentation for the VertexRagStoreRagResourceDict type, detailing its attributes for RAG resource configuration.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexRagStoreRagResourceDict:
  rag_corpus: The RAG corpus to use.
  rag_file_ids: A list of file IDs to include in the RAG resource.
```

----------------------------------------

TITLE: GroundingChunkWebDict Attributes
DESCRIPTION: Details the attributes of the GroundingChunkWebDict, a dictionary representation of web grounding content. It includes the domain, title, and URI of the web resource.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkWebDict:
  domain: The domain of the web resource.
  title: The title of the web page.
  uri: The URI of the web page.
```

----------------------------------------

TITLE: Upload File for Batch Prediction
DESCRIPTION: Uploads a local file (e.g., a JSONL file containing requests) to the Gemini Developer API. This file can then be used as a source for creating batch prediction jobs.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
# Upload a file to Gemini Developer API
file_name = client.files.upload(
    file='myrequest.json',
    config=types.UploadFileConfig(display_name='test_json'),
)
# Create a batch job with file name
```

----------------------------------------

TITLE: GroundingChunkDict Attributes
DESCRIPTION: Details the attributes of GroundingChunkDict, similar to GroundingChunk, covering maps, retrieved_context, and web.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkDict:
  maps: Information related to maps.
  retrieved_context: Context retrieved from external sources.
  web: Information related to web content.
```

----------------------------------------

TITLE: SubjectReferenceImageDict API Documentation
DESCRIPTION: Documentation for the SubjectReferenceImageDict type, mirroring SubjectReferenceImage attributes for dictionary-based access.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SubjectReferenceImageDict:
  config: Configuration for the reference image.
  reference_id: The unique identifier for the reference image.
  reference_image: The reference image data.
  reference_type: The type of the reference image.
```

----------------------------------------

TITLE: TuningJobDict Attributes
DESCRIPTION: Defines the structure and attributes for a TuningJob dictionary, encompassing details like base model, creation time, custom base model, descriptions, various tuning specifications (distillation, preference optimization), encryption, error information, experiment details, labels, output URI, partner model tuning, pipeline job status, pre-tuned model information, PZI/PZS satisfaction, SDK HTTP response, service account, start/update times, state, and tuned model details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningJobDict:
  base_model: The base model used for tuning.
  create_time: The timestamp when the tuning job was created.
  custom_base_model: The custom base model used, if any.
  description: A description of the tuning job.
  distillation_spec: Configuration for distillation tuning.
  encryption_spec: Configuration for data encryption.
  end_time: The timestamp when the tuning job ended.
  error: Information about any errors encountered during the tuning job.
  experiment: The experiment associated with the tuning job.
  labels: Key-value pairs for labeling the tuning job.
  name: The unique name of the tuning job.
  output_uri: The URI where the output of the tuning job is stored.
  partner_model_tuning_spec: Configuration for partner model tuning.
  pipeline_job: Information about the associated pipeline job.
  pre_tuned_model: The pre-tuned model used as a starting point.
  preference_optimization_spec: Configuration for preference optimization.
  satisfies_pzi: Boolean indicating if PZI requirements are met.
  satisfies_pzs: Boolean indicating if PZS requirements are met.
  sdk_http_response: The HTTP response from the SDK call.
  service_account: The service account used for the tuning job.
  start_time: The timestamp when the tuning job started.
  state: The current state of the tuning job.
  supervised_tuning_spec: Configuration for supervised tuning.
  tuned_model: The name of the tuned model.
  tuned_model_display_name: A user-friendly display name for the tuned model.
```

----------------------------------------

TITLE: VideoMetadata Attributes
DESCRIPTION: Documentation for the VideoMetadata type, outlining its attributes for video metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VideoMetadata:
  end_offset: The end offset of the video segment.
  fps: The frames per second of the video.
```

----------------------------------------

TITLE: GenAI Chats Module Documentation
DESCRIPTION: Documentation for the genai.chats module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.chats
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: GenerateContentResponsePromptFeedback Attributes
DESCRIPTION: Details the attributes of GenerateContentResponsePromptFeedback, including block reason, block reason message, and safety ratings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponsePromptFeedback:
  block_reason: The reason the response was blocked.
  block_reason_message: A message explaining the block reason.
  safety_ratings: Safety ratings for the response.
```

----------------------------------------

TITLE: LiveMusicConnectParameters Attributes
DESCRIPTION: Details the 'model' attribute for LiveMusicConnectParameters, used for connection parameters in live music scenarios.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicConnectParameters:
  model: Specifies the model to be used for live music connection.
```

----------------------------------------

TITLE: List Base Models (Asynchronous)
DESCRIPTION: Retrieves a list of available base models asynchronously. This is useful for discovering which models can be used for generation tasks.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

async def list_models():
    models = await genai.list_models()
    for model in models:
        print(model.name)

# To run this:
# import asyncio
# asyncio.run(list_models())
```

----------------------------------------

TITLE: SearchEntryPointDict Properties
DESCRIPTION: Details the properties available for the SearchEntryPointDict type in the Python GenAI library. These properties are dictionary representations of search entry points.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SearchEntryPointDict:
  rendered_content: The rendered content of the search result.
  sdk_blob: Blob data associated with the SDK.
```

----------------------------------------

TITLE: Function Calling
DESCRIPTION: Enables generative models to call external functions based on user prompts. This allows for dynamic interaction and task execution.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai
import json

# Define a tool with a function
my_tool = {
    "function_declarations": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]
}

# Configure the model with the tool
model = genai.GenerativeModel('gemini-pro', tools=[my_tool])

# Example of invoking the function (simulated)
# response = model.generate_content("What's the weather in Boston?")
# print(response.candidates[0].content.parts[0].function_call)
```

----------------------------------------

TITLE: Compute Tokens (Vertex AI Only)
DESCRIPTION: Shows how to compute tokens, which is a feature exclusively supported in Vertex AI. This method is used for more advanced token analysis.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)

```

----------------------------------------

TITLE: ListCachedContentsConfig Fields
DESCRIPTION: Defines configuration options for listing cached content, including http_options for customizing HTTP requests, page_size for limiting results, and page_token for pagination.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListCachedContentsConfig:
    http_options: dict
    page_size: int
    page_token: str
```

----------------------------------------

TITLE: VoiceConfig Types
DESCRIPTION: Defines types for voice configuration, specifically for prebuilt voice settings. This allows for customization of text-to-speech output.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VoiceConfig:
  prebuilt_voice_config: Configuration for a prebuilt voice.

VoiceConfigDict:
  prebuilt_voice_config: Configuration for a prebuilt voice.
```

----------------------------------------

TITLE: GroundingChunkWeb Attributes
DESCRIPTION: Details the attributes of the GroundingChunkWeb object, representing web-based grounding content. It includes the domain, title, and URI of the web resource.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkWeb:
  domain: The domain of the web resource.
  title: The title of the web page.
  uri: The URI of the web page.
```

----------------------------------------

TITLE: ImagePromptLanguage Enum
DESCRIPTION: Enumeration for specifying the language of image prompts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ImagePromptLanguage:
  auto: Automatically detect the language.
  en: English.
  es: Spanish.
  hi: Hindi.
  ja: Japanese.
  ko: Korean.
  pt: Portuguese.
  zh: Chinese.
```

----------------------------------------

TITLE: RAG LLM Ranker Configuration
DESCRIPTION: Details the configuration for using a Large Language Model (LLM) to re-rank search results. Specifies the model name to be used for the ranking task.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigRankingLlmRanker:
  model_name: str
    The name of the LLM model to use for ranking.

RagRetrievalConfigRankingLlmRankerDict:
  model_name: str
    The name of the LLM model to use for ranking.
```

----------------------------------------

TITLE: EditImageConfig and EditImageConfigDict Parameters
DESCRIPTION: Details the parameters available for configuring image editing operations using EditImageConfig and EditImageConfigDict. These include output settings, safety filters, and generation parameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EditImageConfig:
  output_gcs_uri: str
    The Cloud Storage URI for the output image.
  output_mime_type: str
    The MIME type of the output image (e.g., 'image/png').
  person_generation: bool
    Whether to enable person generation.
  safety_filter_level: int
    The level of safety filtering to apply.
  seed: int
    The seed for random number generation.

EditImageConfigDict:
  add_watermark: bool
    Whether to add a watermark to the generated image.
  aspect_ratio: str
    The desired aspect ratio of the output image (e.g., '16:9').
  base_steps: int
    The base number of diffusion steps for image generation.
  edit_mode: EditMode
    The mode of editing to perform (e.g., INPAINT_INSERTION).
  guidance_scale: float
    The guidance scale for image generation.
  http_options: dict
    Additional HTTP options for the request.
  include_rai_reason: bool
    Whether to include RAI (Responsible AI) reasons in the output.
  include_safety_attributes: bool
    Whether to include safety attributes in the output.
  language: str
    The language for any text-based image generation.
  negative_prompt: str
    A prompt describing what to avoid in the generated image.
  number_of_images: int
    The number of images to generate.
  output_compression_quality: int
    The compression quality for the output image.
  output_gcs_uri: str
    The Cloud Storage URI for the output image.
  output_mime_type: str
    The MIME type of the output image (e.g., 'image/png').
  person_generation: bool
    Whether to enable person generation.
  safety_filter_level: int
    The level of safety filtering to apply.
  seed: int
    The seed for random number generation.
```

----------------------------------------

TITLE: EmbedContentConfig and EmbedContentConfigDict
DESCRIPTION: Configuration options for embedding content. These settings control aspects like automatic truncation, HTTP options, MIME type, output dimensionality, task type, and title.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EmbedContentConfig:
  auto_truncate: Whether to automatically truncate content if it exceeds the model's limit.
  http_options: Custom HTTP options for the request.
  mime_type: The MIME type of the content being embedded.
  output_dimensionality: The desired dimensionality of the output embeddings.
  task_type: The type of task for which embeddings are generated (e.g., 'retrieval_query').
  title: An optional title for the content.

EmbedContentConfigDict:
  auto_truncate: Whether to automatically truncate content if it exceeds the model's limit.
  http_options: Custom HTTP options for the request.
  mime_type: The MIME type of the content being embedded.
  output_dimensionality: The desired dimensionality of the output embeddings.
  task_type: The type of task for which embeddings are generated (e.g., 'retrieval_query').
  title: An optional title for the content.
```

----------------------------------------

TITLE: File Operations Configuration
DESCRIPTION: Configuration options for deleting files, including HTTP options and response details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DeleteFileConfig:
  http_options: dict

DeleteFileConfigDict:
  http_options: dict

DeleteFileResponse:
  (No specific fields documented)

DeleteFileResponseDict:
  (No specific fields documented)
```

----------------------------------------

TITLE: ListFilesResponseDict Fields
DESCRIPTION: Provides the dictionary representation of ListFilesResponse, exposing files, next_page_token, and sdk_http_response for accessing file listing results.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListFilesResponseDict:
    files: list
    next_page_token: str
    sdk_http_response: google.api_core.http_response.HTTPResponse
```

----------------------------------------

TITLE: ListBatchJobsResponseDict Fields
DESCRIPTION: Details the fields available in the ListBatchJobsResponseDict, including next_page_token for pagination and sdk_http_response for HTTP communication details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListBatchJobsResponseDict:
    next_page_token: str
    sdk_http_response: google.api_core.http_response.HTTPResponse
```

----------------------------------------

TITLE: GroundingSupport Object
DESCRIPTION: Represents support information related to grounding operations. This object may contain details about how grounding was achieved or verified.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingSupport:
  # Attributes specific to grounding support would be detailed here.
```

----------------------------------------

TITLE: Model Tuning
DESCRIPTION: Enables fine-tuning of generative models on custom datasets to improve performance for specific tasks or domains. This includes tuning, managing, and using tuned models.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

# Tune a model (requires a dataset)
# tuning_job = genai.tune_model(training_data='gs://your-bucket/training_data.jsonl')
# print(f'Tuning job started: {tuning_job.name}')

# Get tuning job status
# job_status = genai.get_tuning_job(tuning_job.name)
# print(f'Tuning job status: {job_status.state}')

# Use a tuned model
# tuned_model_name = 'tunedModels/your-tuned-model-id'
# model = genai.GenerativeModel(tuned_model_name)
# response = model.generate_content('Generate text using the tuned model.')
# print(response.text)
```

----------------------------------------

TITLE: Python GenAI: Manually Declare and Invoke Function for Function Calling
DESCRIPTION: Shows how to manually declare a function with its schema and pass it as a tool. The response will contain function call parts that can then be invoked.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)
print(response.function_calls[0])
```

----------------------------------------

TITLE: Authentication Type Enumeration
DESCRIPTION: Enumerates the different types of authentication supported by the GenAI library. This includes API key, service account, HTTP basic, OAuth, OIDC, and no authentication.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AuthType:
  API_KEY_AUTH: str
    Authentication using an API key.
  AUTH_TYPE_UNSPECIFIED: str
    Unspecified authentication type.
  GOOGLE_SERVICE_ACCOUNT_AUTH: str
    Authentication using a Google service account.
  HTTP_BASIC_AUTH: str
    HTTP Basic authentication.
  NO_AUTH: str
    No authentication required.
  OAUTH: str
    OAuth authentication.
  OIDC_AUTH: str
    OpenID Connect (OIDC) authentication.
```

----------------------------------------

TITLE: Google Generative AI Batches API
DESCRIPTION: Manages batch operations within the Google Generative AI API. Includes methods for creating, deleting, retrieving, and listing batches.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.batches.Batches:
  create(): Creates a new batch.
  delete(batch_id: str): Deletes a batch by its ID.
  get(batch_id: str): Retrieves a specific batch by its ID.
  list(): Lists all available batches.
```

----------------------------------------

TITLE: Handle API Errors
DESCRIPTION: Demonstrates how to catch and handle API errors raised by the model using the SDK's APIError class. It shows how to access error codes and messages for debugging.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
try:
    client.models.generate_content(
        model="invalid-model-name",
        contents="What is your name?",
    )
except errors.APIError as e:
    print(e.code) # 404
    print(e.message)
```

----------------------------------------

TITLE: GoogleTypeDateDict Attributes
DESCRIPTION: Explains the attributes of GoogleTypeDateDict, mirroring GoogleTypeDate with day, month, and year components.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GoogleTypeDateDict:
  day: The day of the month (1-31).
  month: The month of the year (1-12).
  year: The year.
```

----------------------------------------

TITLE: LogprobsResultCandidate and LogprobsResultCandidateDict
DESCRIPTION: Represents a single candidate with its associated log probability and token information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LogprobsResultCandidate:
  log_probability: float
    The log probability of the candidate.
  token: str
    The token generated for this candidate.
  token_id: int
    The ID of the token generated.

LogprobsResultCandidateDict:
  log_probability: float
    The log probability of the candidate.
  token: str
    The token generated for this candidate.
  token_id: int
    The ID of the token generated.
```

----------------------------------------

TITLE: Stream Image Content from Google Cloud Storage
DESCRIPTION: Demonstrates how to stream image content from Google Cloud Storage using the `from_uri` class method. This is useful for processing images stored in GCS with the generative AI model.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')

```

----------------------------------------

TITLE: GenerateContentResponseDict Attributes
DESCRIPTION: Describes the attributes of the GenerateContentResponseDict, which includes candidates, creation time, model version, prompt feedback, response ID, HTTP response details, and usage metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponseDict:
  candidates: A list of candidate responses.
  create_time: The timestamp when the response was created.
  model_version: The version of the model used.
  prompt_feedback: Feedback related to the prompt.
  response_id: The unique identifier for the response.
  sdk_http_response: The HTTP response object from the SDK.
  usage_metadata: Metadata about token usage.
```

----------------------------------------

TITLE: Automatic Activity Detection Configuration
DESCRIPTION: Configures automatic activity detection parameters for speech or audio processing. This includes settings for disabling detection, sensitivity, and silence/padding durations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AutomaticActivityDetection:
  disabled: bool
    Disables automatic activity detection.
  end_of_speech_sensitivity: float
    Sensitivity level for detecting the end of speech.
  prefix_padding_ms: int
    Padding in milliseconds before the start of speech.
  silence_duration_ms: int
    Duration of silence in milliseconds to detect activity.
  start_of_speech_sensitivity: float
    Sensitivity level for detecting the start of speech.

AutomaticActivityDetectionDict:
  disabled: bool
    Disables automatic activity detection.
  end_of_speech_sensitivity: float
    Sensitivity level for detecting the end of speech.
  prefix_padding_ms: int
    Padding in milliseconds before the start of speech.
  silence_duration_ms: int
    Duration of silence in milliseconds to detect activity.
  start_of_speech_sensitivity: float
    Sensitivity level for detecting the start of speech.
```

----------------------------------------

TITLE: ListCachedContentsResponse Fields
DESCRIPTION: Outlines the fields in ListCachedContentsResponse, including the list of cached_contents, next_page_token for pagination, and sdk_http_response for HTTP details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListCachedContentsResponse:
    cached_contents: list
    next_page_token: str
    sdk_http_response: google.api_core.http_response.HTTPResponse
```

----------------------------------------

TITLE: Generate Content (Synchronous Streaming)
DESCRIPTION: Generates text content from a prompt and streams the response back token by token in real-time. This is useful for interactive applications.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('gemini-pro')

# Generate content and stream the response
response = model.generate_content('Tell me a short story about a brave knight.', stream=True)

for chunk in response:
    print(chunk.text, end='')
print()
```

----------------------------------------

TITLE: GenAI Types Module Documentation
DESCRIPTION: Documentation for the genai.types module, listing its members, undocumented members, and inheritance.

SOURCE: https://googleapis.github.io/python-genai/_sources/genai.rst

LANGUAGE: python
CODE:
```
.. automodule:: genai.types
   :members:
   :undoc-members:
   :show-inheritance:
```

----------------------------------------

TITLE: Python GenAI Types Documentation
DESCRIPTION: This section details the various data types available in the Python GenAI library, including their attributes and methods. It serves as a reference for developers using the library.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MultiSpeakerVoiceConfigDict:
  speaker_voice_configs: Configuration for speaker voices.

MusicGenerationMode:
  DIVERSITY: Mode for music generation focused on diversity.
  MUSIC_GENERATION_MODE_UNSPECIFIED: Unspecified music generation mode.
  QUALITY: Mode for music generation focused on quality.
  VOCALIZATION: Mode for music generation focused on vocalization.

Operation:
  done: Boolean indicating if the operation is complete.
  error: Error details if the operation failed.
  metadata: Metadata associated with the operation.
  name: The name of the operation.
  from_api_response(response: dict) -> 'Operation': Creates an Operation object from an API response.

Outcome:
  OUTCOME_DEADLINE_EXCEEDED: Operation timed out.
  OUTCOME_FAILED: Operation failed.
  OUTCOME_OK: Operation succeeded.
  OUTCOME_UNSPECIFIED: Unspecified outcome.

Part:
  code_execution_result: Result of code execution.
  executable_code: Code that can be executed.
  file_data: Data associated with a file.
  function_call: A function call.
  function_response: A response from a function call.
  inline_data: Inline data.
  text: Text content.
  video_metadata: Metadata for video content.
  thought: A thought or reasoning step.
  thought_signature: Signature for a thought.
  from_bytes(data: bytes, mime_type: str) -> 'Part': Creates a Part object from bytes.
  from_code_execution_result(code: str, result: str) -> 'Part': Creates a Part object from code execution results.
  from_executable_code(code: str) -> 'Part': Creates a Part object from executable code.
  from_function_call(function_call: dict) -> 'Part': Creates a Part object from a function call.
  from_function_response(function_response: dict) -> 'Part': Creates a Part object from a function response.
  from_text(text: str) -> 'Part': Creates a Part object from text.
  from_uri(uri: str, mime_type: str) -> 'Part': Creates a Part object from a URI.

PartDict:
  code_execution_result: Result of code execution.
  executable_code: Code that can be executed.
  file_data: Data associated with a file.
```

----------------------------------------

TITLE: FinishReason Enumerations
DESCRIPTION: Lists the possible reasons why a model's generation might have stopped. This includes safety-related reasons, token limits, and other specific conditions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FinishReason:
  BLOCKLIST: Generation stopped due to content on a blocklist.
  FINISH_REASON_UNSPECIFIED: The finish reason is unspecified.
  IMAGE_SAFETY: Generation stopped due to image safety concerns.
  LANGUAGE: Generation stopped due to language policy violations.
  MALFORMED_FUNCTION_CALL: Generation stopped due to an improperly formatted function call.
  MAX_TOKENS: Generation stopped because the maximum token limit was reached.
  OTHER: Generation stopped for an unspecified reason.
  PROHIBITED_CONTENT: Generation stopped due to prohibited content.
  RECITATION: Generation stopped due to citation policy violations.
  SAFETY: Generation stopped due to general safety policy violations.
  SPII: Generation stopped due to Sensitive Personally Identifiable Information (SPII).
  STOP: Generation stopped because a natural stop sequence was encountered.
```

----------------------------------------

TITLE: ControlReferenceType API
DESCRIPTION: API documentation for ControlReferenceType, listing available control types such as CANNY, DEFAULT, FACE_MESH, and SCRIBBLE.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ControlReferenceType:
  CONTROL_TYPE_CANNY: str
    Canny edge detection control type.
  CONTROL_TYPE_DEFAULT: str
    Default control type.
  CONTROL_TYPE_FACE_MESH: str
    Face mesh control type.
  CONTROL_TYPE_SCRIBBLE: str
    Scribble control type.
```

----------------------------------------

TITLE: MaskReferenceConfig and MaskReferenceConfigDict
DESCRIPTION: Configuration for mask reference, including dilation, mode, and segmentation classes.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MaskReferenceConfig:
  mask_dilation: int
    The dilation factor for the mask.
  mask_mode: str
    The mode for the mask (e.g., 'BOX', 'POINT').
  segmentation_classes: List[str]
    A list of segmentation classes.

MaskReferenceConfigDict:
  mask_dilation: int
    The dilation factor for the mask.
  mask_mode: str
    The mode for the mask (e.g., 'BOX', 'POINT').
  segmentation_classes: List[str]
    A list of segmentation classes.
```

----------------------------------------

TITLE: RAG Rank Service Configuration
DESCRIPTION: Specifies the configuration for an external rank service used in RAG. This includes the model name of the service to be invoked for ranking.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigRankingRankService:
  model_name: str
    The name of the model used by the rank service.

RagRetrievalConfigRankingRankServiceDict:
  model_name: str
    The name of the model used by the rank service.
```

----------------------------------------

TITLE: TuningDatasetDict Fields
DESCRIPTION: Details the fields within the TuningDatasetDict, a dictionary representation of a tuning dataset.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningDatasetDict:
  examples: A list of tuning examples.
  gcs_uri: The Google Cloud Storage URI for the dataset.
  vertex_dataset_resource: The Vertex AI dataset resource identifier.
```

----------------------------------------

TITLE: MediaResolution API
DESCRIPTION: Specifies the resolution levels for media, including low, medium, high, and unspecified.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MediaResolution:
  MEDIA_RESOLUTION_HIGH: High resolution.
  MEDIA_RESOLUTION_LOW: Low resolution.
  MEDIA_RESOLUTION_MEDIUM: Medium resolution.
  MEDIA_RESOLUTION_UNSPECIFIED: Resolution is not specified.
```

----------------------------------------

TITLE: GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict
DESCRIPTION: A dictionary representation of author attribution information for grounding chunks, mirroring the GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution structure.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsPlaceAnswerSourcesAuthorAttributionDict:
  display_name: str
    The display name of the author.
  photo_uri: str
    The URI of the author's photo.
  uri: str
    The URI associated with the author.
```

----------------------------------------

TITLE: Modality API
DESCRIPTION: Defines the core modalities supported by the system, such as text, image, and audio.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Modality:
  AUDIO: Represents audio data.
  IMAGE: Represents image data.
  MODALITY_UNSPECIFIED: The modality is not specified.
  TEXT: Represents text data.
```

----------------------------------------

TITLE: UpdateCachedContentConfigDict
DESCRIPTION: Dictionary-based configuration for updating cached content, mirroring UpdateCachedContentConfig.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpdateCachedContentConfigDict:
  Dictionary-based configuration for updating cached content.
  Attributes:
    expire_time: The expiration time for the cached content.
    http_options: HTTP options for the cache.
    ttl: Time-to-live for the cached content.
```

----------------------------------------

TITLE: Chat: Send Message (Synchronous Streaming)
DESCRIPTION: Illustrates sending a message to a chat model and receiving a streamed response. This is useful for providing real-time feedback to the user as the model generates its output.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')  # end='' is optional, for demo purposes.
```

----------------------------------------

TITLE: SlidingWindowDict Properties
DESCRIPTION: Details the properties available for the SlidingWindowDict type in the Python GenAI library. These properties are dictionary representations of sliding window configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SlidingWindowDict:
  target_tokens: The target number of tokens for the window.
```

----------------------------------------

TITLE: ExecutableCode Type Attributes
DESCRIPTION: Represents executable code with its content and language. Includes dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import ExecutableCode, ExecutableCodeDict

# Example usage of ExecutableCode
code_snippet = ExecutableCode(code='print("Hello, world!")', language='python')
print(f"Code: {code_snippet.code}")
print(f"Language: {code_snippet.language}")

# Example usage of ExecutableCodeDict
code_dict = ExecutableCodeDict({'code': 'console.log("Hi!")', 'language': 'javascript'})
print(f"Code Dict: {code_dict['code']}")
print(f"Code Dict Language: {code_dict['language']}")
```

----------------------------------------

TITLE: GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution
DESCRIPTION: Represents author attribution information for grounding chunks, including display name, photo URI, and a general URI.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMapsPlaceAnswerSourcesAuthorAttribution:
  display_name: str
    The display name of the author.
  photo_uri: str
    The URI of the author's photo.
  uri: str
    The URI associated with the author.
```

----------------------------------------

TITLE: Python GenAI: Invoking Function and Passing Response Back to Model
DESCRIPTION: Illustrates the process of receiving a function call from the model, invoking the corresponding function with provided arguments, handling potential errors, and then sending the function's response back to the model for further processing.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')]
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content


try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}


function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[
        user_prompt_content,
        function_call_content,
        function_response_content,
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)
```

----------------------------------------

TITLE: Generate Content (Asynchronous Streaming)
DESCRIPTION: Generates text content from a prompt asynchronously and streams the response token by token. This combines the benefits of async operations and real-time feedback.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai
import asyncio

async def generate_content_stream_async():
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content_async('Write a poem about the stars.', stream=True)
    async for chunk in response:
        print(chunk.text, end='')
    print()

# To run this:
# asyncio.run(generate_content_stream_async())
```

----------------------------------------

TITLE: Python GenAI Types and Enums
DESCRIPTION: This section details various types and enumerations used within the Python GenAI library, including finish reasons, function call structures, and function calling configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FinishReason.UNEXPECTED_TOOL_CALL:
  Represents an unexpected tool call as a finish reason.

FunctionCall:
  Represents a function call made by the model.
  - args: Dictionary of arguments for the function call.
  - id: Unique identifier for the function call.
  - name: The name of the function being called.

FunctionCallDict:
  A dictionary representation of a FunctionCall.
  - args: Dictionary of arguments for the function call.
  - id: Unique identifier for the function call.
  - name: The name of the function being called.

FunctionCallingConfig:
  Configuration for enabling and controlling function calling.
  - allowed_function_names: List of function names that are allowed to be called.
  - mode: The mode for function calling (e.g., ANY, AUTO, NONE).

FunctionCallingConfigDict:
  A dictionary representation of a FunctionCallingConfig.
  - allowed_function_names: List of function names that are allowed to be called.
  - mode: The mode for function calling (e.g., ANY, AUTO, NONE).

FunctionCallingConfigMode:
  Enum for different function calling modes.
  - ANY: Allows any function to be called.
  - AUTO: Automatically determines which functions to call.
  - MODE_UNSPECIFIED: The function calling mode is not specified.
  - NONE: Disables function calling.

FunctionDeclaration:
  Defines a function that the model can call.
  - behavior: Description of the function's behavior.
  - description: A detailed description of the function.
  - name: The name of the function.
  - parameters: Schema defining the function's parameters.
  - parameters_json_schema: JSON schema for the function's parameters.
  - response: Schema defining the function's response.
  - response_json_schema: JSON schema for the function's response.
  - from_callable(func: callable, name: str | None = None, description: str | None = None, ...):
    Creates a FunctionDeclaration from a Python callable.
  - from_callable_with_api_option(func: callable, name: str | None = None, description: str | None = None, ...):
    Creates a FunctionDeclaration from a Python callable with API options.
```

----------------------------------------

TITLE: Generate Content with Text
DESCRIPTION: Generates content using the Gemini API with a text prompt. The response object contains the generated text.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)
```

----------------------------------------

TITLE: Streaming Image Content Generation from URI
DESCRIPTION: Generates content based on an image provided via a Google Cloud Storage URI. The response is streamed back in chunks.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[
        'What is this image about?',
        types.Part.from_uri(
            file_uri='gs://generativeai-downloads/images/scones.jpg',
            mime_type='image/jpeg',
        ),
    ],
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: PartDict Attributes
DESCRIPTION: Provides access to different types of content within a PartDict, including function calls, inline data, text, and video metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PartDict:
  function_call: Accesses function call information.
  function_response: Accesses function response information.
  inline_data: Accesses inline data.
  text: Accesses text content.
  thought: Accesses thought process information.
  thought_signature: Accesses thought signature information.
  video_metadata: Accesses video metadata.
```

----------------------------------------

TITLE: PreferenceOptimizationSpec Fields
DESCRIPTION: Defines the specification for preference optimization, primarily containing the hyper-parameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationSpec:
  hyper_parameters: PreferenceOptimizationHyperParameters
    The hyperparameters to use for preference optimization.
```

----------------------------------------

TITLE: MediaModality API
DESCRIPTION: Defines the supported media modalities for input and output, including text, image, audio, video, document, and unspecified.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MediaModality:
  AUDIO: Represents audio data.
  DOCUMENT: Represents document data.
  IMAGE: Represents image data.
  MODALITY_UNSPECIFIED: The modality is not specified.
  TEXT: Represents text data.
  VIDEO: Represents video data.
```

----------------------------------------

TITLE: Set API Version to v1 for Vertex AI
DESCRIPTION: Configures the genai client to use the stable 'v1' API endpoints for Vertex AI by specifying http_options. This ensures compatibility with stable API versions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from googleimport genai
from google.genaiimport types

client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    http_options=types.HttpOptions(api_version='v1')
)
```

----------------------------------------

TITLE: VertexAISearchDict Attributes
DESCRIPTION: Documentation for the VertexAISearchDict type, specifically its max_results attribute.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexAISearchDict.max_results:
  description: The maximum number of results to return.
  type: int
```

----------------------------------------

TITLE: GenerateVideosResponse Fields
DESCRIPTION: Contains the results of a video generation request, including generated videos and filtering information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateVideosResponse:
  generated_videos: list[GeneratedVideo]
    A list of generated videos.
  rai_media_filtered_count: int
    The number of media items filtered by RAI.
  rai_media_filtered_reasons: list[str]
    The reasons why media items were filtered by RAI.
```

----------------------------------------

TITLE: JobErrorDict Type Documentation
DESCRIPTION: Documentation for the JobErrorDict type, a dictionary-like object for representing job errors.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
JobErrorDict:
  A dictionary-like object for job errors.
  Attributes:
    code: The error code.
    details: Additional details about the error.
    message: A human-readable error message.
```

----------------------------------------

TITLE: GroundingMetadata Attributes
DESCRIPTION: Details the attributes of the GroundingMetadata object, which provides comprehensive information about the grounding process. It includes context tokens, grounding chunks, supports, retrieval metadata, queries, search entry points, and web search queries.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingMetadata:
  google_maps_widget_context_token: Context token for Google Maps widget.
  grounding_chunks: A list of grounding chunks.
  grounding_supports: Information about grounding supports.
  retrieval_metadata: Metadata related to retrieval operations.
  retrieval_queries: Queries used for retrieval.
  search_entry_point: The entry point for search operations.
  web_search_queries: Queries used for web searches.
```

----------------------------------------

TITLE: Error Handling
DESCRIPTION: Details on how to handle errors that may occur during the use of the Generative AI Python SDK. This includes common error types and strategies for recovery.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.api_core.exceptions import GoogleAPIError

try:
    # Code that might raise an error
    pass
except GoogleAPIError as e:
    print(f"An API error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

----------------------------------------

TITLE: UrlContextMetadataDict
DESCRIPTION: Dictionary representation of UrlContextMetadata, including URL metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlContextMetadataDict:
  url_metadata: dict
    Metadata associated with the URL.
```

----------------------------------------

TITLE: List Batch Prediction Jobs with Pager (Asynchronous)
DESCRIPTION: Asynchronously lists batch prediction jobs using a pager, facilitating non-blocking, page-by-page retrieval. It showcases how to manage asynchronous pagination and access job details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: PreferenceOptimizationDataStats Attributes
DESCRIPTION: Provides statistics related to preference optimization data, including score distributions and token counts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationDataStats:
  score_variance_per_example_distribution: Distribution of score variance per example.
  scores_distribution: Distribution of scores.
  total_billable_token_count: Total count of billable tokens.
```

----------------------------------------

TITLE: ListCachedContentsConfigDict Fields
DESCRIPTION: Specifies the dictionary representation of configuration for listing cached content, mirroring ListCachedContentsConfig with http_options, page_size, and page_token.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListCachedContentsConfigDict:
    http_options: dict
    page_size: int
    page_token: str
```

----------------------------------------

TITLE: LiveMusicFilteredPromptDict Attributes
DESCRIPTION: Details the attributes 'filtered_reason' and 'text' for LiveMusicFilteredPromptDict, used to describe filtered prompts in live music generation, similar to LiveMusicFilteredPrompt.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicFilteredPromptDict:
  filtered_reason: The reason why the prompt was filtered.
  text: The text of the filtered prompt.
```

----------------------------------------

TITLE: UpscaleImageResponse
DESCRIPTION: Represents the response from an image upscaling operation, including generated images and SDK HTTP response details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpscaleImageResponse:
  generated_images: list[bytes]
    A list of generated image data.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response object from the SDK.
```

----------------------------------------

TITLE: Batch Job Configuration
DESCRIPTION: Configuration options for deleting batch jobs, including HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DeleteBatchJobConfig:
  http_options: dict

DeleteBatchJobConfigDict:
  http_options: dict
```

----------------------------------------

TITLE: Musical Scales
DESCRIPTION: Enumerates various musical scales, including major and minor keys, and an unspecified default.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
Scale.A_FLAT_MAJOR_F_MINOR
Scale.A_MAJOR_G_FLAT_MINOR
Scale.B_FLAT_MAJOR_G_MINOR
Scale.B_MAJOR_A_FLAT_MINOR
Scale.C_MAJOR_A_MINOR
Scale.D_FLAT_MAJOR_B_FLAT_MINOR
Scale.D_MAJOR_B_MINOR
Scale.E_FLAT_MAJOR_C_MINOR
Scale.E_MAJOR_D_FLAT_MINOR
Scale.F_MAJOR_D_MINOR
Scale.G_FLAT_MAJOR_E_FLAT_MINOR
Scale.G_MAJOR_E_MINOR
Scale.SCALE_UNSPECIFIED
```

----------------------------------------

TITLE: Generate Content with Caches
DESCRIPTION: Generates content using a specified model and the previously created cached content. This allows for more efficient and context-aware content generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Provide a non-function call part (URI)
DESCRIPTION: Illustrates how a part that is not a function call, such as a file URI, is converted by the SDK. It's placed into a `types.UserContent` object.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

contents = types.Part.from_uri(
    file_uri='gs://generativeai-downloads/images/scones.jpg',
    mime_type='image/jpeg',
)

# The SDK converts all non function call parts into a content with a `user` role.
# 
# [
# types.UserContent(parts=[
#     types.Part.from_uri(
#     file_uri: 'gs://generativeai-downloads/images/scones.jpg',
#     mime_type: 'image/jpeg',
#     )
# ])
# ]
```

----------------------------------------

TITLE: GoogleSearchRetrievalDict Attributes
DESCRIPTION: Describes the attributes of GoogleSearchRetrievalDict, highlighting dynamic_retrieval_config for retrieval settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GoogleSearchRetrievalDict:
  dynamic_retrieval_config: Configuration for dynamic retrieval.
```

----------------------------------------

TITLE: API Specification Types
DESCRIPTION: Defines the different API specification types available for use with the GenAI library. Includes unspecified, elastic search, and simple search options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ApiSpec:
  API_SPEC_UNSPECIFIED: Represents an unspecified API specification.
  ELASTIC_SEARCH: Specifies the use of Elasticsearch for API operations.
  SIMPLE_SEARCH: Specifies the use of a simple search mechanism for API operations.
```

----------------------------------------

TITLE: LiveMusicFilteredPrompt Attributes
DESCRIPTION: Details the attributes 'filtered_reason' and 'text' for LiveMusicFilteredPrompt, used to describe filtered prompts in live music generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicFilteredPrompt:
  filtered_reason: The reason why the prompt was filtered.
  text: The text of the filtered prompt.
```

----------------------------------------

TITLE: Content Attributes
DESCRIPTION: Describes the Content type, including its parts and role.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Content:
  parts: list[str | dict | object]
    The parts of the content.
  role: str
    The role associated with the content (e.g., 'user', 'model').
```

----------------------------------------

TITLE: PreferenceOptimizationDataStatsDict Fields
DESCRIPTION: Details the fields available within the PreferenceOptimizationDataStatsDict, mirroring PreferenceOptimizationDataStats with additional score-related statistics.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreferenceOptimizationDataStatsDict:
  score_variance_per_example_distribution: dict
    Distribution of score variance per example.
  scores_distribution: dict
    Distribution of scores.
  total_billable_token_count: int
    The total count of billable tokens.
  tuning_dataset_example_count: int
    The number of examples in the tuning dataset.
  tuning_step_count: int
    The number of tuning steps performed.
  user_dataset_examples: int
    The number of examples provided by the user.
  user_input_token_distribution: dict
    A dictionary representing the distribution of input tokens.
  user_output_token_distribution: dict
    A dictionary representing the distribution of output tokens.
```

----------------------------------------

TITLE: Generate Content with Safety Settings
DESCRIPTION: Configures content generation to include safety settings, specifically blocking high-severity hate speech. This allows control over the types of content the model will generate.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_ONLY_HIGH',
            )
        ]
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Create Batch Prediction Job (BigQuery/GCS Source)
DESCRIPTION: Creates a batch prediction job using either a BigQuery table or a Google Cloud Storage (GCS) file as the data source. The model and source are specified, while destination and job display name are auto-populated.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
# Specify model and source file only, destination and job display name will be auto-populated
job = client.batches.create(
    model='gemini-2.0-flash-001',
    src='bq://my-project.my-dataset.my-table',  # or gcs://my-bucket/my-file.jsonl
)
```

----------------------------------------

TITLE: Safety Setting Dictionary Properties
DESCRIPTION: Details the properties of a SafetySettingDict object, allowing dictionary-based access to safety settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
SafetySettingDict.category
SafetySettingDict.method
SafetySettingDict.threshold
```

----------------------------------------

TITLE: LiveClientMessageDict Attributes
DESCRIPTION: Specifies the attributes for LiveClientMessageDict, corresponding to LiveClientMessage attributes.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveClientMessageDict:
  client_content: The content provided by the client.
  realtime_input: Real-time input data from the client.
  setup: Setup information for the client message.
  tool_response: The response from a tool used by the client.
```

----------------------------------------

TITLE: TuningTask Enum
DESCRIPTION: Specifies the types of tuning tasks supported, such as image-to-video (I2V) or text-to-video (T2V) tuning.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningTask:
  TUNING_TASK_I2V: Image-to-video tuning task.
  TUNING_TASK_T2V: Text-to-video tuning task.
  TUNING_TASK_UNSPECIFIED: Unspecified tuning task.
```

----------------------------------------

TITLE: PersonGeneration Constants
DESCRIPTION: Constants for controlling person generation behavior, specifying different levels of content allowance.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PersonGeneration:
  ALLOW_ADULT: Allows adult content.
  ALLOW_ALL: Allows all content.
  DONT_ALLOW: Does not allow content.
```

----------------------------------------

TITLE: ReplayRequest Types
DESCRIPTION: Details the structure for ReplayRequest and its dictionary representation, covering body segments, headers, method, and URL.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ReplayRequest:
  body_segments: Segments of the request body.
  headers: Headers of the request.
  method: The HTTP method used.
  url: The URL of the request.

ReplayRequestDict:
  body_segments: Segments of the request body (dictionary format).
  headers: Headers of the request (dictionary format).
  method: The HTTP method used (dictionary format).
  url: The URL of the request (dictionary format).
```

----------------------------------------

TITLE: LiveMusicConnectParametersDict Attributes
DESCRIPTION: Details the 'model' attribute for LiveMusicConnectParametersDict, used for connection parameters in live music scenarios, similar to LiveMusicConnectParameters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicConnectParametersDict:
  model: Specifies the model to be used for live music connection.
```

----------------------------------------

TITLE: Chat: Send Message (Synchronous Streaming)
DESCRIPTION: Initiates a chat session and sends a message to the model, receiving the response in streaming chunks. Each chunk's text is printed as it arrives.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: Live Music Playback Control Types
DESCRIPTION: Defines control types for live music playback, including a STOP constant.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import LiveMusicPlaybackControl

# Example usage:
playback_control = LiveMusicPlaybackControl.STOP
```

----------------------------------------

TITLE: TuningJob Fields
DESCRIPTION: Details the fields within the TuningJob, representing a model tuning job configuration and status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningJob:
  base_model: The base model used for tuning.
  create_time: The timestamp when the tuning job was created.
  custom_base_model: The custom base model used for tuning.
  description: A description of the tuning job.
  distillation_spec: Specification for distillation tuning.
  encryption_spec: Specification for data encryption.
  end_time: The timestamp when the tuning job finished.
  error: Information about any errors encountered during the job.
  experiment: The experiment associated with the tuning job.
  labels: Labels associated with the tuning job.
  name: The name of the tuning job.
  output_uri: The URI where the tuned model output is stored.
  partner_model_tuning_spec: Specification for partner model tuning.
  pipeline_job: The pipeline job associated with the tuning.
  pre_tuned_model: The pre-tuned model used as a starting point.
  preference_optimization_spec: Specification for preference optimization tuning.
  satisfies_pzi: Indicates if the job satisfies PZI requirements.
  satisfies_pzs: Indicates if the job satisfies PZS requirements.
  sdk_http_response: The raw HTTP response from the SDK.
```

----------------------------------------

TITLE: GroundingSupport API
DESCRIPTION: Provides access to confidence scores, grounding chunk indices, and segment information for grounding support.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingSupport:
  confidence_scores: List of confidence scores.
  grounding_chunk_indices: Indices of grounding chunks.
  segment: The segment of text.

GroundingSupportDict:
  confidence_scores: List of confidence scores.
  grounding_chunk_indices: Indices of grounding chunks.
  segment: The segment of text.
```

----------------------------------------

TITLE: Generate Content with JSON Schema
DESCRIPTION: Shows how to generate content with a JSON schema for the response. The model's output will be a JSON object adhering to the provided schema.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            'required': [
                'name',
                'population',
                'capital',
                'continent',
                'gdp',
                'official_language',
                'total_area_sq_mi',
            ],
            'properties': {
                'name': {'type': 'STRING'},
                'population': {'type': 'INTEGER'},
                'capital': {'type': 'STRING'},
                'continent': {'type': 'STRING'},
                'gdp': {'type': 'INTEGER'},
                'official_language': {'type': 'STRING'},
                'total_area_sq_mi': {'type': 'INTEGER'},
            },
            'type': 'OBJECT',
        },
    ),
)
print(response.text)
```

----------------------------------------

TITLE: Function Response Types
DESCRIPTION: Details the structure for function responses, including ID, name, response content, scheduling, and continuation status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FunctionResponse:
  id: str
    Unique identifier for the function response.
  name: str
    The name of the function that was called.
  response: Union[str, Dict[str, Any]]
    The actual response content from the function.
  scheduling: FunctionResponseScheduling
    Specifies how the response should be scheduled.
  will_continue: bool
    Indicates if the function call is part of a multi-turn interaction.
  from_mcp_response(response: Dict[str, Any]) -> 'FunctionResponse'
    Class method to create a FunctionResponse from a dictionary.

FunctionResponseDict:
  id: str
    Unique identifier for the function response.
  name: str
    The name of the function that was called.
  response: Union[str, Dict[str, Any]]
    The actual response content from the function.
  scheduling: FunctionResponseScheduling
    Specifies how the response should be scheduled.
  will_continue: bool
    Indicates if the function call is part of a multi-turn interaction.
```

----------------------------------------

TITLE: Schema Methods
DESCRIPTION: Details the methods available for the Schema type in the Python GenAI library, used for schema manipulation and conversion.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Schema:
  from_json_schema(schema: dict) -> Schema: Creates a Schema object from a JSON schema dictionary.
  json_schema() -> dict: Returns the JSON schema representation of the Schema object.
```

----------------------------------------

TITLE: GeneratedImageDict Fields
DESCRIPTION: A dictionary representation of the GeneratedImage.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GeneratedImageDict:
  enhanced_prompt: str | None
    The prompt after potential enhancements by the model.
  image: ImageDict
    A dictionary representation of the generated image.
  rai_filtered_reason: str | None
    The reason if the image was filtered by RAI.
  safety_attributes: SafetyAttributesDict
    A dictionary representation of safety attributes.
```

----------------------------------------

TITLE: HarmCategory API
DESCRIPTION: Enumerates categories of harmful content that can be detected and blocked.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HarmCategory:
  HARM_CATEGORY_CIVIC_INTEGRITY: Civic integrity harm.
  HARM_CATEGORY_DANGEROUS_CONTENT: Dangerous content harm.
  HARM_CATEGORY_HARASSMENT: Harassment harm.
  HARM_CATEGORY_HATE_SPEECH: Hate speech harm.
  HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT: Image dangerous content harm.
  HARM_CATEGORY_IMAGE_HARASSMENT: Image harassment harm.
  HARM_CATEGORY_IMAGE_HATE: Image hate harm.
  HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT: Image sexually explicit harm.
  HARM_CATEGORY_SEXUALLY_EXPLICIT: Sexually explicit harm.
  HARM_CATEGORY_UNSPECIFIED: Unspecified harm category.
```

----------------------------------------

TITLE: VertexRagStoreDict Attributes
DESCRIPTION: Documentation for the VertexRagStoreDict type, detailing its attributes for RAG configuration.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
VertexRagStoreDict:
  rag_corpora: List of RAG corpora to use.
  rag_resources: List of RAG resources to use.
  rag_retrieval_config: Configuration for RAG retrieval.
  similarity_top_k: The number of similar documents to retrieve.
  store_context: Whether to store context.
  vector_distance_threshold: The threshold for vector distance.
```

----------------------------------------

TITLE: SubjectReferenceType API Documentation
DESCRIPTION: Defines the types of subjects that can be referenced, including animal, default, person, and product.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SubjectReferenceType:
  SUBJECT_TYPE_ANIMAL: Represents an animal subject.
  SUBJECT_TYPE_DEFAULT: Represents a default subject type.
  SUBJECT_TYPE_PERSON: Represents a person subject.
  SUBJECT_TYPE_PRODUCT: Represents a product subject.
```

----------------------------------------

TITLE: CountTokensConfig and CountTokensResponse Types
DESCRIPTION: Defines the configuration and response structures for the CountTokens API. Includes settings for generation, HTTP options, system instructions, and tools. The response provides token counts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CountTokensConfig:
  description: Configuration for counting tokens.
  fields:
    generation_config: Configuration for generation.
    http_options: HTTP options for the request.
    system_instruction: System instruction for the model.
    tools: Tools to be used by the model.

CountTokensConfigDict:
  description: Dictionary representation of CountTokensConfig.
  fields:
    generation_config: Configuration for generation.
    http_options: HTTP options for the request.
    system_instruction: System instruction for the model.
    tools: Tools to be used by the model.

CountTokensResponse:
  description: Response from the CountTokens API.
  fields:
    cached_content_token_count: Token count for cached content.
    sdk_http_response: The HTTP response from the SDK.
    total_tokens: The total number of tokens.

CountTokensResponseDict:
  description: Dictionary representation of CountTokensResponse.
  fields:
    cached_content_token_count: Token count for cached content.
    sdk_http_response: The HTTP response from the SDK.
    total_tokens: The total number of tokens.
```

----------------------------------------

TITLE: UrlContext
DESCRIPTION: Represents context information related to a URL.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlContext:
  # No specific fields documented in the provided text.
```

----------------------------------------

TITLE: Streaming Image Content Generation from Local File
DESCRIPTION: Generates content based on an image from the local file system. The image is read as bytes and used to stream the response.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

YOUR_IMAGE_PATH = 'your_image_path'

# Assuming YOUR_IMAGE_PATH is correctly set and the file exists
# with open(YOUR_IMAGE_PATH, 'rb') as f:
#     image_bytes = f.read()
# 
# for chunk in client.models.generate_content_stream(
#     model='gemini-2.0-flash-001',
#     contents=[
#         'What is this image about?',
#         types.Part.from_bytes(image_bytes, mime_type='image/jpeg'), # Adjust mime_type as needed
#     ],
# ):
#     print(chunk.text, end='')
```

----------------------------------------

TITLE: RetrievalMetadata Attributes
DESCRIPTION: Documentation for RetrievalMetadata and RetrievalMetadataDict types, which contain metadata related to retrieval operations. Includes the Google Search dynamic retrieval score.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RetrievalMetadata:
  google_search_dynamic_retrieval_score: The dynamic retrieval score from Google Search.

RetrievalMetadataDict:
  google_search_dynamic_retrieval_score: The dynamic retrieval score from Google Search.
```

----------------------------------------

TITLE: Enum Response Schema Support
DESCRIPTION: Illustrates how to use Python Enums to define expected response values, ensuring that the model returns one of the predefined enum members. Supports both text and JSON formats.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from enum import Enum

class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

LANGUAGE: python
CODE:
```
class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'application/json',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

----------------------------------------

TITLE: Create Batch Prediction Job (BigQuery/GCS Source)
DESCRIPTION: Creates a batch prediction job using either a BigQuery table or a Google Cloud Storage (GCS) file as the data source. The model and source are specified, while destination and job display name can be auto-populated.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
# Specify model and source file only, destination and job display name will be auto-populated
job = client.batches.create(
    model='gemini-2.0-flash-001',
    src='bq://my-project.my-dataset.my-table',  # or gcs://my-bucket/my-file.jsonl
)
```

----------------------------------------

TITLE: SearchEntryPoint Properties
DESCRIPTION: Details the properties available for the SearchEntryPoint type in the Python GenAI library. These properties relate to search results and their content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SearchEntryPoint:
  rendered_content: The rendered content of the search result.
  sdk_blob: Blob data associated with the SDK.
```

----------------------------------------

TITLE: Chat: Send Message (Asynchronous Streaming)
DESCRIPTION: Demonstrates sending a message to a chat model asynchronously and receiving a streamed response. This allows for non-blocking, real-time updates in asynchronous applications.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text, end='') # end='' is optional, for demo purposes.
```

----------------------------------------

TITLE: File Management
DESCRIPTION: Provides functionalities to upload, retrieve, and delete files that can be used as input for generative models. This includes images, documents, and other data types.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

# Upload a file
# with open('my_document.pdf', 'rb') as f:
#     uploaded_file = genai.upload_file(f)
# print(f'Uploaded file: {uploaded_file.uri}')

# Get file information
# file_uri = 'gs://your-bucket/my_document.pdf'
# file_info = genai.get_file(file_uri)
# print(f'File name: {file_info.display_name}')

# Delete a file
# genai.delete_file(file_uri)
# print('File deleted.')
```

----------------------------------------

TITLE: Python GenAI: Function Calling with ANY Tool Config Mode (Set Max Turns)
DESCRIPTION: Explains how to set a specific number of automatic function call turns when using the 'ANY' function calling mode. This is achieved by configuring the maximum remote calls.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: GenerateContentResponse Structure
DESCRIPTION: Defines the structure of the GenerateContentResponse object returned by the Google Generative AI Python SDK. It includes information about generated candidates, creation time, and model version.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponse:
  automatic_function_calling_history: History of automatic function calls.
  candidates: List of generated candidates.
  create_time: Timestamp when the content was created.
  model_version: Version of the model used for generation.
```

----------------------------------------

TITLE: DynamicRetrievalConfigMode Enum
DESCRIPTION: Defines the possible modes for dynamic retrieval.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DynamicRetrievalConfigMode:
  MODE_DYNAMIC: Enables dynamic retrieval.
  MODE_UNSPECIFIED: Unspecified mode for dynamic retrieval.
```

----------------------------------------

TITLE: Chat: Send Message (Synchronous Non-Streaming)
DESCRIPTION: Initiates a chat session and sends a message to the model. It then sends a follow-up message to continue the conversation and prints the text response.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
response = chat.send_message('tell me a story')
print(response.text)
response = chat.send_message('summarize the story you told me in 1 sentence')
print(response.text)
```

----------------------------------------

TITLE: List Batch Jobs (Asynchronous)
DESCRIPTION: Provides an asynchronous way to list batch jobs, suitable for non-blocking operations in async applications. It uses an async iterator to fetch jobs.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)
```

----------------------------------------

TITLE: TuningValidationDataset and TuningValidationDatasetDict Attributes
DESCRIPTION: Defines attributes for specifying validation datasets, including Google Cloud Storage URIs and Vertex AI dataset resource references.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningValidationDataset:
  gcs_uri: The Google Cloud Storage URI of the validation dataset.
  vertex_dataset_resource: The Vertex AI dataset resource.

TuningValidationDatasetDict:
  gcs_uri: The Google Cloud Storage URI of the validation dataset.
  vertex_dataset_resource: The Vertex AI dataset resource.
```

----------------------------------------

TITLE: TuningDataStatsDict Fields
DESCRIPTION: Details the fields within the TuningDataStatsDict, which provides statistics for different tuning data types.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningDataStatsDict:
  distillation_data_stats: Statistics for distillation data.
  preference_optimization_data_stats: Statistics for preference optimization data.
  supervised_tuning_data_stats: Statistics for supervised tuning data.
```

----------------------------------------

TITLE: WeightedPrompt Types
DESCRIPTION: Defines types for weighted prompts, allowing users to assign different weights to text inputs for influencing model generation. Includes text content and its associated weight.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
WeightedPrompt:
  text: The text content of the prompt.
  weight: The weight assigned to this prompt (higher weight means more influence).

WeightedPromptDict:
  text: The text content of the prompt.
  weight: The weight assigned to this prompt.
```

----------------------------------------

TITLE: GenerateContentResponseUsageMetadata Attributes
DESCRIPTION: Explains the attributes of GenerateContentResponseUsageMetadata, covering cached content token count, candidate token count, and details about cached tokens.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponseUsageMetadata:
  cache_tokens_details: Details about cached tokens.
  cached_content_token_count: The token count for cached content.
  candidates_token_count: The token count for the candidates.
```

----------------------------------------

TITLE: Function Declaration Types
DESCRIPTION: Defines the structure for declaring functions, including response schema and content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FunctionDeclarationDict:
  response: Union[str, Dict[str, Any]]
    The response from the function call.
  response_json_schema: Dict[str, Any]
    A JSON schema defining the structure of the response.
```

----------------------------------------

TITLE: Dataset Distribution Statistics
DESCRIPTION: Provides statistical information about a dataset's distribution, including buckets, min, max, mean, median, and percentiles. Supports both object and dictionary representations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DatasetDistribution:
  buckets: List of distribution buckets.
  max: Maximum value in the dataset.
  mean: Mean value of the dataset.
  median: Median value of the dataset.
  min: Minimum value in the dataset.
  p5: 5th percentile of the dataset.
  p95: 95th percentile of the dataset.
  sum: Sum of all values in the dataset.

DatasetDistributionDict:
  buckets: List of distribution buckets.
  max: Maximum value in the dataset.
  mean: Mean value of the dataset.
  median: Median value of the dataset.
  min: Minimum value in the dataset.
  p5: 5th percentile of the dataset.
  p95: 95th percentile of the dataset.
  sum: Sum of all values in the dataset.
```

----------------------------------------

TITLE: GenerateContentResponseUsageMetadata Fields
DESCRIPTION: Details the fields available within the GenerateContentResponseUsageMetadata object, which provides information about token usage for generated content. This includes counts and detailed breakdowns for prompts, candidates, and tool usage.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponseUsageMetadata:
  candidates_tokens_details: List of token details for candidates.
  prompt_token_count: Total prompt token count.
  prompt_tokens_details: List of token details for the prompt.
  thoughts_token_count: Token count for thoughts.
  tool_use_prompt_token_count: Token count for tool use in the prompt.
  tool_use_prompt_tokens_details: List of token details for tool use in the prompt.
  total_token_count: Total token count for the response.
  traffic_type: Type of traffic for the response.
```

----------------------------------------

TITLE: GeneratedVideo Fields
DESCRIPTION: Represents a generated video.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GeneratedVideo:
  video: Video
    The generated video object.
```

----------------------------------------

TITLE: FileSource Enumerations
DESCRIPTION: Defines the possible sources for a file. These indicate whether the file was uploaded by a user or generated by the system.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileSource:
  GENERATED: The file was generated by the system.
  SOURCE_UNSPECIFIED: The source of the file is unspecified.
  UPLOADED: The file was uploaded by a user.
```

----------------------------------------

TITLE: Thinking Configuration Types
DESCRIPTION: Defines types for configuring thinking parameters, such as thinking budget and inclusion of thoughts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ThinkingConfig:
  thinking_budget: int | None
    The maximum number of tokens to use for thinking.

ThinkingConfigDict:
  include_thoughts: bool
    Whether to include thoughts in the output.
  thinking_budget: int | None
    The maximum number of tokens to use for thinking.
```

----------------------------------------

TITLE: CitationDict Attributes
DESCRIPTION: Details the attributes available within the CitationDict type, including license, publication_date, start_index, title, and uri.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CitationDict:
  license: str
    The license associated with the citation.
  publication_date: str
    The publication date of the cited work.
  start_index: int
    The starting index of the citation within the content.
  title: str
    The title of the cited work.
  uri: str
    The Uniform Resource Identifier (URI) of the cited work.
```

----------------------------------------

TITLE: GenAI Types and Attributes
DESCRIPTION: This section details various types available in the Python GenAI library, including BlockedReason, CachedContent, CancelBatchJobConfig, and Candidate. It outlines their attributes and provides links to specific documentation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BlockedReason:
  PROHIBITED_CONTENT: Represents prohibited content.
  SAFETY: Represents safety-related blocking.

CachedContent:
  create_time: Timestamp of content creation.
  display_name: Display name for the cached content.
  expire_time: Timestamp when the content expires.
  model: The model used for caching.
  name: The name of the cached content.
  update_time: Timestamp of the last update.
  usage_metadata: Metadata about the usage of the cached content.

CachedContentDict:
  create_time: Timestamp of content creation.
  display_name: Display name for the cached content.
  expire_time: Timestamp when the content expires.
  model: The model used for caching.
  name: The name of the cached content.
  update_time: Timestamp of the last update.
  usage_metadata: Metadata about the usage of the cached content.

CachedContentUsageMetadata:
  audio_duration_seconds: Duration of audio content in seconds.
  image_count: Number of images.
  text_count: Number of text elements.
  total_token_count: Total token count.
  video_duration_seconds: Duration of video content in seconds.

CachedContentUsageMetadataDict:
  audio_duration_seconds: Duration of audio content in seconds.
  image_count: Number of images.
  text_count: Number of text elements.
  total_token_count: Total token count.
  video_duration_seconds: Duration of video content in seconds.

CancelBatchJobConfig:
  http_options: HTTP options for cancelling a batch job.

CancelBatchJobConfigDict:
  http_options: HTTP options for cancelling a batch job.

Candidate: Represents a candidate response from the model.
```

----------------------------------------

TITLE: Embed Content
DESCRIPTION: Illustrates how to embed content into vector representations using the `client.models.embed_content` method. This is commonly used for semantic search and similarity tasks.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)

```

----------------------------------------

TITLE: Chat: Send Message (Asynchronous Non-Streaming)
DESCRIPTION: Initiates an asynchronous chat session and sends a message to the model. It awaits the response and prints the text.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
response = await chat.send_message('tell me a story')
print(response.text)
```

----------------------------------------

TITLE: RagRetrievalConfigFilter Fields
DESCRIPTION: Details the fields available for RagRetrievalConfigFilter, used for filtering retrieval results.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigFilter:
  (No specific fields detailed in the provided text, but it's a filter configuration object.)
```

----------------------------------------

TITLE: LiveServerGoAway Types
DESCRIPTION: Details for the LiveServerGoAway type and its 'time_left' attribute, used for signaling server-side go-away events.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerGoAway:
  time_left: int
    The time in seconds until the server will go away.

LiveServerGoAwayDict:
  time_left: int
    The time in seconds until the server will go away.
```

----------------------------------------

TITLE: JobState Type Documentation
DESCRIPTION: Documentation for the JobState type, which indicates the current state of a job.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
JobState:
  Indicates the state of a job.
```

----------------------------------------

TITLE: Model Deletion Configuration
DESCRIPTION: Configuration options for deleting models, including HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DeleteModelConfig:
  http_options: dict
```

----------------------------------------

TITLE: ExternalApiSimpleSearchParamsDict
DESCRIPTION: A dictionary representation for simple search parameters in the External API.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ExternalApiSimpleSearchParamsDict:
  (No specific parameters documented in this snippet)
```

----------------------------------------

TITLE: Create Batch Prediction Job (Inlined Requests)
DESCRIPTION: Creates a batch prediction job using inlined request data. Each request includes the content to be processed and optional generation configuration, such as response modalities.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
# Create a batch job with inlined requests
batch_job = client.batches.create(
    model="gemini-2.0-flash",
    src=[{
      "contents": [{
        "parts": [{
          "text": "Hello!",
        }],
       "role": "user",
     }],
     "config:": {"response_modalities": ["text"]},
    }],
)
```

----------------------------------------

TITLE: LiveServerMessage Types
DESCRIPTION: Defines the various message types within the LiveServerMessage structure, including server content, tool calls, and usage metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerMessage:
  go_away: LiveServerGoAway | None
    If set, the server is going away.
  server_content: str | None
    Server-generated content.
  session_resumption_update: LiveServerSessionResumptionUpdate | None
    Information about session resumption.
  setup_complete: LiveServerSetupComplete | None
    Indicates that the setup is complete.
  tool_call: ToolCall | None
    A tool call from the server.
  tool_call_cancellation: ToolCallCancellation | None
    A cancellation for a tool call.
  usage_metadata: UsageMetadata | None
    Metadata about the usage of the model.
  data: str | bytes | None
    Raw data payload.
  text: str | None
    Text content of the message.

LiveServerMessageDict:
  go_away: LiveServerGoAwayDict | None
    If set, the server is going away.
  server_content: str | None
    Server-generated content.
  session_resumption_update: LiveServerSessionResumptionUpdateDict | None
    Information about session resumption.
  setup_complete: LiveServerSetupCompleteDict | None
    Indicates that the setup is complete.
  tool_call: ToolCallDict | None
    A tool call from the server.
  tool_call_cancellation: ToolCallCancellationDict | None
    A cancellation for a tool call.
  usage_metadata: UsageMetadataDict | None
    Metadata about the usage of the model.
```

----------------------------------------

TITLE: UrlContextMetadata
DESCRIPTION: Contains metadata related to a URL context, specifically the URL metadata itself.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlContextMetadata:
  url_metadata: UrlMetadata
    Metadata associated with the URL.
```

----------------------------------------

TITLE: GenerateImagesResponse Structure
DESCRIPTION: Represents the response from an image generation request, containing the generated images.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateImagesResponse:
  generated_images: list
    A list of generated images. Each item in the list may contain image data or a URI.
```

----------------------------------------

TITLE: UrlMetadata
DESCRIPTION: Provides metadata about a URL, including the retrieved URL and its retrieval status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlMetadata:
  retrieved_url: str
    The actual URL that was retrieved.
  url_retrieval_status: str
    The status of the URL retrieval.
```

----------------------------------------

TITLE: List Tuned Models (Asynchronous)
DESCRIPTION: Provides an asynchronous way to list tuned models, allowing for non-blocking iteration over the model list. It mirrors the synchronous `list` functionality but uses async iterators.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}}):
    print(job)
```

LANGUAGE: python
CODE:
```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False}})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])
```

----------------------------------------

TITLE: Cache Management
DESCRIPTION: Allows for creating, retrieving, and managing caches for model responses. Caching can improve performance and reduce costs by reusing previous results.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

# Create a cache
# cache_id = genai.create_cache(display_name='my-response-cache')
# print(f'Created cache: {cache_id}')

# Get cache information
# cache_info = genai.get_cache(cache_id)
# print(f'Cache display name: {cache_info.display_name}')

# Generate content using a cache
# model = genai.GenerativeModel('gemini-pro', cache_id=cache_id)
# response = model.generate_content('What is the capital of France?')
# print(response.text)
```

----------------------------------------

TITLE: ContentEmbeddingStatistics API
DESCRIPTION: API documentation for ContentEmbeddingStatistics, including its truncated attribute and related dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ContentEmbeddingStatistics:
  truncated: bool
    Indicates if the content was truncated during embedding.

ContentEmbeddingStatisticsDict:
  token_count: int
    The number of tokens in the content.
  truncated: bool
    Indicates if the content was truncated during embedding.
```

----------------------------------------

TITLE: GroundingChunkRetrievedContext
DESCRIPTION: Represents retrieved context information for grounding chunks.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkRetrievedContext:
  # Further details about the structure of GroundingChunkRetrievedContext would go here.
```

----------------------------------------

TITLE: Batch Prediction API
DESCRIPTION: Provides functionalities for creating, listing, and deleting batch prediction jobs. This includes methods for managing prediction tasks in batches.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchPrediction:
  Create(model_name: str, instances: list, parameters: dict = None) -> BatchPredictionJob
    model_name: The name of the model to use for prediction.
    instances: A list of input instances for the model.
    parameters: Optional parameters for the prediction.
    Returns: The created BatchPredictionJob object.

  List(filter: str = None, page_size: int = None, page_token: str = None) -> list[BatchPredictionJob]
    filter: Optional filter for listing jobs.
    page_size: Number of results to return per page.
    page_token: Token to retrieve the next page of results.
    Returns: A list of BatchPredictionJob objects.

  ListBatchJobsWithPager(filter: str = None, page_size: int = None) -> Pager[BatchPredictionJob]
    filter: Optional filter for listing jobs.
    page_size: Number of results to return per page.
    Returns: A Pager object for iterating through BatchPredictionJob objects.

  ListBatchJobsAsynchronous(filter: str = None, page_size: int = None, page_token: str = None) -> list[BatchPredictionJob]
    filter: Optional filter for listing jobs.
    page_size: Number of results to return per page.
    page_token: Token to retrieve the next page of results.
    Returns: A list of BatchPredictionJob objects (asynchronous).

  ListBatchJobsWithPagerAsynchronous(filter: str = None, page_size: int = None) -> Pager[BatchPredictionJob]
    filter: Optional filter for listing jobs.
    page_size: Number of results to return per page.
    Returns: A Pager object for iterating through BatchPredictionJob objects (asynchronous).

  Delete(job_id: str) -> None
    job_id: The ID of the batch prediction job to delete.
    Returns: None on successful deletion.
```

----------------------------------------

TITLE: GoogleSearchRetrieval Attributes
DESCRIPTION: Outlines the attributes for GoogleSearchRetrieval, focusing on dynamic_retrieval_config for configuring retrieval behavior.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GoogleSearchRetrieval:
  dynamic_retrieval_config: Configuration for dynamic retrieval.
```

----------------------------------------

TITLE: SessionResumptionConfigDict Properties
DESCRIPTION: Details the properties available for the SessionResumptionConfigDict type in the Python GenAI library. These properties are dictionary representations of session resumption configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SessionResumptionConfigDict:
  handle: The handle for session resumption.
  transparent: Whether session resumption is transparent.
```

----------------------------------------

TITLE: ContentEmbedding Attributes
DESCRIPTION: Outlines the attributes of ContentEmbedding, including statistics and values.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ContentEmbedding:
  statistics: ContentEmbeddingStatistics
    Statistics related to the content embedding.
  values: list[float]
    The embedding values for the content.
```

----------------------------------------

TITLE: TurnCoverage Enum
DESCRIPTION: Enumerates coverage options for turns in a conversation or dataset, indicating whether all input is included or only activity.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TurnCoverage:
  TURN_COVERAGE_UNSPECIFIED: Unspecified turn coverage.
  TURN_INCLUDES_ALL_INPUT: The turn includes all input.
  TURN_INCLUDES_ONLY_ACTIVITY: The turn includes only activity.
```

----------------------------------------

TITLE: ModelContent Fields
DESCRIPTION: Defines the structure for model content, including parts and the associated role.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ModelContent:
  parts: The content parts of the model's response.
  role: The role associated with the content (e.g., 'user', 'model').
```

----------------------------------------

TITLE: GenerateContentResponse Attributes
DESCRIPTION: Details the attributes available on the GenerateContentResponse object, such as parsed content, prompt feedback, response ID, HTTP response details, usage metadata, code execution results, executable code, function calls, and text content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponse:
  parsed: The parsed content of the response.
  prompt_feedback: Feedback related to the prompt.
  response_id: The unique identifier for the response.
  sdk_http_response: The HTTP response object from the SDK.
  usage_metadata: Metadata about token usage.
  code_execution_result: The result of code execution.
  executable_code: The executable code generated.
  function_calls: Any function calls made by the model.
  text: The text content of the response.
```

----------------------------------------

TITLE: CodeExecutionResult Attributes
DESCRIPTION: Outlines the attributes of the CodeExecutionResult type, including outcome and output.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CodeExecutionResult:
  outcome: str
    The outcome of the code execution.
  output: str
    The output generated by the code execution.
```

----------------------------------------

TITLE: SpeakerVoiceConfigDict Properties
DESCRIPTION: Details the properties available for the SpeakerVoiceConfigDict type in the Python GenAI library. These properties are dictionary representations of speaker voice configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SpeakerVoiceConfigDict:
  speaker: The speaker identifier.
  voice_config: Configuration for the voice.
```

----------------------------------------

TITLE: ComputeTokensConfig Attributes
DESCRIPTION: Details the ComputeTokensConfig type and its http_options attribute.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ComputeTokensConfig:
  http_options: dict
    HTTP options for the compute tokens request.
```

----------------------------------------

TITLE: Distillation Data Statistics
DESCRIPTION: Defines data structures for capturing statistics related to distillation datasets, specifically focusing on training dataset statistics.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DistillationDataStats:
  training_dataset_stats: Statistics pertaining to the training dataset.

DistillationDataStatsDict:
  A dictionary representation of DistillationDataStats.
  training_dataset_stats: Statistics pertaining to the training dataset.
```

----------------------------------------

TITLE: GroundingChunkMaps Attributes
DESCRIPTION: Describes the attributes of GroundingChunkMaps, including place_answer_sources, place_id, text, title, and uri for map-related data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkMaps:
  place_answer_sources: Sources for place answers.
  place_id: Unique identifier for a place.
  text: Textual content associated with the map.
  title: Title of the map entry.
  uri: Uniform Resource Identifier for the map.
```

----------------------------------------

TITLE: JSON Response Schema Support
DESCRIPTION: Allows the model to return structured data in JSON format, which can be directly parsed into Python objects, including Pydantic models.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
import google.generativeai as genai

class Person(BaseModel):
    name: str
    age: int

# Configure the model to return JSON
model = genai.GenerativeModel('gemini-pro', response_mime_type='application/json')

# Request a JSON response that conforms to the Person schema
# response = model.generate_content("Create a JSON object for a person named Alice who is 30 years old.")
# print(response.text)

# To parse into Pydantic:
# person_data = json.loads(response.text)
# person_obj = Person(**person_data)
```

----------------------------------------

TITLE: ComputeTokensResponseDict Attributes
DESCRIPTION: Details the ComputeTokensResponseDict type, containing sdk_http_response and tokens_info.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ComputeTokensResponseDict:
  sdk_http_response: object
    The HTTP response from the SDK.
  tokens_info: object
    Information about the computed tokens.
```

----------------------------------------

TITLE: LiveServerGoAway
DESCRIPTION: Represents a signal from the live server indicating that the connection should be closed or terminated.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerGoAway:
  (No specific fields documented, represents a termination signal.)
```

----------------------------------------

TITLE: MaskReferenceMode API
DESCRIPTION: Enumerates the different modes for referencing masks, specifying how the mask should be applied or interpreted.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MaskReferenceMode:
  MASK_MODE_BACKGROUND: Indicates background masking.
  MASK_MODE_DEFAULT: Uses the default masking behavior.
  MASK_MODE_FOREGROUND: Indicates foreground masking.
  MASK_MODE_SEMANTIC: Uses semantic information for masking.
  MASK_MODE_USER_PROVIDED: The mask is provided by the user.
```

----------------------------------------

TITLE: TuningDataset Fields
DESCRIPTION: Details the fields within the TuningDataset, representing a dataset used for model tuning.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningDataset:
  examples: A list of tuning examples.
  gcs_uri: The Google Cloud Storage URI for the dataset.
  vertex_dataset_resource: The Vertex AI dataset resource identifier.
```

----------------------------------------

TITLE: SpeakerVoiceConfig Properties
DESCRIPTION: Details the properties available for the SpeakerVoiceConfig type in the Python GenAI library. These properties configure speaker and voice settings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SpeakerVoiceConfig:
  speaker: The speaker identifier.
  voice_config: Configuration for the voice.
```

----------------------------------------

TITLE: HttpRetryOptionsDict Attributes
DESCRIPTION: Details the attributes available within the HttpRetryOptionsDict type, which are used for configuring HTTP retry behavior.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HttpRetryOptionsDict:
  attempts: Number of HTTP retry attempts.
  exp_base: The base for exponential backoff.
  http_status_codes: List of HTTP status codes that trigger a retry.
  initial_delay: The initial delay in seconds before the first retry.
  jitter: The factor for adding randomness to the delay.
  max_delay: The maximum delay in seconds between retries.
```

----------------------------------------

TITLE: PreTunedModelDict Attributes
DESCRIPTION: A dictionary representation of PreTunedModel, used for specifying pre-tuned model details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreTunedModelDict:
  base_model: The base model used for tuning.
  checkpoint_id: The ID of the model checkpoint.
  tuned_model_name: The name of the tuned model.
```

----------------------------------------

TITLE: EncryptionSpec and EncryptionSpecDict
DESCRIPTION: Specifies encryption settings, particularly the KMS key name for encrypting data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EncryptionSpec:
  kms_key_name: The Cloud KMS key name to use for encryption.

EncryptionSpecDict:
  kms_key_name: The Cloud KMS key name to use for encryption.
```

----------------------------------------

TITLE: JSONSchema Type Documentation
DESCRIPTION: Documentation for the JSONSchema type, used to define the structure and constraints of JSON data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
JSONSchema:
  Defines the structure and constraints of JSON data.
  Attributes:
    any_of: A list of schemas, where the data must validate against at least one.
    default: The default value for the schema.
    description: A description of the schema.
    enum: A list of allowed values.
    format: The format of the data (e.g., 'date-time', 'email').
    items: Defines the schema for array items.
    max_items: The maximum number of items in an array.
    max_length: The maximum length of a string.
    max_properties: The maximum number of properties in an object.
    maximum: The maximum allowed value for a number.
    min_items: The minimum number of items in an array.
    min_length: The minimum length of a string.
    min_properties: The minimum number of properties in an object.
    minimum: The minimum allowed value for a number.
    pattern: A regular expression pattern for a string.
    properties: An object defining the schemas for object properties.
    required: A list of required property names.
    title: A title for the schema.
    type: The data type (e.g., 'string', 'integer', 'object').
```

----------------------------------------

TITLE: SchemaDict Properties
DESCRIPTION: Details the properties available for the SchemaDict type in the Python GenAI library. These properties are used for defining schema constraints and metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SchemaDict:
  min_properties: Minimum number of properties required.
  minimum: The minimum value for a number.
  nullable: Whether the value can be null.
  pattern: A regular expression pattern.
  properties: A dictionary of properties.
  property_ordering: The order of properties.
  ref: A reference to another schema.
  required: A list of required properties.
  title: The title of the schema.
  type: The data type of the schema.
```

----------------------------------------

TITLE: Token Information Types
DESCRIPTION: Provides types for retrieving information about tokens, including role, token IDs, and token counts.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TokensInfo:
  role: str
    The role associated with the tokens (e.g., 'user', 'model').
  token_ids: list[int]
    A list of token IDs.
  tokens: str
    The tokenized text.

TokensInfoDict:
  role: str
    The role associated with the tokens (e.g., 'user', 'model').
  token_ids: list[int]
    A list of token IDs.
  tokens: str
    The tokenized text.
```

----------------------------------------

TITLE: Dataset Distribution Bucket Details
DESCRIPTION: Represents a single bucket within a dataset's distribution, detailing the count of items and the range (left and right bounds). Supports both object and dictionary representations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DatasetDistributionDistributionBucket:
  count: The number of items in this bucket.
  left: The left boundary of the bucket.
  right: The right boundary of the bucket.

DatasetDistributionDistributionBucketDict:
  count: The number of items in this bucket.
  left: The left boundary of the bucket.
  right: The right boundary of the bucket.
```

----------------------------------------

TITLE: Generate Content with Caches
DESCRIPTION: Generates content using a model, referencing a previously created cache. This allows the model to leverage cached information for its response.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)
```

----------------------------------------

TITLE: ModalityTokenCountDict Fields
DESCRIPTION: Provides access to modality and token count information within the ModalityTokenCountDict.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ModalityTokenCountDict:
  modality: The modality of the token count.
  token_count: The number of tokens for the given modality.
```

----------------------------------------

TITLE: Safety Settings
DESCRIPTION: Configures safety settings for generative AI models to control the output of harmful content. This involves defining thresholds for different categories of harm.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

# Example safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
]

# When configuring a model:
# model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
```

----------------------------------------

TITLE: BlockedReason Enum Values
DESCRIPTION: Enumeration for reasons why content might be blocked. Includes unspecified, blocklist violations, image safety issues, and other reasons.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BlockedReason:
  BLOCKED_REASON_UNSPECIFIED: The reason for blocking is not specified.
  BLOCKLIST: Content was blocked due to being on a blocklist.
  IMAGE_SAFETY: Content was blocked due to image safety policies.
  OTHER: Content was blocked for a reason not otherwise specified.
```

----------------------------------------

TITLE: FileStatusDict Fields
DESCRIPTION: Details the fields within FileStatusDict, which is a dictionary representation of file status information, mirroring the FileStatus structure.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileStatusDict:
  code: The status code of the operation.
  details: Additional details about the status.
  message: A message describing the status.
```

----------------------------------------

TITLE: RecontextImageResponse Types
DESCRIPTION: Details the structure for RecontextImageResponse and its dictionary representation, focusing on generated images.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RecontextImageResponse:
  generated_images: List of generated images.

RecontextImageResponseDict:
  generated_images: List of generated images (dictionary format).
```

----------------------------------------

TITLE: AutomaticFunctionCallingConfigDict Properties
DESCRIPTION: Specifies the dictionary-based configuration for automatic function calling, mirroring the properties of AutomaticFunctionCallingConfig for disabling the feature, ignoring call history, and setting a maximum for remote calls.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AutomaticFunctionCallingConfigDict:
  disable: bool
    Description: Whether to disable automatic function calling.
  ignore_call_history: bool
    Description: Whether to ignore the call history when making function calls.
  maximum_remote_calls: int
    Description: The maximum number of remote function calls allowed.
```

----------------------------------------

TITLE: Python GenAI: Function Calling with ANY Tool Config Mode (Disable Automatic)
DESCRIPTION: Demonstrates how to configure function calling mode to 'ANY' and simultaneously disable automatic function calling. This ensures the model always returns function calls, but requires manual invocation.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: HarmBlockMethod API
DESCRIPTION: Defines methods for blocking harmful content, including unspecified, probability-based, and severity-based blocking.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HarmBlockMethod:
  HARM_BLOCK_METHOD_UNSPECIFIED: Unspecified harm block method.
  PROBABILITY: Block based on probability.
  SEVERITY: Block based on severity.
```

----------------------------------------

TITLE: ReplayResponse Types
DESCRIPTION: Defines the structure for ReplayResponse and its dictionary representation, including body segments, headers, SDK response segments, and status code.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ReplayResponse:
  body_segments: Segments of the response body.
  headers: Headers of the response.
  sdk_response_segments: Segments of the SDK response.
  status_code: The HTTP status code.

ReplayResponseDict:
  body_segments: Segments of the response body (dictionary format).
  headers: Headers of the response (dictionary format).
  sdk_response_segments: Segments of the SDK response (dictionary format).
  status_code: The HTTP status code (dictionary format).
```

----------------------------------------

TITLE: PreTunedModel Attributes
DESCRIPTION: Represents a pre-tuned model, including its base model, checkpoint ID, and tuned model name.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
PreTunedModel:
  base_model: The base model used for tuning.
  checkpoint_id: The ID of the model checkpoint.
  tuned_model_name: The name of the tuned model.
```

----------------------------------------

TITLE: FileState Enumerations
DESCRIPTION: Specifies the possible states of a file during processing. This helps track the lifecycle of a file within the system.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileState:
  ACTIVE: The file is active and ready for use.
  FAILED: The file processing failed.
  PROCESSING: The file is currently being processed.
  STATE_UNSPECIFIED: The state of the file is unspecified.
```

----------------------------------------

TITLE: TestTableItem Attributes
DESCRIPTION: Outlines the attributes for TestTableItem, representing individual test items. This includes configurations for exceptions in different environments, union handling, ignored keys, names, replay ID overrides, parameters, and skip conditions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import TestTableItem

# Accessing attributes
exception_if_mldev: bool = TestTableItem.exception_if_mldev
exception_if_vertex: bool = TestTableItem.exception_if_vertex
has_union: bool = TestTableItem.has_union
ignore_keys: list[str] = TestTableItem.ignore_keys
name: str = TestTableItem.name
override_replay_id: str = TestTableItem.override_replay_id
parameters: dict = TestTableItem.parameters
skip_in_api_mode: bool = TestTableItem.skip_in_api_mode
```

----------------------------------------

TITLE: Safety Setting Properties
DESCRIPTION: Defines the properties of a SafetySetting object, including category, method, and threshold for safety configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
SafetySetting.category
SafetySetting.method
SafetySetting.threshold
```

----------------------------------------

TITLE: Chat: Send Message (Asynchronous Streaming)
DESCRIPTION: Initiates an asynchronous chat session and sends a message to the model, receiving the response in streaming chunks. Each chunk's text is printed as it arrives.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: RAG Hybrid Search Configuration
DESCRIPTION: Configures hybrid search behavior, balancing keyword and vector search. The 'alpha' parameter determines the weighting between the two search methods.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigHybridSearch:
  alpha: float
    Weighting factor for hybrid search (0.0 to 1.0).

RagRetrievalConfigHybridSearchDict:
  alpha: float
    Weighting factor for hybrid search (0.0 to 1.0).
```

----------------------------------------

TITLE: Google Generative AI Chats API
DESCRIPTION: Handles chat interactions, including sending messages and managing chat sessions. Supports both asynchronous and synchronous operations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.chats.AsyncChat:
  send_message(message: str):
    Sends a message to the chat asynchronously.
  send_message_stream(message: str):
    Sends a message to the chat and streams the response asynchronously.

genai.chats.AsyncChats:
  create(): Creates a new chat session asynchronously.

genai.chats.Chat:
  send_message(message: str):
    Sends a message to the chat.
  send_message_stream(message: str):
    Sends a message to the chat and streams the response.

genai.chats.Chats:
  create(): Creates a new chat session.
```

----------------------------------------

TITLE: GenerateImagesResponse Attributes
DESCRIPTION: Details the attributes available in the GenerateImagesResponse and GenerateImagesResponseDict for image generation results. Includes safety attributes, HTTP response details, and generated images.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateImagesResponse:
  positive_prompt_safety_attributes: Safety attributes related to the positive prompt.
  sdk_http_response: The HTTP response from the SDK.
  images: The generated images.

GenerateImagesResponseDict:
  generated_images: The generated images.
  positive_prompt_safety_attributes: Safety attributes related to the positive prompt.
  sdk_http_response: The HTTP response from the SDK.
```

----------------------------------------

TITLE: RetrievalDict Attributes
DESCRIPTION: Details the attributes of the RetrievalDict type, a dictionary representation for retrieval configurations. Mirrors the Retrieval type with options for disabling attribution, external APIs, and Vertex AI search or RAG store.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RetrievalDict:
  disable_attribution: Boolean indicating if attribution should be disabled.
  external_api: Configuration for an external API retrieval.
  vertex_ai_search: Configuration for Vertex AI search retrieval.
  vertex_rag_store: Configuration for Vertex AI RAG store retrieval.
```

----------------------------------------

TITLE: Count Tokens and Compute Tokens
DESCRIPTION: Provides utilities to count the number of tokens in a given text or prompt, and to compute token usage for specific models. This is crucial for managing API costs and understanding model input limits.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('gemini-pro')

# Count tokens in a text
text = "This is a sample text to count tokens."
response = model.count_tokens(text)
print(f"Number of tokens: {response.total_tokens}")

# Compute tokens for a prompt (example)
# prompt = "Generate a story about a dragon."
# response = model.count_tokens(prompt)
# print(f"Tokens for prompt: {response.total_tokens}")
```

----------------------------------------

TITLE: SessionResumptionConfig Properties
DESCRIPTION: Details the properties available for the SessionResumptionConfig type in the Python GenAI library. These properties configure session resumption.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SessionResumptionConfig:
  handle: The handle for session resumption.
  transparent: Whether session resumption is transparent.
```

----------------------------------------

TITLE: HttpResponse and HttpResponseDict
DESCRIPTION: Represents an HTTP response, containing the response body and headers. HttpResponseDict is the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HttpResponse:
  body: any
    The body of the HTTP response.
  headers: dict
    The headers of the HTTP response.

HttpResponseDict:
  body: any
    The body of the HTTP response.
  headers: dict
    The headers of the HTTP response.
```

----------------------------------------

TITLE: File Object Attributes
DESCRIPTION: Attributes of the File object, providing metadata about uploaded files.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
File:
  create_time: datetime
    The timestamp when the file was created.
  display_name: str
    The display name of the file.
  download_uri: str
    The URI to download the file.
  error: Error
    Details about any errors encountered during file processing.
  expiration_time: datetime
    The timestamp when the file will expire.
  mime_type: str
    The MIME type of the file.
  name: str
    The unique identifier for the file.
  sha256_hash: str
    The SHA256 hash of the file content.
  size_bytes: int
    The size of the file in bytes.
  source: int
    The source of the file (enum).
  state: int
    The current state of the file (enum).
  update_time: datetime
    The timestamp when the file was last updated.
  uri: str
    The URI of the file.
  video_metadata: VideoMetadata
    Metadata specific to video files.
```

----------------------------------------

TITLE: Raw Reference Image Data
DESCRIPTION: Represents raw image data used as a reference in the RAG system. Includes a unique identifier, the image data itself, and its type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RawReferenceImage:
  reference_id: str
    A unique identifier for the reference image.
  reference_image: bytes
    The raw image data in bytes.
  reference_type: str
    The type of the reference image (e.g., 'image/jpeg').

RawReferenceImageDict:
  reference_id: str
    A unique identifier for the reference image.
  reference_image: bytes
    The raw image data in bytes.
  reference_type: str
    The type of the reference image (e.g., 'image/jpeg').
```

----------------------------------------

TITLE: RawReferenceImageDict and RecontextImageResponse
DESCRIPTION: Defines the structure for raw reference image dictionaries and the response for recontextualized images. These are used for image processing and analysis within the GenAI library.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RawReferenceImageDict:
  reference_type: The type of reference image.

RecontextImageResponse:
  (No specific fields detailed in the provided text, implies a response object for recontextualized images.)
```

----------------------------------------

TITLE: UpdateCachedContentConfig
DESCRIPTION: Configuration for updating cached content, including expiration time, TTL, and HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UpdateCachedContentConfig:
  Configuration for updating cached content.
  Attributes:
    expire_time: The expiration time for the cached content.
    http_options: HTTP options for the cache.
    ttl: Time-to-live for the cached content.
```

----------------------------------------

TITLE: TunedModelInfo Type
DESCRIPTION: Provides information about a tuned model, including its base model and creation/update times.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TunedModelInfo:
  base_model: The base model used for tuning.
  create_time: The timestamp when the tuned model was created.
  update_time: The timestamp when the tuned model was last updated.
```

----------------------------------------

TITLE: CandidateDict Type Attributes
DESCRIPTION: Details the attributes available for the CandidateDict type, which is a dictionary representation of a Candidate. These attributes mirror those of the Candidate type and provide access to generation details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CandidateDict:
  avg_logprobs: Average log probabilities of the generated content.
  citation_metadata: Metadata related to citations in the generated content.
  content: The generated content itself.
  finish_message: A message indicating why the generation finished.
  finish_reason: The reason for the generation finishing (e.g., STOP, MAX_TOKENS).
  grounding_metadata: Metadata related to grounding the generated content.
  index: The index of the candidate.
  logprobs_result: The result of log probability calculations.
  safety_ratings: Safety ratings for the generated content.
  token_count: The number of tokens in the generated content.
  url_context_metadata: Metadata related to URL context.
```

----------------------------------------

TITLE: BatchJobSourceDict Attributes
DESCRIPTION: Defines the attributes for BatchJobSourceDict, similar to BatchJobSource, for specifying batch job data sources. Includes BigQuery URIs, GCS URIs, file names, data format, and inlined requests.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchJobSourceDict:
  bigquery_uri: URI for BigQuery data source.
  file_name: Name of the file containing the data.
  format: Format of the data (e.g., CSV, JSON).
  gcs_uri: URI for Google Cloud Storage data source.
  inlined_requests: Direct inclusion of requests within the source.
```

----------------------------------------

TITLE: Monitor Tuning Job Completion
DESCRIPTION: Polls the status of a tuning job at regular intervals (10 seconds) until it reaches a completed state (SUCCEEDED, FAILED, or CANCELLED). It prints the current state during the process.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
import time

completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
    ]
)

while tuning_job.state not in completed_states:
    print(tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)
```

----------------------------------------

TITLE: Audio Chunk Data Types
DESCRIPTION: Defines the structure for audio data chunks, including the raw data, MIME type, and any associated source metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AudioChunk:
  data: The audio data itself.
  mime_type: The MIME type of the audio data (e.g., 'audio/wav').
  source_metadata: Metadata related to the source of the audio data.

AudioChunkDict:
  data: The audio data as a dictionary.
  mime_type: The MIME type of the audio data.
  source_metadata: Metadata related to the source of the audio data.
```

----------------------------------------

TITLE: Generate Enum Response Schema (Text)
DESCRIPTION: Demonstrates generating content where the response is one of the enum values, returned as plain text. This is useful for constrained text outputs.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from enum import Enum

class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

----------------------------------------

TITLE: SchemaDict Type Properties
DESCRIPTION: Details the various properties available for the SchemaDict type in the Python GenAI library. These properties are similar to Schema but are intended for dictionary-based schema representations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SchemaDict:
  additional_properties: Defines constraints on additional properties not explicitly defined in the schema.
  any_of: Specifies that the schema must be valid against at least one of the subschemas.
  default: Sets a default value for the property.
  defs: Allows defining reusable schema fragments.
  description: Provides a human-readable description of the schema.
  enum: Restricts the value to a specific set of allowed values.
  example: Provides an example value for the property.
  format: Specifies a format for the property (e.g., 'date-time', 'email').
  max_items: Sets the maximum number of items allowed in an array.
  max_length: Sets the maximum length for a string.
  max_properties: Sets the maximum number of properties allowed in an object.
  maximum: Sets the maximum allowed value for a number.
  min_items: Sets the minimum number of items allowed in an array.
  min_length: Sets the minimum length for a string.
  min_properties: Sets the minimum number of properties allowed in an object.
  minimum: Sets the minimum allowed value for a number.
  nullable: Indicates if the property can be null.
  pattern: Specifies a regular expression pattern for a string.
  properties: Defines the properties of an object.
  property_ordering: Specifies the order of properties.
  ref: References another schema definition.
  required: Lists properties that are required in an object.
  title: Provides a title for the schema.
  type: Specifies the data type of the property (e.g., 'string', 'integer', 'object').
```

----------------------------------------

TITLE: GoogleTypeDate Attributes
DESCRIPTION: Details the components of the GoogleTypeDate structure, including day, month, and year for representing dates.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GoogleTypeDate:
  day: The day of the month (1-31).
  month: The month of the year (1-12).
  year: The year.
```

----------------------------------------

TITLE: CodeExecutionResultDict Attributes
DESCRIPTION: Describes the CodeExecutionResultDict type, which holds the outcome and output of code execution.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CodeExecutionResultDict:
  outcome: str
    The outcome of the code execution.
  output: str
    The output generated by the code execution.
```

----------------------------------------

TITLE: Generate Content (Asynchronous Non-Streaming)
DESCRIPTION: Generates text content from a prompt asynchronously without streaming. The complete response is returned once generation is finished.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai
import asyncio

async def generate_content_async():
    model = genai.GenerativeModel('gemini-pro')
    response = await model.generate_content_async('Explain the concept of recursion.')
    print(response.text)

# To run this:
# asyncio.run(generate_content_async())
```

----------------------------------------

TITLE: FeatureSelectionPreference
DESCRIPTION: Enumeration for specifying feature selection preferences. Allows balancing cost, quality, or prioritizing one over the other.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FeatureSelectionPreference:
  BALANCED: Select features to balance cost and quality.
  FEATURE_SELECTION_PREFERENCE_UNSPECIFIED: Default value, the feature selection preference is not set.
  PRIORITIZE_COST: Prioritize lower cost when selecting features.
  PRIORITIZE_QUALITY: Prioritize higher quality when selecting features.
```

----------------------------------------

TITLE: EmbedContentResponse and EmbedContentResponseDict
DESCRIPTION: Represents the response from an embed content operation, including the generated embeddings, metadata, and the raw SDK HTTP response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EmbedContentResponse:
  embeddings: A list of embedding vectors generated for the content.
  metadata: Metadata related to the embedding operation.
  sdk_http_response: The raw HTTP response object from the SDK.

EmbedContentResponseDict:
  embeddings: A list of embedding vectors generated for the content.
  metadata: Metadata related to the embedding operation.
  sdk_http_response: The raw HTTP response object from the SDK.
```

----------------------------------------

TITLE: Chat: Send Message (Synchronous Non-Streaming)
DESCRIPTION: Demonstrates sending a message to a chat model and receiving a response. This is a synchronous, non-streaming method suitable for single turn interactions or when immediate full responses are needed.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
chat = client.chats.create(model='gemini-2.0-flash-001')
response = chat.send_message('tell me a story')
print(response.text)
response = chat.send_message('summarize the story you told me in 1 sentence')
print(response.text)
```

----------------------------------------

TITLE: FileDict Attributes
DESCRIPTION: Details the attributes available for the FileDict object, which represents file metadata. These include download URI, error information, expiration time, MIME type, name, hash, size, source, state, update time, URI, and video metadata.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileDict:
  download_uri: The URI to download the file.
  error: Information about any errors encountered during file processing.
  expiration_time: The time when the file will expire.
  mime_type: The MIME type of the file.
  name: The name of the file.
  sha256_hash: The SHA256 hash of the file.
  size_bytes: The size of the file in bytes.
  source: The source of the file (e.g., UPLOADED, GENERATED).
  state: The current state of the file (e.g., ACTIVE, PROCESSING, FAILED).
  update_time: The time when the file was last updated.
  uri: The URI of the file.
  video_metadata: Metadata specific to video files.
```

----------------------------------------

TITLE: LatLng and LatLngDict Structures
DESCRIPTION: Represents geographical coordinates using latitude and longitude. LatLng is a structured object, while LatLngDict is a dictionary-based representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LatLng:
  latitude: The latitude coordinate (float).
  longitude: The longitude coordinate (float).

LatLngDict:
  latitude: The latitude coordinate (float).
  longitude: The longitude coordinate (float).
```

----------------------------------------

TITLE: GroundingMetadataDict Attributes
DESCRIPTION: Details the attributes of the GroundingMetadataDict, a dictionary representation of grounding metadata. It encompasses context tokens, grounding chunks, supports, retrieval metadata, queries, search entry points, and web search queries.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingMetadataDict:
  google_maps_widget_context_token: Context token for Google Maps widget.
  grounding_chunks: A list of grounding chunks.
  grounding_supports: Information about grounding supports.
  retrieval_metadata: Metadata related to retrieval operations.
  retrieval_queries: Queries used for retrieval.
  search_entry_point: The entry point for search operations.
  web_search_queries: Queries used for web searches.
```

----------------------------------------

TITLE: List Tuning Jobs
DESCRIPTION: Retrieves a list of tuning jobs. This function is part of the core functionality for managing model tuning processes.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.generativeai.types import TuningJob

def list_tuning_jobs() -> list[TuningJob]:
    """Lists all tuning jobs.

    Returns:
        A list of TuningJob objects.
    """
    # Implementation details omitted for brevity
    pass
```

----------------------------------------

TITLE: GenerateContentResponseUsageMetadataDict Fields
DESCRIPTION: Details the fields available within the GenerateContentResponseUsageMetadataDict object, a dictionary representation of usage metadata. It mirrors the GenerateContentResponseUsageMetadata object, providing token counts and details for various components of the response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GenerateContentResponseUsageMetadataDict:
  cache_tokens_details: List of token details for cached content.
  cached_content_token_count: Token count for cached content.
  candidates_token_count: Token count for candidates.
  candidates_tokens_details: List of token details for candidates.
  prompt_token_count: Total prompt token count.
  prompt_tokens_details: List of token details for the prompt.
  thoughts_token_count: Token count for thoughts.
  tool_use_prompt_token_count: Token count for tool use in the prompt.
  tool_use_prompt_tokens_details: List of token details for tool use in the prompt.
  total_token_count: Total token count for the response.
  traffic_type: Type of traffic for the response.
```

----------------------------------------

TITLE: ReplayInteraction Types
DESCRIPTION: Defines the structure for ReplayInteraction and its dictionary representation, encompassing request and response data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ReplayInteraction:
  request: The request details.
  response: The response details.

ReplayInteractionDict:
  request: The request details (dictionary format).
  response: The response details (dictionary format).
```

----------------------------------------

TITLE: RAG Retrieval Configuration Filters
DESCRIPTION: Defines filtering options for RAG retrieval, including metadata filters and vector similarity/distance thresholds. These properties control how relevant documents are selected based on metadata and vector embeddings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagRetrievalConfigFilter:
  metadata_filter: dict | None
    Filters documents based on metadata.
  vector_distance_threshold: float | None
    Maximum allowed distance between query and document vectors.
  vector_similarity_threshold: float | None
    Minimum required similarity between query and document vectors.

RagRetrievalConfigFilterDict:
  metadata_filter: dict | None
    Filters documents based on metadata.
  vector_distance_threshold: float | None
    Maximum allowed distance between query and document vectors.
  vector_similarity_threshold: float | None
    Minimum required similarity between query and document vectors.
```

----------------------------------------

TITLE: ContentDict Attributes
DESCRIPTION: Details the ContentDict type, which holds content parts and role information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ContentDict:
  parts: list[str | dict | object]
    The parts of the content.
  role: str
    The role associated with the content (e.g., 'user', 'model').
```

----------------------------------------

TITLE: CitationMetadataDict Attributes
DESCRIPTION: Details the CitationMetadataDict type, which contains a list of citations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CitationMetadataDict:
  citations: list[CitationDict]
    A list of citation dictionaries.
```

----------------------------------------

TITLE: Candidate Type Attributes
DESCRIPTION: Details the attributes available for the Candidate type in the Python GenAI library. These attributes provide information about the generated content, including log probabilities, citations, finish reason, and safety ratings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Candidate:
  avg_logprobs: Average log probabilities of the generated content.
  citation_metadata: Metadata related to citations in the generated content.
  content: The generated content itself.
  finish_message: A message indicating why the generation finished.
  finish_reason: The reason for the generation finishing (e.g., STOP, MAX_TOKENS).
  grounding_metadata: Metadata related to grounding the generated content.
  index: The index of the candidate.
  logprobs_result: The result of log probability calculations.
  safety_ratings: Safety ratings for the generated content.
  token_count: The number of tokens in the generated content.
  url_context_metadata: Metadata related to URL context.
```

----------------------------------------

TITLE: HarmBlockThreshold API
DESCRIPTION: Specifies thresholds for blocking harmful content across different levels of risk.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HarmBlockThreshold:
  BLOCK_LOW_AND_ABOVE: Block low and above severity.
  BLOCK_MEDIUM_AND_ABOVE: Block medium and above severity.
  BLOCK_NONE: No blocking.
  BLOCK_ONLY_HIGH: Block only high severity.
  HARM_BLOCK_THRESHOLD_UNSPECIFIED: Unspecified harm block threshold.
  OFF: Harm blocking is turned off.
```

----------------------------------------

TITLE: BatchJobSource Attributes
DESCRIPTION: Details the attributes for the BatchJobSource data structure, specifying the source of data for a batch job. Supports BigQuery URIs, GCS URIs, file names, data format, and inlined requests.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BatchJobSource:
  bigquery_uri: URI for BigQuery data source.
  file_name: Name of the file containing the data.
  format: Format of the data (e.g., CSV, JSON).
  gcs_uri: URI for Google Cloud Storage data source.
  inlined_requests: Direct inclusion of requests within the source.
```

----------------------------------------

TITLE: TunedModelInfoDict Type
DESCRIPTION: A dictionary representation of tuned model information, mirroring the TunedModelInfo type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TunedModelInfoDict:
  base_model: The base model used for tuning.
  create_time: The timestamp when the tuned model was created.
  update_time: The timestamp when the tuned model was last updated.
```

----------------------------------------

TITLE: ContentEmbeddingDict Attributes
DESCRIPTION: Describes the ContentEmbeddingDict type, containing statistics for content embeddings.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ContentEmbeddingDict:
  statistics: ContentEmbeddingStatistics
    Statistics related to the content embedding.
```

----------------------------------------

TITLE: AutomaticFunctionCallingConfig Properties
DESCRIPTION: Defines the configuration options for automatic function calling, including settings to disable the feature, ignore call history, and set a maximum number of remote calls.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AutomaticFunctionCallingConfig:
  disable: bool
    Description: Whether to disable automatic function calling.
  ignore_call_history: bool
    Description: Whether to ignore the call history when making function calls.
  maximum_remote_calls: int
    Description: The maximum number of remote function calls allowed.
```

----------------------------------------

TITLE: TunedModel Type
DESCRIPTION: Information about a tuned model, including its checkpoints, endpoint, and base model.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TunedModel:
  checkpoints: List of checkpoints for the tuned model.
  endpoint: The endpoint associated with the tuned model.
  model: The base model used for tuning.
```

----------------------------------------

TITLE: Authentication Token Information
DESCRIPTION: Represents an authentication token, typically containing a name. This is used to identify the type or purpose of the token.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
AuthToken:
  name: str
    The name of the authentication token.

AuthTokenDict:
  name: str
    The name of the authentication token.
```

----------------------------------------

TITLE: TuningMode Enum
DESCRIPTION: Defines the different modes available for tuning operations, such as full tuning or PEFT adapter tuning.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningMode:
  TUNING_MODE_FULL: Represents full tuning mode.
  TUNING_MODE_PEFT_ADAPTER: Represents PEFT adapter tuning mode.
  TUNING_MODE_UNSPECIFIED: Represents an unspecified tuning mode.
```

----------------------------------------

TITLE: Cached Content Configuration
DESCRIPTION: Configuration options for deleting cached content, including HTTP options.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DeleteCachedContentConfig:
  http_options: dict

DeleteCachedContentConfigDict:
  http_options: dict

DeleteCachedContentResponse:
  (No specific fields documented)

DeleteCachedContentResponseDict:
  (No specific fields documented)
```

----------------------------------------

TITLE: ModalityTokenCountDict API
DESCRIPTION: Represents ModalityTokenCount as a dictionary, providing access to the modality and its token count.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ModalityTokenCountDict:
  modality: The media modality.
  token_count: The number of tokens for the modality.
```

----------------------------------------

TITLE: FileDict Object Attributes
DESCRIPTION: Dictionary representation of the File object, providing metadata about uploaded files.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileDict:
  create_time: datetime
    The timestamp when the file was created.
  display_name: str
    The display name of the file.
```

----------------------------------------

TITLE: ExternalApiElasticSearchParamsDict
DESCRIPTION: Defines parameters for Elasticsearch-based searches within the External API. Includes settings for the number of hits and the search template.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ExternalApiElasticSearchParamsDict:
  num_hits: int
    The number of search hits to return.
  search_template: str
    The template to use for the search query.
```

----------------------------------------

TITLE: ListCachedContentsResponseDict Fields
DESCRIPTION: Details the dictionary representation of ListCachedContentsResponse, providing access to cached_contents, next_page_token, and sdk_http_response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
class ListCachedContentsResponseDict:
    cached_contents: list
    next_page_token: str
    sdk_http_response: google.api_core.http_response.HTTPResponse
```

----------------------------------------

TITLE: FunctionDeclaration Dictionary Representation
DESCRIPTION: Details the dictionary representation of a FunctionDeclaration, including its properties like behavior, description, name, and parameter schemas.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FunctionDeclarationDict:
  A dictionary representation of a FunctionDeclaration.
  - behavior: Description of the function's behavior.
  - description: A detailed description of the function.
  - name: The name of the function.
  - parameters: Schema defining the function's parameters.
  - parameters_json_schema: JSON schema for the function's parameters.
```

----------------------------------------

TITLE: RagChunkPageSpan Fields
DESCRIPTION: Details the fields available for RagChunkPageSpan, specifying the first and last page of a chunk.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagChunkPageSpan:
  first_page: The first page number of the chunk.
  last_page: The last page number of the chunk.
```

----------------------------------------

TITLE: Delete Model Operations
DESCRIPTION: Defines data structures for configuring and responding to model deletion requests. Includes HTTP options for requests and details about the deletion response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DeleteModelConfigDict:
  http_options: Options for HTTP requests related to model deletion.

DeleteModelResponse:
  Represents the response after a model deletion operation.

DeleteModelResponseDict:
  A dictionary representation of the DeleteModelResponse.
```

----------------------------------------

TITLE: GroundingChunkRetrievedContextDict Attributes
DESCRIPTION: Details the attributes of the GroundingChunkRetrievedContextDict, a dictionary representation of retrieved context. It mirrors the attributes of GroundingChunkRetrievedContext, providing access to the retrieved chunk, text, title, and URI.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkRetrievedContextDict:
  rag_chunk: The retrieved chunk object.
  text: The text content of the retrieved chunk.
  title: The title of the retrieved chunk.
  uri: The URI associated with the retrieved chunk.
```

----------------------------------------

TITLE: RagChunk Fields
DESCRIPTION: Details the fields available for RagChunk, including text content and page span information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagChunk:
  page_span: Information about the page span of the chunk.
  text: The text content of the chunk.
```

----------------------------------------

TITLE: Dataset Statistics
DESCRIPTION: Contains overall statistics for a dataset, specifically the total billable character count.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
DatasetStats:
  total_billable_character_count: The total number of billable characters in the dataset.
```

----------------------------------------

TITLE: FileStatus Fields
DESCRIPTION: Describes the fields within FileStatus, which provides information about the status of an operation, typically related to file processing.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileStatus:
  code: The status code of the operation.
  details: Additional details about the status.
  message: A message describing the status.
```

----------------------------------------

TITLE: Schema Type Properties
DESCRIPTION: Details the various properties available for the Schema type in the Python GenAI library. These properties correspond to standard JSON Schema keywords used for defining data validation and structure.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Schema:
  additional_properties: Defines constraints on additional properties not explicitly defined in the schema.
  any_of: Specifies that the schema must be valid against at least one of the subschemas.
  default: Sets a default value for the property.
  defs: Allows defining reusable schema fragments.
  description: Provides a human-readable description of the schema.
  enum: Restricts the value to a specific set of allowed values.
  example: Provides an example value for the property.
  format: Specifies a format for the property (e.g., 'date-time', 'email').
  items: Defines the schema for items in an array.
  max_items: Sets the maximum number of items allowed in an array.
  max_length: Sets the maximum length for a string.
  max_properties: Sets the maximum number of properties allowed in an object.
  maximum: Sets the maximum allowed value for a number.
  min_items: Sets the minimum number of items allowed in an array.
  min_length: Sets the minimum length for a string.
  min_properties: Sets the minimum number of properties allowed in an object.
  minimum: Sets the minimum allowed value for a number.
  nullable: Indicates if the property can be null.
  pattern: Specifies a regular expression pattern for a string.
  properties: Defines the properties of an object.
  property_ordering: Specifies the order of properties.
  ref: References another schema definition.
  required: Lists properties that are required in an object.
  title: Provides a title for the schema.
  type: Specifies the data type of the property (e.g., 'string', 'integer', 'object').
```

----------------------------------------

TITLE: RagChunkDict Fields
DESCRIPTION: Details the fields available for RagChunkDict, including text content and page span information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagChunkDict:
  page_span: Information about the page span of the chunk.
  text: The text content of the chunk.
```

----------------------------------------

TITLE: Data Types in genai.types
DESCRIPTION: Defines various data structures and enumerations used within the genai library for representing different types of data and configurations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
genai.types.ActivityEnd
  - Represents the end of an activity.

genai.types.ActivityEndDict
  - Dictionary representation of ActivityEnd.

genai.types.ActivityHandling
  - Enum for handling activity interruptions.
  - Values:
    - ACTIVITY_HANDLING_UNSPECIFIED: Unspecified activity handling.
    - NO_INTERRUPTION: No interruption during activity.
    - START_OF_ACTIVITY_INTERRUPTS: Interrupts at the start of an activity.

genai.types.ActivityStart
  - Represents the start of an activity.

genai.types.ActivityStartDict
  - Dictionary representation of ActivityStart.

genai.types.AdapterSize
  - Enum for specifying adapter sizes.
  - Values:
    - ADAPTER_SIZE_UNSPECIFIED: Unspecified adapter size.
    - ADAPTER_SIZE_ONE: Adapter size of 1.
    - ADAPTER_SIZE_TWO: Adapter size of 2.
    - ADAPTER_SIZE_FOUR: Adapter size of 4.
    - ADAPTER_SIZE_EIGHT: Adapter size of 8.
    - ADAPTER_SIZE_SIXTEEN: Adapter size of 16.
    - ADAPTER_SIZE_THIRTY_TWO: Adapter size of 32.

genai.types.ApiAuth
  - Represents API authentication configuration.
  - Attributes:
    - api_key_config: Configuration for API key authentication.

genai.types.ApiAuthApiKeyConfig
  - Configuration for API key authentication.
  - Attributes:
    - api_key_secret_version: The version of the API key secret.
    - api_key_string: The API key string.

genai.types.ApiAuthApiKeyConfigDict
  - Dictionary representation of ApiAuthApiKeyConfig.
  - Attributes:
    - api_key_secret_version: The version of the API key secret.
    - api_key_string: The API key string.

genai.types.ApiAuthDict
  - Dictionary representation of ApiAuth.
  - Attributes:
    - api_key_config: Configuration for API key authentication.

genai.types.ApiKeyConfig
  - Configuration for API key.
```

----------------------------------------

TITLE: JSONSchemaType Enum Documentation
DESCRIPTION: Documentation for the JSONSchemaType enum, providing constants for JSON data types.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
JSONSchemaType:
  Enum for JSON data types.
  Members:
    ARRAY: Represents an array.
    BOOLEAN: Represents a boolean.
    INTEGER: Represents an integer.
    NULL: Represents a null value.
    NUMBER: Represents a number.
    OBJECT: Represents an object.
    STRING: Represents a string.
```

----------------------------------------

TITLE: UsageMetadataDict Structure
DESCRIPTION: A dictionary representation of UsageMetadata, providing similar information about model usage including token counts and details. This structure is often used for data serialization or deserialization.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UsageMetadataDict:
  prompt_token_count: int
    The number of tokens in the prompt.
  response_token_count: int
    The number of tokens in the response.
  cached_content_token_count: int
    The number of tokens from cached content.
  total_token_count: int
    The total number of tokens used.
  prompt_tokens_details: list[TokenCountDetails]
    Details about prompt token usage.
  response_tokens_details: list[TokenCountDetails]
    Details about response token usage.
  cache_tokens_details: list[TokenCountDetails]
    Details about cache token usage.
  thoughts_token_count: int
    The number of tokens used for thoughts.
  tool_use_prompt_token_count: int
    The number of tokens used for tool use prompts.
  tool_use_prompt_tokens_details: list[TokenCountDetails]
    Details about tool use prompt token usage.
  traffic_type: TrafficType
    The type of traffic for this usage.
```

----------------------------------------

TITLE: InlinedRequest Type
DESCRIPTION: Defines the structure for an inlined request, containing model configuration and content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
InlinedRequest:
  config: Configuration for the model.
  contents: The content of the request.
  model: The model to use for the request.
```

----------------------------------------

TITLE: LiveServerToolCallCancellation and LiveServerToolCallCancellationDict
DESCRIPTION: Represents the cancellation of a tool call in a live server. Specifies the IDs of the calls to be cancelled.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerToolCallCancellation:
  ids: List[str]
    A list of tool call IDs to cancel.

LiveServerToolCallCancellationDict:
  ids: List[str]
    A list of tool call IDs to cancel.
```

----------------------------------------

TITLE: RagChunkPageSpanDict Fields
DESCRIPTION: Details the fields available for RagChunkPageSpanDict, specifying the first and last page of a chunk.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
RagChunkPageSpanDict:
  first_page: The first page number of the chunk.
  last_page: The last page number of the chunk.
```

----------------------------------------

TITLE: LiveMusicFilteredPromptDict.text
DESCRIPTION: Represents the text content within a LiveMusicFilteredPromptDict, likely used for filtering or providing prompts for music generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveMusicFilteredPromptDict.text:
  text: str
    The text content of the prompt.
```

----------------------------------------

TITLE: Enum Response Schema Support
DESCRIPTION: Enables the model to return responses that conform to a specified enum (enumeration) type, ensuring the output is one of a predefined set of values.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

# Define an enum for response
class Mood(str):
    HAPPY = 'happy'
    SAD = 'sad'
    NEUTRAL = 'neutral'

# Configure the model to return enum
model = genai.GenerativeModel('gemini-pro', response_enum_type=Mood)

# Request a response that is one of the Mood enum values
# response = model.generate_content("Describe your current mood.")
# print(response.text) # Expected output: 'happy', 'sad', or 'neutral'
```

----------------------------------------

TITLE: TunedModelCheckpointDict Type
DESCRIPTION: A dictionary representation of a tuned model checkpoint, mirroring the TunedModelCheckpoint type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TunedModelCheckpointDict:
  checkpoint_id: The unique identifier for the checkpoint.
  endpoint: The endpoint associated with this checkpoint.
  epoch: The epoch number for this checkpoint.
  step: The training step number for this checkpoint.
```

----------------------------------------

TITLE: Create Batch Prediction Job (Inlined Requests)
DESCRIPTION: Creates a batch prediction job with inlined requests, where the input data is provided directly within the request payload. This is useful for smaller, ad-hoc batch predictions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
# Create a batch job with inlined requests
batch_job = client.batches.create(
    model="gemini-2.0-flash",
    src=[
      {
        "contents": [
          {
            "parts": [
              {
                "text": "Hello!",
              }
            ],
            "role": "user",
          }
        ],
        "config:": {"response_modalities": ["text"]},
      }
    ],
)
```

----------------------------------------

TITLE: CheckpointDict Type Attributes
DESCRIPTION: Details the attributes for the CheckpointDict type, a dictionary representation of a Checkpoint. It provides access to the checkpoint's ID, epoch, and step.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CheckpointDict:
  checkpoint_id: The unique identifier for the checkpoint.
  epoch: The epoch number at which the checkpoint was saved.
  step: The training step number at which the checkpoint was saved.
```

----------------------------------------

TITLE: JobState Enum
DESCRIPTION: Defines the possible states for a job within the GenAI system. These states indicate the current lifecycle stage of a batch job.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
JobState:
  JOB_STATE_UNSPECIFIED: Job state is not specified.
  JOB_STATE_QUEUED: Job is queued and waiting to start.
  JOB_STATE_PENDING: Job is pending, preparing to run.
  JOB_STATE_RUNNING: Job is currently executing.
  JOB_STATE_SUCCEEDED: Job completed successfully.
  JOB_STATE_PARTIALLY_SUCCEEDED: Job completed with some partial successes.
  JOB_STATE_FAILED: Job failed to complete.
  JOB_STATE_CANCELLED: Job was cancelled by the user or system.
  JOB_STATE_CANCELLING: Job is in the process of being cancelled.
  JOB_STATE_PAUSED: Job execution has been temporarily paused.
  JOB_STATE_EXPIRED: Job has expired due to inactivity or time limits.
  JOB_STATE_UPDATING: Job is currently being updated.
```

----------------------------------------

TITLE: Edit Image (Vertex AI) (Python)
DESCRIPTION: Edits an image based on a text prompt and reference images, using a separate model. Supports inpainting and insertion modes, and is only available in Vertex AI.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()
```

----------------------------------------

TITLE: List Batch Prediction Jobs (Asynchronous)
DESCRIPTION: Asynchronously lists batch prediction jobs, enabling non-blocking operations. It utilizes async iterators for efficient retrieval and management of job data.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)
```

----------------------------------------

TITLE: Embed Content (Single and Multiple) (Python)
DESCRIPTION: Generates embeddings for text content. Supports single or multiple content inputs and allows specifying output dimensionality for embeddings.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)
```

LANGUAGE: python
CODE:
```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)
```

----------------------------------------

TITLE: Streaming Text Content Generation
DESCRIPTION: Generates text content in a streaming format, allowing the model's output to be received in chunks as it's produced, rather than waiting for the entire response.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')
```

----------------------------------------

TITLE: SupervisedTuningDatasetDistribution Attributes
DESCRIPTION: Details the attributes of the `SupervisedTuningDatasetDistribution` type, which describes the distribution of data within a supervised tuning dataset. It includes sums, buckets, and statistical measures like mean and max.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SupervisedTuningDatasetDistribution:
  billable_sum: int
    The sum of billable units (e.g., characters or tokens).
  buckets: list[dict]
    A list of buckets representing data distribution.
  max: float
    The maximum value in the distribution.
  mean: float
    The average value in the distribution.
```

----------------------------------------

TITLE: Python GenAI: Function Calls with Automatic Calling Disabled
DESCRIPTION: Demonstrates how to receive function call parts in the response when automatic function calling is disabled. This allows for manual inspection and handling of function calls.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)

function_calls: Optional[List[types.FunctionCall]] = response.function_calls
```

----------------------------------------

TITLE: MaskReferenceImage API
DESCRIPTION: Defines the structure and attributes for MaskReferenceImage, used for specifying reference images with masks. Includes configurations for the mask and reference image details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MaskReferenceImage:
  config: Configuration for the mask.
  mask_image_config: Configuration specific to the mask image.
  reference_id: Unique identifier for the reference image.
  reference_image: The reference image data.
  reference_type: The type of reference image provided.
```

----------------------------------------

TITLE: Upscale Image (Vertex AI) (Python)
DESCRIPTION: Upscales a previously generated image using the Imagen model. This feature is specifically supported in Vertex AI and allows specifying an upscale factor.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-002',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()
```

----------------------------------------

TITLE: Count Tokens (Asynchronous)
DESCRIPTION: Demonstrates how to count tokens asynchronously using `client.aio.models.count_tokens`. This allows for token counting operations without blocking the main thread.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)

```

----------------------------------------

TITLE: Delete File
DESCRIPTION: Demonstrates how to delete a previously uploaded file from the Generative AI API using its name.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)
```

----------------------------------------

TITLE: TestTableFileDict Attributes
DESCRIPTION: Details the attributes for TestTableFileDict, a dictionary representation for test table data. Similar to TestTableFile, it covers comments, parameter names, test methods, and the test table content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import TestTableFileDict

# Accessing attributes
comment: str = TestTableFileDict.comment
parameter_names: list[str] = TestTableFileDict.parameter_names
test_method: str = TestTableFileDict.test_method
test_table: list[dict] = TestTableFileDict.test_table
```

----------------------------------------

TITLE: Checkpoint Type Attributes
DESCRIPTION: Details the attributes for the Checkpoint type, which represents a model checkpoint. This includes the checkpoint ID, epoch, and step number.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Checkpoint:
  checkpoint_id: The unique identifier for the checkpoint.
  epoch: The epoch number at which the checkpoint was saved.
  step: The training step number at which the checkpoint was saved.
```

----------------------------------------

TITLE: Chat: Send Message (Asynchronous Non-Streaming)
DESCRIPTION: Shows how to send a message to a chat model asynchronously and receive a response. This non-streaming approach is suitable for applications that need to perform other tasks while waiting for the model's output.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
response = await chat.send_message('tell me a story')
print(response.text)
```

----------------------------------------

TITLE: GroundingChunkRetrievedContext Attributes
DESCRIPTION: Details the attributes of the GroundingChunkRetrievedContext object, which represents context retrieved from a retrieval system. It includes the original retrieved chunk, its text content, title, and URI.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GroundingChunkRetrievedContext:
  rag_chunk: The retrieved chunk object.
  text: The text content of the retrieved chunk.
  title: The title of the retrieved chunk.
  uri: The URI associated with the retrieved chunk.
```

----------------------------------------

TITLE: ExternalApi Type Attributes
DESCRIPTION: Defines an external API with various configuration parameters including authentication, endpoint, and search parameters. Includes dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import ExternalApi, ExternalApiDict, ExternalApiElasticSearchParams, ExternalApiElasticSearchParamsDict

# Example usage of ExternalApi
es_params = ExternalApiElasticSearchParams(index='my-index', num_hits=10, search_template='my-template')
external_api = ExternalApi(
    api_auth='api_key', 
    api_spec='openapi.yaml', 
    auth_config={'type': 'oauth2'},
    elastic_search_params=es_params,
    endpoint='https://api.example.com/v1',
    simple_search_params={'query': 'test'}
)

print(f"External API Endpoint: {external_api.endpoint}")
print(f"External API Auth: {external_api.api_auth}")
print(f"External API Elastic Search Index: {external_api.elastic_search_params.index}")

# Example usage of ExternalApiDict
es_params_dict = ExternalApiElasticSearchParamsDict({'index': 'another-index', 'num_hits': 5, 'search_template': 'another-template'})
external_api_dict = ExternalApiDict({
    'api_auth': 'bearer_token',
    'api_spec': 'swagger.json',
    'auth_config': {'type': 'api_key'},
    'elastic_search_params': es_params_dict,
    'endpoint': 'https://api.another.com/v2',
    'simple_search_params': {'query': 'sample'}
})

print(f"External API Dict Endpoint: {external_api_dict['endpoint']}")
print(f"External API Dict Auth: {external_api_dict['api_auth']}")
print(f"External API Dict Elastic Search Num Hits: {external_api_dict['elastic_search_params']['num_hits']}")
```

----------------------------------------

TITLE: TuningDataStats Type
DESCRIPTION: Statistics related to tuning data, including stats for distillation, preference optimization, and supervised tuning.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TuningDataStats:
  distillation_data_stats: Statistics for distillation data.
  preference_optimization_data_stats: Statistics for preference optimization data.
  supervised_tuning_data_stats: Statistics for supervised tuning data.
```

----------------------------------------

TITLE: EditImageResponse and EditImageResponseDict Fields
DESCRIPTION: Describes the fields within EditImageResponse and EditImageResponseDict, which contain the results of an image editing operation, including generated images and SDK HTTP response details.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EditImageResponse:
  generated_images: list[Image]
    A list of generated images.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response from the SDK.

EditImageResponseDict:
  generated_images: list[ImageDict]
    A list of generated images in dictionary format.
  sdk_http_response: google.api_core.http_response.HttpResponse
    The HTTP response from the SDK.
```

----------------------------------------

TITLE: Chat Functionality
DESCRIPTION: Enables conversational interactions with generative models. Users can send messages and receive responses, maintaining conversation history for context.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Send a message and get a response
response = chat.send_message("Hello, how are you?")
print(response.text)

# Continue the conversation
response = chat.send_message("What is the weather like today?")
print(response.text)

# Access conversation history
# print(chat.history)
```

----------------------------------------

TITLE: BlobDict Attributes
DESCRIPTION: Defines attributes for BlobDict, a dictionary representation of binary data. Includes the data content, a display name, and the MIME type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
BlobDict:
  data: The binary data content.
  display_name: A user-friendly name for the blob.
  mime_type: The MIME type of the data (e.g., 'image/png').
```

----------------------------------------

TITLE: GeneratedImage Fields
DESCRIPTION: Represents a generated image, including enhanced prompts and safety attributes.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GeneratedImage:
  enhanced_prompt: str | None
    The prompt after potential enhancements by the model.
  image: Image
    The generated image object.
  rai_filtered_reason: str | None
    The reason if the image was filtered by RAI.
  safety_attributes: SafetyAttributes
    Safety attributes associated with the generated image.
```

----------------------------------------

TITLE: ThinkingConfig Attributes
DESCRIPTION: Details the attributes for ThinkingConfig, used to configure thinking processes. It primarily includes a setting to control whether thoughts should be included in the output.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import ThinkingConfig

# Accessing attributes
include_thoughts: bool = ThinkingConfig.include_thoughts
```

----------------------------------------

TITLE: TranscriptionDict Type
DESCRIPTION: A dictionary representation of transcription results, mirroring the Transcription type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TranscriptionDict:
  finished: Boolean indicating if the transcription is complete.
  text: The transcribed text content.
```

----------------------------------------

TITLE: MaskReferenceImage
DESCRIPTION: Represents an image used as a reference for masking.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MaskReferenceImage:
  # Attributes would typically be defined here, e.g., image data or URI.
```

----------------------------------------

TITLE: HarmProbability API
DESCRIPTION: Defines probability levels for harmful content.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HarmProbability:
  HARM_PROBABILITY_UNSPECIFIED: Unspecified harm probability.
  HIGH: High probability of harm.
  LOW: Low probability of harm.
  MEDIUM: Medium probability of harm.
```

----------------------------------------

TITLE: Language Enum
DESCRIPTION: Specifies supported programming languages for the GenAI service. This enum is used to indicate the language context for operations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Language:
  LANGUAGE_UNSPECIFIED: Language is not specified.
  PYTHON: Python programming language.
```

----------------------------------------

TITLE: LiveServerSessionResumptionUpdate Types
DESCRIPTION: Details for session resumption, including the last consumed message index, new handle, and resumability status.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
LiveServerSessionResumptionUpdate:
  last_consumed_client_message_index: int | None
    The index of the last client message that was successfully consumed.
  new_handle: str | None
    A new handle for the session if it was resumed.
  resumable: bool | None
    Indicates whether the session is resumable.

LiveServerSessionResumptionUpdateDict:
  last_consumed_client_message_index: int | None
    The index of the last client message that was successfully consumed.
  new_handle: str | None
    A new handle for the session if it was resumed.
  resumable: bool | None
    Indicates whether the session is resumable.
```

----------------------------------------

TITLE: MaskReferenceImageDict API
DESCRIPTION: Represents MaskReferenceImage as a dictionary, providing access to its configuration, reference ID, reference image, and reference type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
MaskReferenceImageDict:
  config: Configuration for the mask.
  reference_id: Unique identifier for the reference image.
  reference_image: The reference image data.
  reference_type: The type of reference image provided.
```

----------------------------------------

TITLE: InlinedRequestDict Type
DESCRIPTION: Defines the structure for an inlined request in dictionary format.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
InlinedRequestDict:
  config: Configuration for the model.
  contents: The content of the request.
  model: The model to use for the request.
```

----------------------------------------

TITLE: Mode Enum Values
DESCRIPTION: Defines the available modes for operations within the GenAI library.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Mode:
  MODE_DYNAMIC: Dynamic mode.
  MODE_UNSPECIFIED: Unspecified mode.
```

----------------------------------------

TITLE: TestTableItemDict Attributes
DESCRIPTION: Describes the attributes for TestTableItemDict, a dictionary format for test items. It mirrors the TestTableItem attributes, covering environment-specific exceptions, union handling, key ignoring, naming, replay ID overrides, parameters, and API mode skipping.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import TestTableItemDict

# Accessing attributes
exception_if_mldev: bool = TestTableItemDict.exception_if_mldev
exception_if_vertex: bool = TestTableItemDict.exception_if_vertex
has_union: bool = TestTableItemDict.has_union
ignore_keys: list[str] = TestTableItemDict.ignore_keys
name: str = TestTableItemDict.name
override_replay_id: str = TestTableItemDict.override_replay_id
parameters: dict = TestTableItemDict.parameters
skip_in_api_mode: bool = TestTableItemDict.skip_in_api_mode
```

----------------------------------------

TITLE: TestTableFile Attributes
DESCRIPTION: Defines the attributes for the TestTableFile type, used for specifying test cases. It includes fields for comments, parameter names, test methods, and the test table data itself.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import TestTableFile

# Accessing attributes
comment: str = TestTableFile.comment
parameter_names: list[str] = TestTableFile.parameter_names
test_method: str = TestTableFile.test_method
test_table: list[dict] = TestTableFile.test_table
```

----------------------------------------

TITLE: Update Tuned Model Configuration
DESCRIPTION: Updates the configuration of a tuned model, such as its display name and description. This allows for modification of model metadata after creation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

tuned_model = client.models.update(
    model=tuning_job.tuned_model.model,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)
print(tuned_model)
```

----------------------------------------

TITLE: FileDataDict Attributes
DESCRIPTION: Dictionary representation for FileData, used for providing file information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileDataDict:
  display_name: str
    The display name of the file.
  file_uri: str
    The URI of the file.
  mime_type: str
    The MIME type of the file.
```

----------------------------------------

TITLE: CitationMetadata Attributes
DESCRIPTION: Describes the CitationMetadata type and its associated citations attribute.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CitationMetadata:
  citations: list[CitationDict]
    A list of citation dictionaries.
```

----------------------------------------

TITLE: TunedModelDict Type
DESCRIPTION: A dictionary representation of a tuned model, mirroring the TunedModel type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TunedModelDict:
  checkpoints: List of checkpoints for the tuned model.
  endpoint: The endpoint associated with the tuned model.
  model: The base model used for tuning.
```

----------------------------------------

TITLE: Generate Enum Response Schema (JSON)
DESCRIPTION: Shows how to generate content with an enum response schema, returned as a JSON string. The output will be a JSON string representing one of the enum values.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'application/json',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)
```

----------------------------------------

TITLE: ExternalApiSimpleSearchParams
DESCRIPTION: Represents parameters for simple search operations within the External API.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ExternalApiSimpleSearchParams:
  (No specific parameters documented in this snippet)
```

----------------------------------------

TITLE: GoogleSearchDict Attributes
DESCRIPTION: Details the attributes available within the GoogleSearchDict type, including time_range_filter for specifying search time constraints.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
GoogleSearchDict:
  time_range_filter: Specifies the time range for search results.
```

----------------------------------------

TITLE: Count Tokens (Python)
DESCRIPTION: Calculates the number of tokens for a given text input using a specified model. This is useful for understanding input size limitations and costs.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: TunedModelCheckpoint Type
DESCRIPTION: Details of a specific checkpoint for a tuned model, including ID, endpoint, epoch, and step.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TunedModelCheckpoint:
  checkpoint_id: The unique identifier for the checkpoint.
  endpoint: The endpoint associated with this checkpoint.
  epoch: The epoch number for this checkpoint.
  step: The training step number for this checkpoint.
```

----------------------------------------

TITLE: InlinedResponse Type
DESCRIPTION: Defines the structure for an inlined response, containing either an error or the actual response.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
InlinedResponse:
  error: Error details if the request failed.
  response: The actual response from the model.
```

----------------------------------------

TITLE: ReplayResponseDict Attributes
DESCRIPTION: Details the attributes of the ReplayResponseDict type, which likely represents a response from a replay mechanism. Includes body segments, headers, SDK response segments, and status code.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ReplayResponseDict:
  body_segments: Segments of the response body.
  headers: Headers of the replay response.
  sdk_response_segments: Segments of the SDK response.
  status_code: The HTTP status code of the replay response.
```

----------------------------------------

TITLE: JobError Type Documentation
DESCRIPTION: Documentation for the JobError type, which represents an error that occurred during a job execution.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
JobError:
  Represents an error in a job.
  Attributes:
    code: The error code.
    details: Additional details about the error.
    message: A human-readable error message.
```

----------------------------------------

TITLE: InlinedResponseDict Type
DESCRIPTION: Defines the structure for an inlined response in dictionary format.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
InlinedResponseDict:
  error: Error details if the request failed.
  response: The actual response from the model.
```

----------------------------------------

TITLE: Edit Image
DESCRIPTION: Edits existing images based on textual instructions. This can include tasks like changing elements, adding effects, or modifying the style of an image.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('image-editing@001')

# Assuming 'original_image' is a file-like object or bytes
# with open('path/to/your/image.png', 'rb') as f:
#     original_image = f.read()

# prompt = "Add a hat to the person in the image"
# response = model.edit_content(original_image, prompt)

# print(response.images[0].url)
```

----------------------------------------

TITLE: Endpoint Type Attributes
DESCRIPTION: Represents an endpoint with its deployed model ID and name. Also shows the dictionary representation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import Endpoint, EndpointDict

# Example usage of Endpoint
endpoint_data = Endpoint(deployed_model_id='model-123', name='my-endpoint')
print(f"Endpoint Name: {endpoint_data.name}")
print(f"Deployed Model ID: {endpoint_data.deployed_model_id}")

# Example usage of EndpointDict
endpoint_dict_data = EndpointDict({'deployed_model_id': 'model-456', 'name': 'another-endpoint'})
print(f"Endpoint Dict Name: {endpoint_dict_data['name']}")
print(f"Endpoint Dict Deployed Model ID: {endpoint_dict_data['deployed_model_id']}")
```

----------------------------------------

TITLE: UrlRetrievalStatus Enumeration
DESCRIPTION: Enumerates the possible statuses for URL retrieval operations. These statuses indicate the outcome of attempting to access or process a given URL.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlRetrievalStatus:
  URL_RETRIEVAL_STATUS_UNSPECIFIED: 0
    Default value. This status is not specified.
  URL_RETRIEVAL_STATUS_SUCCESS: 1
    The URL was successfully retrieved.
  URL_RETRIEVAL_STATUS_ERROR: 2
    An error occurred during URL retrieval.
  URL_RETRIEVAL_STATUS_UNSAFE: 3
    The URL was deemed unsafe to retrieve.
  URL_RETRIEVAL_STATUS_PAYWALL: 4
    The URL is behind a paywall and could not be accessed.
```

----------------------------------------

TITLE: ImageDict Attributes
DESCRIPTION: Details the attributes of the ImageDict type, used for representing image data in a dictionary format.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ImageDict:
  gcs_uri: Google Cloud Storage URI of the image.
  image_bytes: The raw bytes of the image.
  mime_type: The MIME type of the image.
```

----------------------------------------

TITLE: Update Tuned Model (using Pager)
DESCRIPTION: Updates a tuned model's configuration, specifically its display name and description, using a model object obtained from a list operation.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    )
)

print(model)
```

----------------------------------------

TITLE: SegmentDict Properties
DESCRIPTION: Details the properties available for the SegmentDict type in the Python GenAI library. These properties are dictionary representations of text segments.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SegmentDict:
  end_index: The ending index of the segment.
  part_index: The index of the part within a larger sequence.
  start_index: The starting index of the segment.
  text: The text content of the segment.
```

----------------------------------------

TITLE: EndSensitivity Enum
DESCRIPTION: Defines sensitivity levels for the end of content generation.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EndSensitivity:
  END_SENSITIVITY_UNSPECIFIED: Default, unspecified sensitivity.
  END_SENSITIVITY_HIGH: High sensitivity for end detection.
  END_SENSITIVITY_MEDIUM: Medium sensitivity for end detection.
  END_SENSITIVITY_LOW: Low sensitivity for end detection.
```

----------------------------------------

TITLE: ContentEmbeddingStatistics Attributes
DESCRIPTION: Details the ContentEmbeddingStatistics type and its token_count attribute.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ContentEmbeddingStatistics:
  token_count: int
    The number of tokens in the content.
```

----------------------------------------

TITLE: Update Tuned Model
DESCRIPTION: Updates an existing tuned model with a new display name and description. This operation requires a model object obtained previously.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model',
        description='my tuned model description'
    ),
)

print(model)
```

----------------------------------------

TITLE: Safety Filter Levels
DESCRIPTION: Defines the levels for safety filtering, specifying how content is blocked based on severity.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
SafetyFilterLevel.BLOCK_ONLY_HIGH
```

----------------------------------------

TITLE: EditMode Enum Values
DESCRIPTION: Lists the available enumeration values for EditMode, specifying different modes for image editing operations.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EditMode:
  EDIT_MODE_BGSWAP: "EDIT_MODE_BGSWAP"
    Background swap editing mode.
  EDIT_MODE_CONTROLLED_EDITING: "EDIT_MODE_CONTROLLED_EDITING"
    Controlled editing mode.
  EDIT_MODE_DEFAULT: "EDIT_MODE_DEFAULT"
    Default editing mode.
  EDIT_MODE_INPAINT_INSERTION: "EDIT_MODE_INPAINT_INSERTION"
    Inpainting and insertion editing mode.
```

----------------------------------------

TITLE: SafetyAttributes and Filtering Levels
DESCRIPTION: Details the SafetyAttributes and SafetyAttributesDict types, used for managing safety configurations and scores. Includes categories, content type, and scores. Also documents SafetyFilterLevel enum for controlling safety filters.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
SafetyAttributes:
  categories: List of safety categories.
  content_type: The type of content being evaluated for safety.
  scores: Safety scores for the content.

SafetyAttributesDict:
  categories: List of safety categories.
  content_type: The type of content being evaluated for safety.
  scores: Safety scores for the content.

SafetyFilterLevel:
  BLOCK_LOW_AND_ABOVE: Blocks content with low safety scores and above.
  BLOCK_MEDIUM_AND_ABOVE: Blocks content with medium safety scores and above.
  BLOCK_NONE: No safety filtering is applied.
```

----------------------------------------

TITLE: Blob Attributes
DESCRIPTION: Attributes for the Blob data structure, representing binary data. Includes the data itself, a display name, and the MIME type.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Blob:
  data: The binary data content.
  display_name: A user-friendly name for the blob.
  mime_type: The MIME type of the data (e.g., 'image/jpeg').
```

----------------------------------------

TITLE: Safety Rating Dictionary Properties
DESCRIPTION: Details the properties of a SafetyRatingDict object, mirroring SafetyRating with dictionary access.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
SafetyRatingDict.blocked
SafetyRatingDict.category
SafetyRatingDict.overwritten_threshold
SafetyRatingDict.probability
SafetyRatingDict.probability_score
SafetyRatingDict.severity
SafetyRatingDict.severity_score
```

----------------------------------------

TITLE: ModalityTokenCount API
DESCRIPTION: Represents the token count for a specific media modality.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
ModalityTokenCount:
  modality: The media modality.
  token_count: The number of tokens for the modality.
```

----------------------------------------

TITLE: Embed Content
DESCRIPTION: Generates embeddings for given text content. Embeddings are numerical representations of text that capture semantic meaning, useful for tasks like similarity search and clustering.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
import google.generativeai as genai

model = genai.GenerativeModel('embedding-001')

texts = ["Hello world", "This is a test"]
response = model.embed_content(content=texts)

# The response contains a list of embeddings, one for each input text
# print(response['embedding'])

```

----------------------------------------

TITLE: UrlMetadataDict Structure
DESCRIPTION: Defines the structure for URL metadata, including the retrieved URL and its retrieval status. This is a data structure used to hold information about a URL that has been processed.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
UrlMetadataDict:
  retrieved_url: str
    The URL that was retrieved.
  url_retrieval_status: UrlRetrievalStatus
    The status of the URL retrieval process.
```

----------------------------------------

TITLE: GenAI Type Enum
DESCRIPTION: Enumeration for different data types used within the GenAI library. Includes NUMBER, OBJECT, STRING, and TYPE_UNSPECIFIED.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Type:
  An enumeration for data types.
  Members:
    NUMBER: Represents a numeric data type.
    OBJECT: Represents an object data type.
    STRING: Represents a string data type.
    TYPE_UNSPECIFIED: Represents an unspecified data type.
```

----------------------------------------

TITLE: FileData Object Attributes
DESCRIPTION: Attributes of the FileData object, used for providing file information.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
FileData:
  display_name: str
    The display name of the file.
  file_uri: str
    The URI of the file.
  mime_type: str
    The MIME type of the file.
```

----------------------------------------

TITLE: TrafficType Enum
DESCRIPTION: Defines the types of traffic for model interactions. Includes PROVISIONED_THROUGHPUT and TRAFFIC_TYPE_UNSPECIFIED.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TrafficType:
  PROVISIONED_THROUGHPUT: Represents provisioned throughput.
  TRAFFIC_TYPE_UNSPECIFIED: Represents an unspecified traffic type.
```

----------------------------------------

TITLE: Behavior Enum Values
DESCRIPTION: Enumeration for behavior types, specifying operational modes such as blocking, non-blocking, or unspecified.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Behavior:
  BLOCKING: Operation will block until completion.
  NON_BLOCKING: Operation will not block.
  UNSPECIFIED: Behavior is not specified.
```

----------------------------------------

TITLE: EmbedContentMetadata and EmbedContentMetadataDict
DESCRIPTION: Metadata associated with content embedding operations, primarily indicating the billable character count.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EmbedContentMetadata:
  billable_character_count: The number of characters that were billed for the embedding operation.

EmbedContentMetadataDict:
  billable_character_count: The number of characters that were billed for the embedding operation.
```

----------------------------------------

TITLE: HarmSeverity Enum
DESCRIPTION: Defines the severity levels of harm. Includes UNSPECIFIED, LOW, MEDIUM, HIGH, and NEGLIGIBLE.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HarmSeverity:
  HARM_SEVERITY_UNSPECIFIED: Harm severity is not specified.
  HARM_SEVERITY_LOW: Low severity of harm.
  HARM_SEVERITY_MEDIUM: Medium severity of harm.
  HARM_SEVERITY_HIGH: High severity of harm.
  HARM_SEVERITY_NEGLIGIBLE: Negligible severity of harm.
```

----------------------------------------

TITLE: Transcription Type
DESCRIPTION: Represents transcription results, including whether the transcription is finished and the transcribed text.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Transcription:
  finished: Boolean indicating if the transcription is complete.
  text: The transcribed text content.
```

----------------------------------------

TITLE: Compute Tokens (Vertex AI) (Python)
DESCRIPTION: Computes tokens for a given text input, specifically supported in Vertex AI. This function is used for detailed token analysis.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Traffic Type Enum
DESCRIPTION: Defines the type of traffic, with ON_DEMAND as a possible value.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
TrafficType:
  ON_DEMAND: str
    Represents on-demand traffic.
```

----------------------------------------

TITLE: CitationDict Type Attributes
DESCRIPTION: Details the attributes for the CitationDict type, a dictionary representation of a Citation. It provides access to citation details like indices, license, publication date, title, and URI.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
CitationDict:
  end_index: The end index of the cited text within the content.
  license: The license associated with the citation.
  publication_date: The publication date of the cited source.
  start_index: The start index of the cited text within the content.
  title: The title of the cited source.
  uri: The Uniform Resource Identifier (URI) of the cited source.
```

----------------------------------------

TITLE: Update Tuned Model Configuration
DESCRIPTION: Updates the display name and description of a tuned model. This operation modifies the metadata associated with the tuned model.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
from google.genai import types

tuned_model = client.models.update(
    model=tuning_job.tuned_model.model,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    )
)
print(tuned_model)
```

----------------------------------------

TITLE: Set Automatic Function Call Turns in ANY Mode
DESCRIPTION: Demonstrates how to limit the number of automatic function call turns when the function calling mode is set to 'ANY'. By configuring `maximum_remote_calls` in `types.AutomaticFunctionCallingConfig`, you can control how many times the SDK will automatically invoke functions.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: EndSensitivity Enum Values
DESCRIPTION: Defines the sensitivity levels for the end of a response. Includes HIGH, LOW, and UNSPECIFIED values.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from genai.types import EndSensitivity

# Accessing enum values
print(EndSensitivity.END_SENSITIVITY_HIGH)
print(EndSensitivity.END_SENSITIVITY_LOW)
print(EndSensitivity.END_SENSITIVITY_UNSPECIFIED)
```

----------------------------------------

TITLE: HarmProbability Enum
DESCRIPTION: Defines the probability of harm. Includes NEGLIGIBLE and other unspecified levels.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
HarmProbability:
  NEGLIGIBLE: The probability of harm is negligible.
```

----------------------------------------

TITLE: Disabling Automatic Function Calling
DESCRIPTION: Explains how to disable the automatic function calling feature when providing Python functions as tools. This results in the model returning function call parts instead of executing them.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)
```

----------------------------------------

TITLE: Count Tokens (Asynchronous) (Python)
DESCRIPTION: Asynchronously counts the tokens for a given text input using a specified model. This is the async version of the count_tokens method.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)
```

----------------------------------------

TITLE: Type Enum
DESCRIPTION: Represents fundamental data types used within the API, such as ARRAY, BOOLEAN, INTEGER, and NULL.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
Type:
  ARRAY: Represents an array data type.
  BOOLEAN: Represents a boolean data type.
  INTEGER: Represents an integer data type.
  NULL: Represents a null data type.
```

----------------------------------------

TITLE: Safety Rating Properties
DESCRIPTION: Details the properties of a SafetyRating object, including blocked status, category, probability, and severity.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
SafetyRating.blocked
SafetyRating.category
SafetyRating.overwritten_threshold
SafetyRating.probability
SafetyRating.probability_score
SafetyRating.severity
SafetyRating.severity_score
```

----------------------------------------

TITLE: Monitor Batch Job Completion
DESCRIPTION: Polls the status of a batch job until it reaches a completed state (succeeded, failed, cancelled, or paused). Includes a delay between checks to avoid excessive API calls.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
completed_states = set(
    [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_PAUSED',
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)

job
```

----------------------------------------

TITLE: Disable Automatic Function Calling in ANY Mode
DESCRIPTION: Shows how to configure function calling mode to 'ANY' and disable automatic function calling by setting `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`. This ensures that the SDK does not automatically invoke functions when the mode is set to 'ANY'.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)
```

----------------------------------------

TITLE: EditMode Enum Values
DESCRIPTION: Defines the different modes for image editing operations. These modes specify the type of manipulation to be performed on an image.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: APIDOC
CODE:
```
EditMode:
  EDIT_MODE_INPAINT_REMOVAL: Inpainting mode for removing objects.
  EDIT_MODE_OUTPAINT: Outpainting mode for extending image boundaries.
  EDIT_MODE_PRODUCT_IMAGE: Mode for generating product images.
  EDIT_MODE_STYLE: Mode for applying artistic styles.
```

----------------------------------------

TITLE: Delete Batch Job
DESCRIPTION: Deletes a specified batch job resource. This action is irreversible and removes the job and its associated data.

SOURCE: https://googleapis.github.io/python-genai/_sources/index.rst

LANGUAGE: python
CODE:
```
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job
```

----------------------------------------

TITLE: Delete Batch Prediction Job
DESCRIPTION: Deletes a specified batch prediction job resource using its unique name. This operation permanently removes the job.

SOURCE: https://googleapis.github.io/python-genai/index

LANGUAGE: python
CODE:
```
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job
```