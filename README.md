# Constitutional AI QA Survey App

Gradio app for testing a parallel constitutional AI chatbot and completing the QA survey from `survey.md`.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The app clones the current constitutional AI kit at runtime into `.deps/constitutional-ai-kit`:

```text
https://github.com/arazilab/constitutional-ai-kit.git
```

That local clone is ignored by git.

Create a local config file:

```bash
cp config.example.json config.json
```

Then edit `config.json`. It can set credentials, writer and judge models, runtime settings, and prompt templates:

```json
{
  "settings": {
    "credentials": {
      "openai_api_key": "sk-..."
    },
    "writer": {
      "provider": "openai",
      "model": "gpt-4.1-nano"
    },
    "judge": {
      "provider": "openai",
      "model": "gpt-4.1-nano"
    }
  }
}
```

The app merges `config.json` over built-in defaults. If the key is not there, it falls back to `OPENAI_API_KEY` from `.env` or the shell. Older config files with top-level `"openai_api_key"` still work.

## Run

```bash
./.venv/bin/python app.py
```

The app automatically uses an available local port and prints the URL:

```text
http://127.0.0.1:<port>
```

To pin a specific port:

```bash
GRADIO_SERVER_PORT=9000 ./.venv/bin/python app.py
```

## Behavior

- Left two-thirds: chatbot using `gpt-4.1-nano` as writer and judge.
- Right one-third: scrollable HTML survey generated from `survey.md`.
- Constitution rules come from `constitution.txt`.
- Constitutional mode is parallel with one revision iteration.
- Chat clearing does not reset the survey form.
- Form clearing does not reset the chat.
- Saving the form directly downloads a JSON file with every survey question and field, including unanswered fields.
