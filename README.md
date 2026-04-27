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
  "app": {
    "server_name": "127.0.0.1",
    "server_port": null,
    "share": false
  },
  "ngrok": {
    "auth_token": ""
  },
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

To create a temporary public Gradio share link, set `"share": true` under `"app"` in your local `config.json` and run the app. Anyone with that link can use the app while your local server is running, so leave it `false` unless you intentionally want to share it.

You can also override sharing from the shell:

```bash
GRADIO_SHARE=1 ./.venv/bin/python app.py
```

## Share with ngrok

For a local public link without using Gradio's share service, run:

```bash
./share_with_ngrok.sh
```

The script creates `.venv` if needed, installs `requirements.txt`, downloads ngrok into `.deps/` if ngrok is not already installed, starts the app on port `7860`, opens an ngrok tunnel, and prints the public `https://...ngrok...` URL. It forces Gradio's built-in share mode off while ngrok is running. Keep the terminal open while people use the app.

If your ngrok account requires an auth token, put it in ignored `config.json`:

```json
{
  "ngrok": {
    "auth_token": "your-ngrok-token"
  }
}
```

You can also set it from the shell; `NGROK_AUTHTOKEN` takes precedence over `config.json`:

```bash
export NGROK_AUTHTOKEN="your-ngrok-token"
./share_with_ngrok.sh
```

To use a different local port:

```bash
GRADIO_SERVER_PORT=9000 ./share_with_ngrok.sh
```

Anyone with the ngrok URL can use the app while it is running, so they can indirectly use the configured API key through the chatbot. The key itself should stay in ignored `config.json`, `.env`, or an environment variable.

## Behavior

- Left two-thirds: chatbot using `gpt-4.1-nano` as writer and judge.
- Right one-third: scrollable HTML survey generated from `survey.md`.
- Constitution rules come from `constitution.txt`.
- Constitutional mode is parallel with one revision iteration.
- Chat clearing does not reset the survey form.
- Form clearing does not reset the chat.
- Saving the form directly downloads a JSON file with every survey question and field, including unanswered fields.
