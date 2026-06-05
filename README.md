# Subtitle Translator — English → Bulgarian 🇧🇬

Translate `.srt` and `.ass` subtitles from English to natural Bulgarian (Cyrillic) using a local [Ollama](https://ollama.com) AI model. Preserves timestamps, formatting tags, and character names — only the dialogue text gets translated.

## Features

- **SRT & ASS support** — metadata, styling tags (`{\pos...}`, `<i>`, etc.), and timestamps left untouched
- **Context‑aware translation** — each block is translated with 3 previous + 2 next blocks as context (sliding window) for natural idioms and narrative flow
- **Checkpoint resume** — if the process is interrupted, `--resume` picks up where you left off
- **Quick validation** — `--test 10` translates only the first 10 blocks to verify quality before a full run
- **Auto‑setup** — detects missing Ollama and installs it on Linux/macOS; pulls the model if not present
- **Retry logic** — automatic retries on connection timeouts
- **Zero external Python dependencies** — uses only the standard library

## Prerequisites

- **Python 3.10+**
- **Ollama** — installed and running (the script can auto‑install it on Linux/macOS)
- **A Gemma4 model** — pulled locally (the script auto‑pulls `gemma4:e2b` by default)

## Quick Start

```bash
# Translate a single file
python subs_translator.py episode.srt

# Translate all .srt/.ass files in a directory
python subs_translator.py ./subtitles/

# Quick test — only 5 blocks
python subs_translator.py --test 5 episode.srt

# Use a different model (e.g. the larger 8B variant)
python subs_translator.py --model gemma4:e4b episode.srt

# Resume after an interruption
python subs_translator.py --resume episode.srt
```

Output is saved as `{filename}_bg.{ext}` (UTF‑8 encoded) in the same directory as the input.

## CLI Reference

```
usage: subs_translator.py [-h] [-m MODEL] [-p PORT] [-o OUTPUT_DIR]
                          [--test N] [--resume]
                          input [input ...]

positional arguments:
  input                 SRT/ASS file(s) or directory

options:
  -m, --model MODEL     Ollama model name (default: gemma4:e2b)
  -p, --port PORT       Ollama API port (default: 11434)
  -o, --output-dir DIR  Output directory (default: same as input)
  --test N              Process only first N blocks (quick validation)
  --resume              Resume from saved checkpoint
```

## How It Works

1. **Environment check** — verifies Ollama is installed and running
2. **Model check** — pulls the requested model if absent
3. **File parsing** — reads `.srt` or `.ass`, separating dialogue from metadata
4. **Sliding‑window translation** — for each block, the LLM receives:
   - 3 previous blocks (narrative context)
   - the current block to translate
   - 2 next blocks (tone continuity)
5. **Output** — saves the translated file as `{name}_bg.{ext}`

Translation engine settings (optimised for GPU):
- `temperature`: 0.3 (consistent, low‑creativity)
- `num_predict`: 512 (sufficient for multi‑line subtitles)

## Checkpoint & Resume

The script writes a hidden checkpoint file next to the output:

```
{output_filename}.checkpoint.json
```

If the process crashes or you hit Ctrl+C, just re‑run with `--resume`:

```bash
python subs_translator.py episode.srt   # runs, crashes at 40%
python subs_translator.py --resume episode.srt  # continues from 40%
```

The checkpoint is automatically deleted when translation reaches 100%.

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` | Start Ollama: `ollama serve` |
| `Model not found` | Pull it: `ollama pull gemma4:e2b`, or use `--model` |
| Slow translation | First inference is cold — allow 1‑2 min. GPU recommended. |
| Bad encoding | All output is UTF‑8. Ensure your player supports Cyrillic. |
| `ollama: command not found` | Linux/macOS: script auto‑installs. Windows: download from [ollama.com](https://ollama.com) |

## File Format Support

**SRT (SubRip):**
- Preserves index numbers, timestamps, blank‑line separation
- Preserves `<i>`, `<b>`, `<font>` tags

**ASS (Advanced SubStation Alpha):**
- Preserves all sections ([Script Info], [V4+ Styles], [Events])
- Translates only `Dialogue:` lines; skips `Comment:` lines
- Preserves style overrides: `{\pos...}`, `{\i1}`, etc.
- Skips vector drawings (`{\p1}`…`{\p0}`) automatically

## License

GNU General Public License v3.0
