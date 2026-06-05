#!/usr/bin/env python3
"""
Subtitle Translator — English → Bulgarian (Cyrillic)
Uses local Ollama (gemma4) with sliding-window context for
idiom-aware, narrative-flow-preserving translation.

Features:
  • SRT / ASS format parsing (metadata & tags untouched)
  • Sliding-window context (3 past + 2 future blocks)
  • Checkpoint-based resume on interrupt
  • --test N for quick validation runs
  • Retry logic with exponential back-off
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ─── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemma4:e2b"
OLLAMA_BASE = "http://localhost"
DEFAULT_PORT = 11434
MAX_RETRIES = 3
RETRY_DELAY = 2
WINDOW_PAST = 3
WINDOW_FUTURE = 2
TIMEOUT = 600
CHECKPOINT_SUFFIX = ".checkpoint.json"

# ─── Progress Bar ──────────────────────────────────────────────────────────────

class ProgressBar:
    """Zero-dependency terminal progress bar."""

    def __init__(self, total: int, prefix: str = "", length: int = 40):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0

    def update(self, n: int = 1) -> None:
        self.current += n
        pct = self.current / self.total if self.total else 0
        filled = int(self.length * pct)
        bar = "█" * filled + "░" * (self.length - filled)
        sys.stdout.write(f"\r{self.prefix} |{bar}| {self.current}/{self.total}")
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")

    def close(self) -> None:
        if self.current < self.total:
            self.update(self.total - self.current)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def detect_encoding(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                f.read()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "latin-1"



# ─── Ollama Environment ────────────────────────────────────────────────────────

def _run_cmd(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", "command not found"
    except subprocess.TimeoutExpired:
        return -2, "", "timeout"


def is_ollama_installed() -> bool:
    return _run_cmd(["ollama", "--version"])[0] == 0


def is_ollama_running(port: int = DEFAULT_PORT) -> bool:
    url = f"{OLLAMA_BASE}:{port}/api/tags"
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except (urllib.error.URLError, ConnectionRefusedError, OSError):
        return False


def install_ollama() -> None:
    import platform
    system = platform.system().lower()
    print("⏳ Ollama not found — installing …")
    if system in ("linux", "darwin"):
        rc, _, err = _run_cmd(
            ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            timeout=120,
        )
        if rc != 0:
            print(f"Installation failed:\n{err}")
            sys.exit(1)
    else:
        print("Windows: install from https://ollama.com and re-run.")
        sys.exit(1)
    print("✔ Ollama installed.")


def ensure_ollama(port: int = DEFAULT_PORT) -> None:
    if not is_ollama_installed():
        install_ollama()
    if not is_ollama_running(port):
        print("⏳ Starting Ollama server …")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            if is_ollama_running(port):
                break
            time.sleep(1)
        else:
            print("✘ Could not start Ollama. Start it manually and re-run.")
            sys.exit(1)
    print("✔ Ollama is running.")


def ensure_model(model: str) -> None:
    rc, out, _ = _run_cmd(["ollama", "list"])
    if rc != 0:
        print("✘ Could not list models.")
        sys.exit(1)
    lines = [line.split()[0] for line in out.strip().split("\n")[1:] if line.strip()]
    if model not in lines:
        print(f"⏳ Model '{model}' not found — pulling …")
        rc2, _, err2 = _run_cmd(["ollama", "pull", model], timeout=600)
        if rc2 != 0:
            print(f"✘ Failed to pull '{model}':\n{err2}")
            sys.exit(1)
        print(f"✔ Model '{model}' pulled.")
    else:
        print(f"✔ Model '{model}' available.")


# ─── SRT Parser ────────────────────────────────────────────────────────────────

def parse_srt(content: str) -> list[dict]:
    blocks: list[dict] = []
    for raw_block in content.strip().split("\n\n"):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        lines = raw_block.split("\n")
        if len(lines) < 2:
            continue
        if not lines[0].strip().isdigit():
            continue
        if "-->" not in lines[1]:
            continue
        text = [l for l in lines[2:] if l.strip()]
        blocks.append({
            "index": lines[0].strip(),
            "timestamp": lines[1].strip(),
            "text": text,
        })
    return blocks


def build_srt(blocks: list[dict]) -> str:
    out: list[str] = []
    for b in blocks:
        out.append(b["index"])
        out.append(b["timestamp"])
        out.extend(b["text"])
        out.append("")
    return "\n".join(out)


# ─── ASS Parser ────────────────────────────────────────────────────────────────

def parse_ass(content: str) -> tuple[list[str], list[dict]]:
    header: list[str] = []
    events: list[dict] = []
    in_events = False
    fmt: list[str] = []

    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("["):
            in_events = s.upper() == "[EVENTS]"
            header.append(line)
            continue
        if in_events:
            if s.upper().startswith("FORMAT:"):
                fmt = [f.strip() for f in s[len("Format:"):].split(",")]
                header.append(line)
                continue
            if s.upper().startswith("DIALOGUE:") or s.upper().startswith("COMMENT:"):
                events.append({
                    "raw": line,
                    "type": "Dialogue" if s.upper().startswith("DIALOGUE:") else "Comment",
                    "fields": fmt,
                })
                continue
        header.append(line)

    return header, events


def ass_text_field(ass_line: str, fields: list[str]) -> tuple[str, int]:
    prefix = "Dialogue: "
    if not ass_line.startswith("Dialogue: "):
        return ass_line, -1
    n = len(fields)
    parts = ass_line[len(prefix):].split(",", maxsplit=n - 1) if n > 1 else [ass_line[len(prefix):]]
    if len(parts) < n:
        return ass_line, -1
    return parts[-1], n - 1


def ass_set_text(original: str, new_text: str, fields: list[str]) -> str:
    prefix = "Dialogue: "
    body = original[len(prefix):]
    n = len(fields)
    parts = body.split(",", maxsplit=n - 1) if n > 1 else [body]
    if len(parts) < n:
        return original
    parts[-1] = new_text
    return prefix + ",".join(parts)


def ass_has_drawing(text: str) -> bool:
    return bool(re.search(r'\\p\d', text))


# ─── Checkpoint System ─────────────────────────────────────────────────────────

def checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + CHECKPOINT_SUFFIX)


def save_checkpoint(cp_file: Path, model: str, completed: set[int]) -> None:
    data = {"model": model, "completed": sorted(completed)}
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_checkpoint(cp_file: Path, model: str) -> set[int] | None:
    if not cp_file.exists():
        return None
    try:
        data = json.loads(cp_file.read_text("utf-8"))
        if data.get("model") != model:
            print(f"⚠ Checkpoint model mismatch ({data.get('model')} vs {model}), ignoring.")
            return None
        return set(data.get("completed", []))
    except (json.JSONDecodeError, KeyError):
        print("⚠ Corrupt checkpoint, starting fresh.")
        return None


def clear_checkpoint(cp_file: Path) -> None:
    if cp_file.exists():
        cp_file.unlink()


# ─── Translation Engine ────────────────────────────────────────────────────────

def ollama_api_url(port: int = DEFAULT_PORT, endpoint: str = "chat") -> str:
    return f"{OLLAMA_BASE}:{port}/api/{endpoint}"


def call_ollama(
    system: str,
    user: str,
    model: str,
    port: int,
) -> str | None:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "num_predict": 512,
            "temperature": 0.3,
        },
    }).encode("utf-8")

    url = ollama_api_url(port, "chat")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            return msg.get("content", "")
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionResetError, TimeoutError, OSError) as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"\n⚠ Conn issue (attempt {attempt}/{MAX_RETRIES}): {exc}")
                print(f"   Retry in {wait}s …")
                time.sleep(wait)
            else:
                print(f"\n✘ Failed after {MAX_RETRIES} attempts: {exc}")
                return None
        except json.JSONDecodeError as exc:
            print(f"\n✘ Invalid JSON: {exc}")
            return None


SYSTEM_PROMPT = (
    "You are an expert EN→BG subtitle translator. "
    "Your task is to translate English dialogue into natural Bulgarian (Cyrillic).\n\n"
    "RULES:\n"
    "• Translate ONLY the SOURCE TEXT — nothing more, nothing less.\n"
    "• Preserve ALL formatting tags (e.g. <i>, </i>, {\\pos...}) exactly as they appear.\n"
    "• Do NOT add any formatting tags that were not present in the SOURCE TEXT.\n"
    "• Keep character-name prefixes (e.g. \"JOHN:\", \"JESSICA:\") unchanged.\n"
    "• Use natural, colloquial Bulgarian — adapt idioms and slang appropriately.\n"
    "• Preserve the EXACT number of lines as the input.\n"
    "• Do NOT think, reason, explain, or add commentary of any kind.\n"
    "• Start your response IMMEDIATELY with the translation — no preamble."
)


def build_user_message(
    current: str,
    past_lines: list[list[str]],
    future_lines: list[list[str]],
) -> str:
    past = "\n".join("\n".join(b) for b in past_lines) if past_lines else "(none — start of file)"
    future = "\n".join("\n".join(b) for b in future_lines) if future_lines else "(none — end of file)"

    return (
        "HISTORICAL CONTEXT (previous scenes — for reference, DO NOT translate):\n"
        f"{past}\n\n"
        "SOURCE TEXT (translate THIS ONLY):\n"
        f"{current}\n\n"
        "UPCOMING CONTEXT (next scenes — for reference, DO NOT translate):\n"
        f"{future}"
    )


def strip_hallucinated_tags(text: str, original_text: str) -> str:
    """Remove HTML-like tags that weren't present in the original."""
    tags = re.findall(r'</?[a-zA-Z]+[^>]*>', text)
    orig_tags = set(re.findall(r'</?[a-zA-Z]+[^>]*>', original_text))
    for tag in tags:
        if tag not in orig_tags:
            text = text.replace(tag, "")
    return text


def translate_block(
    text_lines: list[str],
    past_lines: list[list[str]],
    future_lines: list[list[str]],
    model: str,
    port: int,
) -> list[str] | None:
    current = "\n".join(text_lines)
    user_msg = build_user_message(current, past_lines, future_lines)
    response = call_ollama(SYSTEM_PROMPT, user_msg, model, port)
    if response is None:
        return None
    response = response.strip()
    if not response:
        print("\n⚠ Empty response, keeping original.")
        return text_lines
    result = response.split("\n")
    result = [strip_hallucinated_tags(line, current) for line in result]
    return result


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def _should_skip_srt(block: dict) -> bool:
    text = "".join(block["text"]).lower()
    markers = ["♪", "♫", "[music]", "[sound]", "[laughter]", "[applause]"]
    return any(m in text for m in markers)


def _process_srt(
    blocks: list[dict],
    model: str,
    port: int,
    bar: ProgressBar,
    completed: set[int],
) -> tuple[str, int, int, int]:
    translated = 0
    skipped = 0
    errors = 0

    for i in range(len(blocks)):
        if i in completed:
            bar.update()
            continue

        if _should_skip_srt(blocks[i]):
            skipped += 1
            completed.add(i)
            bar.update()
            continue

        past = [b["text"] for b in blocks[max(0, i - WINDOW_PAST):i]]
        future = [b["text"] for b in blocks[i + 1:i + 1 + WINDOW_FUTURE]]

        result = translate_block(blocks[i]["text"], past, future, model, port)

        if result is None:
            errors += 1
        else:
            if result != blocks[i]["text"]:
                translated += 1
            blocks[i]["text"] = result
            completed.add(i)
        bar.update()

    return build_srt(blocks), translated, skipped, errors


def _process_ass(
    header: list[str],
    events: list[dict],
    model: str,
    port: int,
    bar: ProgressBar,
    completed: set[int],
) -> tuple[str, int, int, int]:
    translated = 0
    skipped = 0
    errors = 0
    dialogue_idx = [i for i, e in enumerate(events) if e["type"] == "Dialogue"]

    for pos, idx in enumerate(dialogue_idx):
        if pos in completed:
            bar.update()
            continue

        ev = events[idx]
        text, tf = ass_text_field(ev["raw"], ev["fields"])
        if tf < 0 or not text.strip() or ass_has_drawing(text):
            skipped += 1
            completed.add(pos)
            bar.update()
            continue

        text_lines = text.split("\\N")
        past_idx = dialogue_idx[max(0, pos - WINDOW_PAST):pos]
        future_idx = dialogue_idx[pos + 1:pos + 1 + WINDOW_FUTURE]
        past_lines = [ass_text_field(events[pi]["raw"], events[pi]["fields"])[0].split("\\N") for pi in past_idx]
        future_lines = [ass_text_field(events[fi]["raw"], events[fi]["fields"])[0].split("\\N") for fi in future_idx]

        result = translate_block(text_lines, past_lines, future_lines, model, port)

        if result is None:
            errors += 1
        else:
            new_text = "\\N".join(result)
            if new_text != text:
                translated += 1
            events[idx]["raw"] = ass_set_text(ev["raw"], new_text, ev["fields"])
            completed.add(pos)
        bar.update()

    out = "\n".join(header) + "\n" + "\n".join(e["raw"] for e in events) + "\n"
    return out, translated, skipped, errors


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def process_file(
    input_path: Path,
    output_path: Path,
    model: str,
    port: int,
    test_limit: int | None = None,
    resume: bool = False,
) -> int:
    encoding = detect_encoding(input_path)
    with open(input_path, "r", encoding=encoding) as f:
        content = f.read()

    ext = input_path.suffix.lower()

    if ext == ".srt":
        blocks = parse_srt(content)
        if test_limit:
            blocks = blocks[:test_limit]
        total = len(blocks)
        if total == 0:
            print("✘ No content found.")
            return 1
        cp_file = checkpoint_path(output_path)
        completed: set[int] = set()
        if resume:
            loaded = load_checkpoint(cp_file, model)
            if loaded is not None:
                completed = loaded
                print(f"↻ Resuming — {len(completed)}/{total} blocks already done.")
        print(f"\n📄 {input_path.name}  ({total} blocks{' · test mode' if test_limit else ''})")
        bar = ProgressBar(total, prefix="Translating")
        output, tc, sc, ec = _process_srt(blocks, model, port, bar, completed)
        bar.close()
        save_checkpoint(cp_file, model, completed)
        if len(completed) >= total:
            clear_checkpoint(cp_file)

    elif ext == ".ass":
        header, events = parse_ass(content)
        dialogue_idx = [i for i, e in enumerate(events) if e["type"] == "Dialogue"]
        if test_limit:
            dialogue_idx = dialogue_idx[:test_limit]
        total = len(dialogue_idx)
        if total == 0:
            print("✘ No content found.")
            return 1
        cp_file = checkpoint_path(output_path)
        completed = set()
        if resume:
            loaded = load_checkpoint(cp_file, model)
            if loaded is not None:
                completed = loaded
                print(f"↻ Resuming — {len(completed)}/{total} blocks already done.")
        print(f"\n📄 {input_path.name}  ({total} blocks{' · test mode' if test_limit else ''})")
        bar = ProgressBar(total, prefix="Translating")
        output, tc, sc, ec = _process_ass(header, events, model, port, bar, completed)
        bar.close()
        save_checkpoint(cp_file, model, completed)
        if len(completed) >= total:
            clear_checkpoint(cp_file)
    else:
        print(f"✘ Unsupported: {ext}")
        return 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"✔ Saved  → {output_path}")
    print(f"   Translated: {tc}  |  Skipped: {sc}  |  Errors: {ec}")
    return 0


def find_subtitle_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("*.srt")) + sorted(pp.glob("*.ass")))
        elif pp.exists() and pp.suffix.lower() in (".srt", ".ass"):
            files.append(pp)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate EN subtitles (SRT/ASS) → BG (Cyrillic) via local Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python subs_translator.py episode.srt\n"
            "  python subs_translator.py -m gemma4:e2b --test 5 episode.srt\n"
            "  python subs_translator.py --resume episode.srt\n"
            "  python subs_translator.py -o ~/BG_subs *.ass\n"
        ),
    )
    parser.add_argument("input", nargs="+", help="SRT/ASS file(s) or directory")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                        help=f"Ollama port (default: {DEFAULT_PORT})")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--test", type=int, default=None, metavar="N",
                        help="Process only first N blocks (quick validation)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved checkpoint")

    args = parser.parse_args()
    ensure_ollama(args.port)
    ensure_model(args.model)

    files = find_subtitle_files(args.input)
    if not files:
        print("✘ No .srt / .ass files found.")
        sys.exit(1)

    for fpath in files:
        out_dir = Path(args.output_dir) if args.output_dir else fpath.parent
        if args.output_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fpath.stem}_bg{fpath.suffix}"
        rc = process_file(fpath, out_path, args.model, args.port, args.test, args.resume)
        if rc != 0:
            print(f"✘ Failed: {fpath.name}")


if __name__ == "__main__":
    main()
