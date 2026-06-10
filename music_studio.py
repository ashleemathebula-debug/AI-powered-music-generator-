"""Music generator studio.

This app wraps the existing symbolic music generator in a local web UI and adds
simple audio rendering so generated music can be previewed as a WAV file.

What it does:
- accepts uploaded/plain text training corpora
- trains the symbolic generator
- generates a new score from a prompt and sampling controls
- exports MIDI and WAV artifacts
- serves everything from a local browser UI

It depends only on the Python standard library plus the existing
music_generator.py module in this folder.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import math
import mimetypes
import os
import json
import tempfile
import threading
import urllib.parse
import webbrowser
import wave
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Sequence

from music_generator import AdaptiveMusicGenerator, DEFAULT_CORPUS, build_midi, load_corpus, parse_token, token_to_musical


@dataclass(frozen=True)
class RenderConfig:
    sample_rate: int = 44100
    amplitude: float = 0.28


def _format_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _frequency_from_midi(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def _sine_sample(frequency: float, time_value: float) -> float:
    return math.sin(2 * math.pi * frequency * time_value)


def _adsr_envelope(position: float, attack: float, decay: float, sustain_level: float, release: float) -> float:
    if position < attack:
        return position / max(attack, 1e-9)
    position -= attack
    if position < decay:
        if decay <= 0:
            return sustain_level
        return 1.0 - (1.0 - sustain_level) * (position / decay)
    return sustain_level if release <= 0 else max(0.0, sustain_level * (1.0 - position / release))


class WavRenderer:
    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    def render_score(self, score: dict[str, object], tempo_bpm: int) -> bytes:
        melody_tokens = [token_to_musical(token) for token in score["melody"]]
        accompaniment = score["accompaniment"]
        return self._render_from_events(melody_tokens, accompaniment, tempo_bpm)

    def _render_from_events(self, melody_tokens: Sequence, accompaniment: Sequence[dict[str, object]], tempo_bpm: int) -> bytes:
        sample_rate = self.config.sample_rate
        beat_seconds = 60.0 / max(tempo_bpm, 1)

        melody_events: list[tuple[float, float, int, float]] = []
        cursor = 0.0
        for token in melody_tokens:
            if token.pitch == "REST":
                cursor += token.duration * beat_seconds
                continue
            midi_note = _pitch_to_midi(token.pitch)
            duration_seconds = token.duration * beat_seconds
            melody_events.append((cursor, cursor + duration_seconds, midi_note, 1.0))
            cursor += duration_seconds

        accompaniment_events: list[tuple[float, float, int, float]] = []
        for bar in accompaniment:
            start_time = (int(bar["start_tick"]) / 480.0) * beat_seconds
            for note in bar["notes"]:
                start = start_time
                end = start_time + (int(note["duration"]) / 480.0) * beat_seconds
                accompaniment_events.append((start, end, int(note["midi"]), 0.45))

        length_seconds = 0.0
        for start, end, _, _ in melody_events + accompaniment_events:
            length_seconds = max(length_seconds, end)
        length_seconds += beat_seconds * 1.5

        total_frames = max(1, int(math.ceil(length_seconds * sample_rate)))
        left = [0.0] * total_frames
        right = [0.0] * total_frames

        self._mix_events(left, right, melody_events, sample_rate, beat_seconds, stereo_spread=0.15, timbre=0.55)
        self._mix_events(left, right, accompaniment_events, sample_rate, beat_seconds, stereo_spread=0.4, timbre=0.32)
        return self._pack_wav(left, right, sample_rate)

    def _mix_events(
        self,
        left: list[float],
        right: list[float],
        events: Sequence[tuple[float, float, int, float]],
        sample_rate: int,
        beat_seconds: float,
        *,
        stereo_spread: float,
        timbre: float,
    ) -> None:
        for start_time, end_time, midi_note, gain in events:
            frequency = _frequency_from_midi(midi_note)
            start_frame = max(0, int(start_time * sample_rate))
            end_frame = min(len(left), int(end_time * sample_rate))
            if end_frame <= start_frame:
                continue
            duration_frames = max(end_frame - start_frame, 1)
            attack = min(duration_frames / sample_rate, 0.025 + beat_seconds * 0.01)
            decay = min(duration_frames / sample_rate, 0.10 + beat_seconds * 0.03)
            release = min(duration_frames / sample_rate, 0.12 + beat_seconds * 0.06)
            sustain_level = 0.65

            for frame in range(start_frame, end_frame):
                time_value = (frame - start_frame) / sample_rate
                note_age = time_value
                note_length = duration_frames / sample_rate
                tail_position = max(0.0, note_length - note_age)
                envelope = _adsr_envelope(note_age, attack, decay, sustain_level, tail_position if note_age > note_length - release else 0.0)
                if envelope <= 0:
                    continue

                harmonic = _sine_sample(frequency, time_value)
                overtone = _sine_sample(frequency * 2.0, time_value) * timbre * 0.45
                body = harmonic * (0.72 + timbre * 0.12) + overtone
                sample = body * envelope * gain * self.config.amplitude
                stereo_panning = stereo_spread * math.sin(frequency / 180.0)
                left[frame] += sample * (1.0 - stereo_panning)
                right[frame] += sample * (1.0 + stereo_panning)

    def _pack_wav(self, left: Sequence[float], right: Sequence[float], sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            frames = bytearray()
            for left_sample, right_sample in zip(left, right):
                l_value = int(max(-1.0, min(1.0, left_sample)) * 32767)
                r_value = int(max(-1.0, min(1.0, right_sample)) * 32767)
                frames.extend(l_value.to_bytes(2, byteorder="little", signed=True))
                frames.extend(r_value.to_bytes(2, byteorder="little", signed=True))
            wav_file.writeframes(bytes(frames))
        return buffer.getvalue()


def _pitch_to_midi(pitch: str) -> int:
    note = pitch[:-1].upper()
    octave = int(pitch[-1])
    note_map = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }
    return (octave + 1) * 12 + note_map[note]


class StudioState:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_dir = root / "generated_music"
        self.output_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.last_result: dict[str, object] | None = None

    def make_generator(self, *, corpus_paths: Sequence[str] | None, order: int, alpha: float, seed: int | None) -> AdaptiveMusicGenerator:
        with self._lock:
            generator = AdaptiveMusicGenerator(order=order, alpha=alpha, seed=seed)
            corpus = load_corpus(corpus_paths)
            generator.train(corpus)
            return generator

    def save_artifacts(self, score: dict[str, object], wav_bytes: bytes, midi_bytes: bytes) -> dict[str, str]:
        prefix = score["key"]["tonic"].lower() + "_" + score["key"]["mode"]
        stamp = f"{prefix}_{len(score['melody'])}_{abs(hash(tuple(score['melody']))) & 0xFFFF:04x}"
        wav_path = self.output_dir / f"{stamp}.wav"
        midi_path = self.output_dir / f"{stamp}.mid"
        json_path = self.output_dir / f"{stamp}.json"
        wav_path.write_bytes(wav_bytes)
        midi_path.write_bytes(midi_bytes)
        json_path.write_text(_result_json(score), encoding="utf-8")
        return {
            "wav": wav_path.name,
            "midi": midi_path.name,
            "json": json_path.name,
        }


def _result_json(score: dict[str, object]) -> str:
    return json.dumps(score, indent=2, ensure_ascii=True)


def _download_link(label: str, path: str) -> str:
    return f'<a class="download" href="/files/{html.escape(path)}" download>{html.escape(label)}</a>'


def _render_home_page(state: StudioState, message: str = "") -> str:
    existing = sorted(state.output_dir.glob("*.mid"), key=lambda item: item.stat().st_mtime, reverse=True)[:8]
    history = "".join(
        f'<li><a href="/files/{html.escape(item.name)}">{html.escape(item.name)}</a></li>'
        for item in existing
    )
    message_block = f'<div class="message">{html.escape(message)}</div>' if message else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Music Studio</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07111f;
      --panel: rgba(10, 18, 32, 0.88);
      --border: rgba(148, 163, 184, 0.16);
      --text: #e6eef7;
      --muted: #9fb0c3;
      --accent: #7dd3fc;
      --accent-2: #fbbf24;
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    body {{ margin: 0; background: radial-gradient(circle at top left, rgba(125, 211, 252, 0.16), transparent 30%), linear-gradient(180deg, #07111f 0%, #050913 100%); color: var(--text); }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 18px 48px; }}
    .hero {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; margin-bottom: 18px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 22px; padding: 22px; backdrop-filter: blur(14px); }}
    h1 {{ margin: 0 0 10px; font-size: clamp(30px, 5vw, 56px); line-height: 1; }}
    .sub {{ color: var(--muted); line-height: 1.55; max-width: 68ch; }}
    form {{ display: grid; gap: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    label {{ display: grid; gap: 8px; font-size: 14px; color: var(--muted); }}
    input, textarea {{ width: 100%; box-sizing: border-box; border-radius: 14px; border: 1px solid var(--border); background: rgba(5, 9, 19, 0.85); color: var(--text); padding: 12px 14px; font: inherit; }}
    textarea {{ min-height: 190px; resize: vertical; }}
        .dropzone {{
            display: grid;
            gap: 10px;
            padding: 18px;
            border-radius: 18px;
            border: 1px dashed rgba(125, 211, 252, 0.35);
            background: rgba(5, 9, 19, 0.55);
            transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
        }}
        .dropzone.is-dragover {{
            border-color: rgba(251, 191, 36, 0.9);
            background: rgba(251, 191, 36, 0.10);
            transform: translateY(-1px);
        }}
        .dropzone-title {{ font-size: 15px; color: var(--text); font-weight: 700; }}
        .dropzone-subtitle {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
        .file-list {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height: 1.6; }}
        .file-tools {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
        .file-tools input[type="file"] {{ padding: 10px 12px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    button, .download {{ display: inline-flex; align-items: center; justify-content: center; border: 0; border-radius: 999px; padding: 12px 18px; background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #08111f; font-weight: 700; text-decoration: none; cursor: pointer; }}
    .secondary {{ background: rgba(148, 163, 184, 0.14); color: var(--text); border: 1px solid var(--border); }}
    .message {{ margin-top: 12px; padding: 12px 14px; border-radius: 14px; background: rgba(125, 211, 252, 0.12); border: 1px solid rgba(125, 211, 252, 0.25); }}
    .history {{ margin: 0; padding-left: 18px; line-height: 1.7; color: var(--muted); }}
    .downloads {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    @media (max-width: 920px) {{ .hero, .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <div class="panel">
        <h1>AI Music Studio</h1>
        <div class="sub">Train a symbolic music model on note-token text, generate fresh melodies, export MIDI, and render a playable WAV preview from the browser.</div>
        {message_block}
        <div class="downloads" style="margin-top: 16px;">
          {_download_link("Open generated files folder", ".")}
        </div>
      </div>
      <div class="panel">
        <h2 style="margin-top:0;">Recent exports</h2>
        <ul class="history">{history or '<li>No exports yet.</li>'}</ul>
      </div>
    </section>

    <section class="panel">
      <form method="post" action="/generate">
                <div class="dropzone" id="dropzone">
                    <div class="dropzone-title">Drag and drop text corpus files here</div>
                    <div class="dropzone-subtitle">Drop one or more `.txt` files to append their contents into the training corpus. This studio trains on symbolic note-token text, not binary MIDI files.</div>
                    <div class="file-tools">
                        <input id="corpusFiles" type="file" accept=".txt,text/plain" multiple />
                        <button class="secondary" id="loadFilesBtn" type="button">Load selected files</button>
                    </div>
                    <ul class="file-list" id="fileList">
                        <li>No files loaded yet.</li>
                    </ul>
                </div>
        <label>
          Training corpus
          <textarea name="corpus" placeholder="Enter note-token phrases like: C4/1 D4/1 E4/1 REST/0.5">{html.escape('\n'.join(DEFAULT_CORPUS))}</textarea>
        </label>
        <div class="grid">
          <label>Prompt<input name="prompt" value="C4/1 D4/1" /></label>
          <label>Output name<input name="name" value="studio_track" /></label>
          <label>Bars<input name="bars" type="number" min="1" max="64" value="8" /></label>
          <label>Tempo<input name="tempo" type="number" min="30" max="260" value="110" /></label>
          <label>Order<input name="order" type="number" min="1" max="8" value="4" /></label>
          <label>Alpha<input name="alpha" type="number" step="0.05" min="0.05" max="3" value="0.7" /></label>
          <label>Temperature<input name="temperature" type="number" step="0.05" min="0.2" max="2.5" value="0.9" /></label>
          <label>Seed<input name="seed" type="number" placeholder="optional" /></label>
        </div>
        <div class="row">
          <button type="submit">Generate music</button>
          <button class="secondary" type="reset">Reset</button>
        </div>
      </form>
    </section>
  </main>
    <script>
        (() => {{
            const textarea = document.querySelector('textarea[name="corpus"]');
            const fileInput = document.getElementById('corpusFiles');
            const fileList = document.getElementById('fileList');
            const dropzone = document.getElementById('dropzone');
            const loadFilesBtn = document.getElementById('loadFilesBtn');

            const renderFileList = (names) => {{
                fileList.innerHTML = '';
                if (!names.length) {{
                    const item = document.createElement('li');
                    item.textContent = 'No files loaded yet.';
                    fileList.appendChild(item);
                    return;
                }}
                for (const name of names) {{
                    const item = document.createElement('li');
                    item.textContent = name;
                    fileList.appendChild(item);
                }}
            }};

            const appendFiles = async (files) => {{
                const loadedNames = [];
                const chunks = [];
                for (const file of files) {{
                    if (!file.name.toLowerCase().endsWith('.txt') && file.type && file.type !== 'text/plain') {{
                        continue;
                    }}
                    const text = await file.text();
                    const cleaned = text.trim();
                    if (cleaned) {{
                        chunks.push(cleaned);
                        loadedNames.push(file.name);
                    }}
                }}
                if (chunks.length) {{
                    const current = textarea.value.trim();
                    textarea.value = [current, ...chunks].filter(Boolean).join('\n');
                }}
                renderFileList(loadedNames);
            }};

            loadFilesBtn.addEventListener('click', async () => {{
                await appendFiles(fileInput.files || []);
            }});

            fileInput.addEventListener('change', async () => {{
                await appendFiles(fileInput.files || []);
            }});

            dropzone.addEventListener('dragover', (event) => {{
                event.preventDefault();
                dropzone.classList.add('is-dragover');
            }});

            dropzone.addEventListener('dragleave', () => {{
                dropzone.classList.remove('is-dragover');
            }});

            dropzone.addEventListener('drop', async (event) => {{
                event.preventDefault();
                dropzone.classList.remove('is-dragover');
                const files = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files : [];
                fileInput.files = event.dataTransfer.files;
                await appendFiles(files);
            }});

            renderFileList([]);
        }})();
    </script>
</body>
</html>"""


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "MusicStudio/1.0"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._write_html(_render_home_page(self.server.state))
            return
        if parsed.path.startswith("/files/"):
            self._serve_file(parsed.path.removeprefix("/files/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/generate":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        form = urllib.parse.parse_qs(raw_body)

        corpus = form.get("corpus", [""])[0].strip()
        prompt = form.get("prompt", [""])[0].strip()
        name = _safe_name(form.get("name", ["studio_track"])[0].strip()) or "studio_track"
        bars = max(1, int(form.get("bars", ["8"])[0]))
        tempo = max(30, min(260, int(form.get("tempo", ["110"])[0])))
        order = max(1, min(8, int(form.get("order", ["4"])[0])))
        alpha = max(0.05, min(3.0, float(form.get("alpha", ["0.7"])[0])))
        temperature = max(0.2, min(2.5, float(form.get("temperature", ["0.9"])[0])))
        seed_value = form.get("seed", [""])[0].strip()
        seed = int(seed_value) if seed_value else None

        if corpus:
            texts = [line.strip() for line in corpus.splitlines() if line.strip()]
        else:
            texts = list(DEFAULT_CORPUS)

        try:
            generator = AdaptiveMusicGenerator(order=order, alpha=alpha, seed=seed)
            generator.train(texts)
            score = generator.generate_score(
                bars=bars,
                prompt=prompt,
                temperature=temperature,
                seed=seed,
            )
            midi_bytes = build_midi([token_to_musical(token) for token in score["melody"]], score["accompaniment"], tempo_bpm=tempo)
            wav_bytes = WavRenderer().render_score(score, tempo)
            files = self.server.state.save_artifacts(score, wav_bytes, midi_bytes)
            generator_path = self.server.state.output_dir / f"{name}.json"
            generator.save(generator_path)
            message = f"Generated {name}: key {score['key']['tonic']} {score['key']['mode']}"
            html_page = _render_result_page(name, score, files, message)
            self._write_html(html_page)
        except Exception as exc:
            self._write_html(_render_home_page(self.server.state, f"Generation failed: {exc}"), status=HTTPStatus.BAD_REQUEST)

    def _serve_file(self, name: str) -> None:
        safe_name = Path(name).name
        file_path = self.server.state.output_dir / safe_name
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime_type, _ = mimetypes.guess_type(file_path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def _write_html(self, text: str, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class MusicStudioServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], RequestHandlerClass, state: StudioState) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.state = state


def _render_result_page(name: str, score: dict[str, object], files: dict[str, str], message: str) -> str:
    melody_preview = " ".join(html.escape(token) for token in score["melody"][:32])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(name)} - Music Studio</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:rgba(10,18,32,0.88); --border:rgba(148,163,184,0.16); --text:#e6eef7; --muted:#9fb0c3; --accent:#7dd3fc; --accent2:#fbbf24; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin:0; background:linear-gradient(180deg,#07111f 0%,#050913 100%); color:var(--text); }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:32px 18px 48px; }}
    .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:22px; padding:22px; backdrop-filter:blur(14px); margin-bottom:18px; }}
    .sub {{ color:var(--muted); line-height:1.6; }}
    .downloads {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; }}
    a {{ color:#08111f; background:linear-gradient(135deg,var(--accent),var(--accent2)); text-decoration:none; padding:12px 18px; border-radius:999px; font-weight:700; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:rgba(5,9,19,0.85); border:1px solid var(--border); padding:14px; border-radius:14px; overflow:auto; }}
    .meta {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .card {{ border:1px solid var(--border); border-radius:14px; padding:14px; background:rgba(5,9,19,0.6); }}
    .label {{ color:var(--muted); font-size:13px; margin-bottom:6px; }}
    @media (max-width: 840px) {{ .meta {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="panel">
      <h1 style="margin-top:0;">{html.escape(name)} generated</h1>
      <div class="sub">{html.escape(message)}</div>
      <div class="downloads">
        <a href="/files/{html.escape(files['wav'])}">Download WAV</a>
        <a href="/files/{html.escape(files['midi'])}">Download MIDI</a>
        <a href="/files/{html.escape(files['json'])}">Download JSON</a>
        <a href="/">Generate another</a>
      </div>
    </section>
    <section class="panel meta">
      <div class="card"><div class="label">Key</div><strong>{html.escape(score['key']['tonic'])} {html.escape(score['key']['mode'])}</strong></div>
      <div class="card"><div class="label">Melody length</div><strong>{len(score['melody'])} tokens</strong></div>
      <div class="card"><div class="label">Register</div><strong>{score['register']}</strong></div>
      <div class="card"><div class="label">Preview</div><strong>WAV and MIDI ready</strong></div>
    </section>
    <section class="panel">
      <h2 style="margin-top:0;">Melody preview</h2>
      <pre>{melody_preview}</pre>
    </section>
  </main>
</body>
</html>"""


def _safe_name(name: str) -> str:
    allowed = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name]
    result = "".join(allowed).strip("_")
    return result or "studio_track"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI music generator studio")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--open-browser", action="store_true", help="Open the web UI in your browser")
    parser.add_argument("--serve", action="store_true", help="Run the local web studio")
    return parser


def run_server(host: str, port: int, open_browser: bool) -> None:
    state = StudioState(Path(__file__).resolve().parent)
    server = MusicStudioServer((host, port), StudioHandler, state)
    url = f"http://{host}:{port}/"
    print(f"Music studio running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.serve or args.open_browser:
        run_server(args.host, args.port, args.open_browser)
    else:
        run_server(args.host, args.port, args.open_browser)
