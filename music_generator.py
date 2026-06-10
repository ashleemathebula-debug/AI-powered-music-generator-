"""AI-powered symbolic music generator.

This module trains a compact probabilistic music model from tokenized note
corpora and renders the result as a standard MIDI file.

Features:
- learn style from note-token text corpora
- infer a likely key and melodic range
- generate melody with backoff sampling and style bias
- add a simple harmonic accompaniment
- export a valid MIDI file without third-party packages

Training corpus format:
    One phrase per line, tokens separated by spaces.
    Each token is a note or rest with an optional duration:
        C4/1   D4/0.5   REST/0.5   G4/2

Example:
    python music_generator.py --train corpus.txt --output song.mid
    python music_generator.py --bars 16 --output demo.mid
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

START_TOKEN = "<s>"
END_TOKEN = "</s>"
REST_TOKEN = "REST"
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO_BPM = 110
NOTE_PATTERN = re.compile(r"^(REST|[A-Ga-g][#b]?-?\d)(?:/([0-9]*\.?[0-9]+))?$")
TOKEN_SPLIT_PATTERN = re.compile(r"\s+")

PITCH_CLASS_TO_NAME = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}
NOTE_TO_PITCH_CLASS = {
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
SCALE_PATTERNS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
}
DEFAULT_CORPUS = [
    "C4/1 D4/1 E4/1 G4/1 A4/1 G4/1 E4/1 D4/1",
    "E4/0.5 G4/0.5 A4/1 G4/1 E4/1 D4/1 C4/2",
    "G3/1 C4/1 E4/1 G4/1 E4/1 C4/1 G3/2",
    "A3/1 C4/1 E4/1 A4/1 G4/1 E4/1 D4/1 C4/2",
    "D4/1 F4/1 A4/1 D5/1 C5/1 A4/1 F4/1 D4/2",
    "E4/1 REST/0.5 E4/0.5 G4/1 A4/1 G4/1 E4/2",
    "C4/0.5 D4/0.5 E4/1 G4/1 A4/1 G4/1 E4/2",
    "F3/1 A3/1 C4/1 F4/1 E4/1 C4/1 A3/1 F3/2",
]


@dataclass(frozen=True)
class MusicalToken:
    pitch: str
    duration: float

    def to_text(self) -> str:
        return f"{self.pitch}/{_format_duration(self.duration)}"


@dataclass(frozen=True)
class MidiNote:
    start_tick: int
    end_tick: int
    midi_note: int
    channel: int
    velocity: int = 96


@dataclass(frozen=True)
class MidiEvent:
    tick: int
    kind: str
    channel: int
    midi_note: int
    velocity: int = 0


@dataclass
class StyleProfile:
    pitch_class_counts: Counter[int] = field(default_factory=Counter)
    pitch_counts: Counter[int] = field(default_factory=Counter)
    duration_counts: Counter[float] = field(default_factory=Counter)
    interval_counts: Counter[int] = field(default_factory=Counter)
    median_pitch: float = 60.0
    key_tonic_pc: int = 0
    key_mode: str = "major"

    def infer_key(self) -> tuple[int, str]:
        best_score = -float("inf")
        best_tonic = 0
        best_mode = "major"

        for tonic in range(12):
            for mode, intervals in SCALE_PATTERNS.items():
                scale = {(tonic + interval) % 12 for interval in intervals}
                in_scale = sum(self.pitch_class_counts.get(pc, 0) for pc in scale)
                tonic_bonus = self.pitch_class_counts.get(tonic, 0) * 1.3
                dominant_bonus = self.pitch_class_counts.get((tonic + 7) % 12, 0) * 0.35
                outside = sum(count for pc, count in self.pitch_class_counts.items() if pc not in scale)
                score = in_scale * 1.2 + tonic_bonus + dominant_bonus - outside * 0.2
                if score > best_score:
                    best_score = score
                    best_tonic = tonic
                    best_mode = mode

        self.key_tonic_pc = best_tonic
        self.key_mode = best_mode
        return best_tonic, best_mode

    def scale_set(self) -> set[int]:
        intervals = SCALE_PATTERNS[self.key_mode]
        return {(self.key_tonic_pc + interval) % 12 for interval in intervals}

    def common_durations(self, limit: int = 4) -> list[float]:
        return [duration for duration, _ in self.duration_counts.most_common(limit)]

    def preferred_register(self) -> int:
        if self.pitch_counts:
            return int(round(self.median_pitch))
        return 60


class AdaptiveMusicGenerator:
    def __init__(self, order: int = 4, alpha: float = 0.7, seed: int | None = None) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.seed = seed
        self.context_counts: dict[int, defaultdict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, order + 1)
        }
        self.unigram_counts: Counter[str] = Counter()
        self.vocab: set[str] = {START_TOKEN, END_TOKEN}
        self.profile = StyleProfile()

    def train(self, texts: Iterable[str]) -> None:
        phrase_count = 0
        for text in texts:
            for phrase in _split_phrases(text):
                tokens = _normalize_phrase_tokens(phrase)
                if not tokens:
                    continue
                phrase_count += 1
                self._observe_phrase(tokens)
                self._observe_style(tokens)

        if phrase_count == 0:
            raise ValueError("No usable musical phrases were found")

        if not self.profile.pitch_class_counts:
            raise ValueError("Training data must contain note tokens like C4/1 or REST/0.5")

        self.profile.infer_key()
        self.vocab.add(REST_TOKEN)

    def _observe_phrase(self, tokens: Sequence[str]) -> None:
        padded = [START_TOKEN] * (self.order - 1) + list(tokens) + [END_TOKEN]
        for index, token in enumerate(padded):
            self.vocab.add(token)
            self.unigram_counts[token] += 1
            for n in range(2, self.order + 1):
                if index < n - 1:
                    continue
                context = tuple(padded[index - (n - 1) : index])
                self.context_counts[n][context][token] += 1

    def _observe_style(self, tokens: Sequence[str]) -> None:
        previous_midi: int | None = None
        midi_values: list[int] = []
        for token in tokens:
            parsed = parse_token(token)
            if parsed is None:
                continue
            pitch_name, duration = parsed
            self.profile.duration_counts[duration] += 1
            if pitch_name == REST_TOKEN:
                previous_midi = None
                continue
            midi_note = note_name_to_midi(pitch_name)
            midi_values.append(midi_note)
            self.profile.pitch_counts[midi_note] += 1
            self.profile.pitch_class_counts[midi_note % 12] += 1
            if previous_midi is not None:
                self.profile.interval_counts[midi_note - previous_midi] += 1
            previous_midi = midi_note

        if midi_values:
            sorted_values = sorted(midi_values)
            middle = len(sorted_values) // 2
            if len(sorted_values) % 2:
                self.profile.median_pitch = float(sorted_values[middle])
            else:
                self.profile.median_pitch = (sorted_values[middle - 1] + sorted_values[middle]) / 2

    def _candidate_scores(self, context: Sequence[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        context = tuple(context)
        max_order = min(self.order, len(context) + 1)
        key_scale = self.profile.scale_set()
        common_durations = set(self.profile.common_durations())
        target_register = self.profile.preferred_register()
        recent_tokens = list(context[-6:])
        recent_pitch = _last_pitch_from_tokens(recent_tokens)

        for n in range(max_order, 1, -1):
            window = context[-(n - 1) :]
            bucket = self.context_counts[n].get(tuple(window))
            if not bucket:
                continue

            total = sum(bucket.values()) + self.alpha * len(self.vocab)
            backoff_weight = 1.0 + math.log1p(sum(bucket.values()))

            for token in self.vocab:
                count = bucket.get(token, 0)
                probability = (count + self.alpha) / total
                score = backoff_weight * probability
                scores[token] = scores.get(token, 0.0) + score

        if not scores:
            total = sum(self.unigram_counts.values()) + self.alpha * len(self.vocab)
            for token in self.vocab:
                count = self.unigram_counts.get(token, 0)
                scores[token] = (count + self.alpha) / total
        else:
            unigram_total = sum(self.unigram_counts.values()) + self.alpha * len(self.vocab)
            for token in self.vocab:
                count = self.unigram_counts.get(token, 0)
                scores[token] += 0.12 * ((count + self.alpha) / unigram_total)

        adjusted: dict[str, float] = {}
        for token, score in scores.items():
            parsed = parse_token(token)
            if parsed is None:
                adjusted[token] = score
                continue

            pitch_name, duration = parsed
            if pitch_name == START_TOKEN:
                continue
            if pitch_name == END_TOKEN:
                adjusted[token] = score * 0.82
                continue

            if pitch_name == REST_TOKEN:
                if recent_tokens and recent_tokens[-1].startswith(REST_TOKEN):
                    score *= 0.7
                score *= 0.9
            else:
                midi_note = note_name_to_midi(pitch_name)
                pitch_class = midi_note % 12
                if pitch_class in key_scale:
                    score *= 1.18
                else:
                    score *= 0.83

                if recent_pitch is not None:
                    interval = abs(midi_note - recent_pitch)
                    score *= 1.0 / (1.0 + interval / 8.0)
                    if interval > 12:
                        score *= 0.75
                distance_from_register = abs(midi_note - target_register)
                score *= 1.0 / (1.0 + distance_from_register / 30.0)

            if duration in common_durations:
                score *= 1.12
            else:
                score *= 0.95

            if token in recent_tokens[-3:]:
                score *= 0.72

            adjusted[token] = score

        return adjusted

    def _sample_token(
        self,
        context: Sequence[str],
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        rng: random.Random,
    ) -> str:
        scores = self._candidate_scores(context)
        filtered = [(token, score) for token, score in scores.items() if score > 0 and token != START_TOKEN]
        if not filtered:
            return END_TOKEN

        filtered.sort(key=lambda item: item[1], reverse=True)

        if top_k is not None and top_k > 0:
            filtered = filtered[:top_k]

        if top_p is not None and 0 < top_p < 1:
            total_mass = sum(score for _, score in filtered)
            cumulative = 0.0
            nucleus: list[tuple[str, float]] = []
            for token, score in filtered:
                nucleus.append((token, score))
                cumulative += score / total_mass if total_mass else 0.0
                if cumulative >= top_p:
                    break
            if nucleus:
                filtered = nucleus

        temperature = max(temperature, 1e-6)
        scaled = [(token, score ** (1.0 / temperature)) for token, score in filtered]
        total = sum(score for _, score in scaled)
        if total <= 0:
            return filtered[0][0]

        threshold = rng.random() * total
        cumulative = 0.0
        for token, score in scaled:
            cumulative += score
            if cumulative >= threshold:
                return token
        return scaled[-1][0]

    def generate_melody(
        self,
        *,
        bars: int = 8,
        beats_per_bar: float = 4.0,
        prompt: str = "",
        temperature: float = 0.9,
        top_k: int | None = 32,
        top_p: float | None = 0.9,
        seed: int | None = None,
    ) -> list[MusicalToken]:
        if bars < 1:
            raise ValueError("bars must be at least 1")
        if beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be positive")

        rng = random.Random(self.seed if seed is None else seed)
        prompt_tokens = _normalize_phrase_tokens(prompt) if prompt else []
        tokens = list(prompt_tokens)
        context = ([START_TOKEN] * max(0, self.order - 1 - len(prompt_tokens))) + prompt_tokens[-(self.order - 1) :]

        target_beats = bars * beats_per_bar
        current_beats = sum(parse_token(token)[1] for token in tokens if parse_token(token))
        if current_beats == 0 and not tokens:
            tokens.extend(self._default_seed_motif())
            current_beats = sum(parse_token(token)[1] for token in tokens if parse_token(token))
            context = ([START_TOKEN] * max(0, self.order - 1 - len(tokens))) + tokens[-(self.order - 1) :]

        while current_beats < target_beats:
            next_token = self._sample_token(
                context,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                rng=rng,
            )
            if next_token == END_TOKEN:
                if current_beats < target_beats * 0.5:
                    next_token = self._fallback_continuation(context)
                else:
                    break

            parsed = parse_token(next_token)
            if parsed is None:
                break

            tokens.append(next_token)
            context = (context + [next_token])[-(self.order - 1) :]
            current_beats += parsed[1]

            if current_beats >= target_beats:
                break

        if current_beats < target_beats:
            tokens.extend(self._fill_time(target_beats - current_beats))

        return [token_to_musical(token) for token in tokens if parse_token(token) is not None]

    def _default_seed_motif(self) -> list[str]:
        tonic = self.profile.key_tonic_pc if self.profile.pitch_class_counts else 0
        mode = self.profile.key_mode
        scale = SCALE_PATTERNS[mode]
        mid_register = self.profile.preferred_register()
        tonic_midi = _closest_midi_for_pitch_class(mid_register, tonic)
        degree_map = [0, 2, 4, 5, 7, 4, 2, 0]
        motif: list[str] = []
        for degree in degree_map:
            midi_note = _midi_from_scale_degree(tonic, mode, degree, tonic_midi)
            motif.append(f"{midi_to_note_name(midi_note)}/1")
        return motif

    def _fallback_continuation(self, context: Sequence[str]) -> str:
        last_pitch = _last_pitch_from_tokens(context)
        if last_pitch is None:
            return self._default_seed_motif()[0]
        tonic = self.profile.key_tonic_pc if self.profile.pitch_class_counts else 0
        mode = self.profile.key_mode
        candidates = [
            last_pitch + 2,
            last_pitch - 2,
            last_pitch + 3,
            last_pitch - 3,
            _closest_midi_for_pitch_class(last_pitch, tonic),
        ]
        best = min(candidates, key=lambda midi_note: abs(midi_note - self.profile.preferred_register()))
        best = _constrain_midi(best)
        return f"{midi_to_note_name(best)}/1"

    def _fill_time(self, remaining_beats: float) -> list[str]:
        tokens: list[str] = []
        while remaining_beats > 1e-6:
            duration = min(1.0, remaining_beats)
            tokens.append(f"REST/{_format_duration(duration)}")
            remaining_beats -= duration
        return tokens

    def generate_score(
        self,
        *,
        bars: int = 8,
        beats_per_bar: float = 4.0,
        prompt: str = "",
        temperature: float = 0.9,
        top_k: int | None = 32,
        top_p: float | None = 0.9,
        seed: int | None = None,
    ) -> dict[str, object]:
        melody = self.generate_melody(
            bars=bars,
            beats_per_bar=beats_per_bar,
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        accompaniment = self._generate_accompaniment(bars=bars, beats_per_bar=beats_per_bar)
        return {
            "melody": [token.to_text() for token in melody],
            "accompaniment": accompaniment,
            "key": {
                "tonic": PITCH_CLASS_TO_NAME[self.profile.key_tonic_pc],
                "mode": self.profile.key_mode,
            },
            "register": self.profile.preferred_register(),
        }

    def _generate_accompaniment(self, *, bars: int, beats_per_bar: float) -> list[dict[str, object]]:
        tonic = self.profile.key_tonic_pc if self.profile.pitch_class_counts else 0
        mode = self.profile.key_mode
        progression = [0, 5, 3, 4] if mode == "major" else [0, 5, 3, 6]
        root_octave = 3
        accompaniment: list[dict[str, object]] = []
        for bar in range(bars):
            degree = progression[bar % len(progression)]
            root_midi = _midi_from_scale_degree(tonic, mode, degree, 12 * (root_octave + 1))
            third = _midi_from_scale_degree(tonic, mode, degree + 2, 12 * (root_octave + 1))
            fifth = _midi_from_scale_degree(tonic, mode, degree + 4, 12 * (root_octave + 1))
            bar_start = int(bar * beats_per_bar * DEFAULT_TICKS_PER_BEAT)
            bar_ticks = int(beats_per_bar * DEFAULT_TICKS_PER_BEAT)
            accompaniment.append(
                {
                    "start_tick": bar_start,
                    "notes": [
                        {"midi": _constrain_midi(root_midi - 12), "duration": bar_ticks, "velocity": 72},
                        {"midi": _constrain_midi(third - 12), "duration": bar_ticks, "velocity": 58},
                        {"midi": _constrain_midi(fifth - 12), "duration": bar_ticks, "velocity": 60},
                    ],
                }
            )
        return accompaniment

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "alpha": self.alpha,
            "seed": self.seed,
            "vocab": sorted(self.vocab),
            "unigram_counts": dict(self.unigram_counts),
            "context_counts": {
                str(order): {
                    "|".join(context): dict(counter)
                    for context, counter in contexts.items()
                }
                for order, contexts in self.context_counts.items()
            },
            "profile": {
                "pitch_class_counts": dict(self.profile.pitch_class_counts),
                "pitch_counts": dict(self.profile.pitch_counts),
                "duration_counts": {str(key): value for key, value in self.profile.duration_counts.items()},
                "interval_counts": dict(self.profile.interval_counts),
                "median_pitch": self.profile.median_pitch,
                "key_tonic_pc": self.profile.key_tonic_pc,
                "key_mode": self.profile.key_mode,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AdaptiveMusicGenerator":
        model = cls(
            order=int(payload["order"]),
            alpha=float(payload["alpha"]),
            seed=payload.get("seed") if payload.get("seed") is None else int(payload["seed"]),
        )
        model.vocab = set(payload.get("vocab", []))
        model.unigram_counts = Counter(payload.get("unigram_counts", {}))
        model.context_counts = {n: defaultdict(Counter) for n in range(1, model.order + 1)}

        context_counts = payload.get("context_counts", {})
        for order_text, contexts in context_counts.items():
            order = int(order_text)
            for context_text, counter_dict in contexts.items():
                context = tuple(context_text.split("|")) if context_text else tuple()
                model.context_counts[order][context] = Counter(counter_dict)

        profile_payload = payload.get("profile", {})
        model.profile = StyleProfile(
            pitch_class_counts=Counter({int(key): value for key, value in profile_payload.get("pitch_class_counts", {}).items()}),
            pitch_counts=Counter({int(key): value for key, value in profile_payload.get("pitch_counts", {}).items()}),
            duration_counts=Counter({float(key): value for key, value in profile_payload.get("duration_counts", {}).items()}),
            interval_counts=Counter({int(key): value for key, value in profile_payload.get("interval_counts", {}).items()}),
            median_pitch=float(profile_payload.get("median_pitch", 60.0)),
            key_tonic_pc=int(profile_payload.get("key_tonic_pc", 0)),
            key_mode=str(profile_payload.get("key_mode", "major")),
        )
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveMusicGenerator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _split_phrases(text: str) -> list[str]:
    phrases = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        phrases.append(line)
    if phrases:
        return phrases
    return [text]


def _normalize_phrase_tokens(phrase: str) -> list[str]:
    tokens = [token for token in TOKEN_SPLIT_PATTERN.split(phrase.replace("|", " ").strip()) if token]
    normalized: list[str] = []
    for token in tokens:
        if token.startswith("#"):
            continue
        if parse_token(token) is not None:
            normalized.append(token)
    return normalized


def parse_token(token: str) -> tuple[str, float] | None:
    match = NOTE_PATTERN.match(token.strip())
    if not match:
        return None
    pitch = match.group(1).upper()
    duration = float(match.group(2)) if match.group(2) else 1.0
    if pitch != REST_TOKEN:
        pitch = pitch[0].upper() + pitch[1:]
        if len(pitch) > 1 and pitch[1] == "#":
            pitch = pitch[0].upper() + "#" + pitch[2:]
        elif len(pitch) > 1 and pitch[1] == "B":
            pitch = pitch[0].upper() + "b" + pitch[2:]
    return pitch, duration


def token_to_musical(token: str) -> MusicalToken:
    parsed = parse_token(token)
    if parsed is None:
        raise ValueError(f"Invalid musical token: {token}")
    return MusicalToken(*parsed)


def note_name_to_midi(note: str) -> int:
    note = note.strip()
    if note == REST_TOKEN:
        raise ValueError("REST does not map to MIDI")
    name = note[:-1].upper()
    octave = int(note[-1])
    if len(note) >= 3 and note[-2] in {"#", "b", "B"}:
        name = note[:-1].upper()
    pitch_class = NOTE_TO_PITCH_CLASS.get(name)
    if pitch_class is None:
        raise ValueError(f"Unknown note name: {note}")
    return (octave + 1) * 12 + pitch_class


def midi_to_note_name(midi_note: int) -> str:
    midi_note = _constrain_midi(midi_note)
    octave = midi_note // 12 - 1
    pitch_class = midi_note % 12
    return f"{PITCH_CLASS_TO_NAME[pitch_class]}{octave}"


def _constrain_midi(midi_note: int) -> int:
    return max(0, min(127, int(round(midi_note))))


def _closest_midi_for_pitch_class(reference_midi: int, pitch_class: int) -> int:
    candidates = [pitch_class + 12 * octave for octave in range(0, 11)]
    return min(candidates, key=lambda candidate: abs(candidate - reference_midi))


def _midi_from_scale_degree(tonic_pc: int, mode: str, degree: int, base_midi: int) -> int:
    intervals = SCALE_PATTERNS[mode]
    scale_degree = degree % 7
    octave_shift = degree // 7
    pitch_class = (tonic_pc + intervals[scale_degree]) % 12
    reference = base_midi + octave_shift * 12
    return _closest_midi_for_pitch_class(reference, pitch_class)


def _last_pitch_from_tokens(tokens: Sequence[str]) -> int | None:
    for token in reversed(tokens):
        parsed = parse_token(token)
        if parsed is None:
            continue
        pitch_name, _ = parsed
        if pitch_name == REST_TOKEN:
            continue
        return note_name_to_midi(pitch_name)
    return None


def _format_duration(duration: float) -> str:
    if abs(duration - round(duration)) < 1e-9:
        return str(int(round(duration)))
    text = f"{duration:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def load_corpus(paths: Sequence[str | Path] | None) -> list[str]:
    if not paths:
        return list(DEFAULT_CORPUS)
    texts: list[str] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        if text.strip():
            texts.append(text)
    return texts or list(DEFAULT_CORPUS)


def build_midi(
    melody: Sequence[MusicalToken],
    accompaniment: Sequence[dict[str, object]],
    *,
    tempo_bpm: int = DEFAULT_TEMPO_BPM,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
) -> bytes:
    notes: list[MidiNote] = []
    current_tick = 0
    for token in melody:
        if token.pitch == REST_TOKEN:
            current_tick += int(round(token.duration * ticks_per_beat))
            continue
        midi_note = note_name_to_midi(token.pitch)
        duration_ticks = max(1, int(round(token.duration * ticks_per_beat)))
        notes.append(MidiNote(current_tick, current_tick + duration_ticks, midi_note, channel=0, velocity=100))
        current_tick += duration_ticks

    for bar in accompaniment:
        start_tick = int(bar["start_tick"])
        for note_info in bar["notes"]:
            notes.append(
                MidiNote(
                    start_tick,
                    start_tick + int(note_info["duration"]),
                    int(note_info["midi"]),
                    channel=1,
                    velocity=int(note_info["velocity"]),
                )
            )

    events = _midi_events_from_notes(notes)
    track_data = bytearray()
    track_data.extend(_varlen_quantity(0))
    track_data.extend(b"\xff\x51\x03")
    tempo_microseconds = int(round(60_000_000 / tempo_bpm))
    track_data.extend(tempo_microseconds.to_bytes(3, byteorder="big"))

    last_tick = 0
    for event in events:
        delta = event.tick - last_tick
        track_data.extend(_varlen_quantity(delta))
        if event.kind == "note_on":
            track_data.extend(bytes([0x90 | event.channel, event.midi_note, event.velocity]))
        else:
            track_data.extend(bytes([0x80 | event.channel, event.midi_note, 0]))
        last_tick = event.tick

    track_data.extend(_varlen_quantity(0))
    track_data.extend(b"\xff\x2f\x00")

    header = bytearray()
    header.extend(b"MThd")
    header.extend((6).to_bytes(4, byteorder="big"))
    header.extend((0).to_bytes(2, byteorder="big"))
    header.extend((1).to_bytes(2, byteorder="big"))
    header.extend(int(ticks_per_beat).to_bytes(2, byteorder="big"))

    track = bytearray()
    track.extend(b"MTrk")
    track.extend(len(track_data).to_bytes(4, byteorder="big"))
    track.extend(track_data)

    return bytes(header + track)


def _midi_events_from_notes(notes: Sequence[MidiNote]) -> list[MidiEvent]:
    events: list[MidiEvent] = []
    for note in notes:
        events.append(MidiEvent(note.start_tick, "note_on", note.channel, note.midi_note, note.velocity))
        events.append(MidiEvent(note.end_tick, "note_off", note.channel, note.midi_note, 0))
    events.sort(key=lambda event: (event.tick, 0 if event.kind == "note_off" else 1, event.channel, event.midi_note))
    return events


def _varlen_quantity(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 0x7F
    value >>= 7
    bytes_out = [buffer]
    while value:
        buffer = value & 0x7F
        bytes_out.append(0x80 | buffer)
        value >>= 7
    return bytes(reversed(bytes_out))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-powered symbolic music generator")
    parser.add_argument("--train", nargs="*", help="Text corpus files for training")
    parser.add_argument("--load-model", type=str, help="Load a saved model JSON file")
    parser.add_argument("--save-model", type=str, help="Save the trained model JSON file")
    parser.add_argument("--output", type=str, default="generated_music.mid", help="Output MIDI file")
    parser.add_argument("--bars", type=int, default=8, help="How many bars to generate")
    parser.add_argument("--beats-per-bar", type=float, default=4.0, help="Meter length in beats")
    parser.add_argument("--tempo", type=int, default=DEFAULT_TEMPO_BPM, help="Tempo in BPM")
    parser.add_argument("--prompt", type=str, default="", help="Seed motif in token format")
    parser.add_argument("--order", type=int, default=4, help="N-gram order to train")
    parser.add_argument("--alpha", type=float, default=0.7, help="Add-alpha smoothing strength")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k sampling limit")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus threshold")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--text-only", action="store_true", help="Print the generated score without writing MIDI")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.load_model:
        generator = AdaptiveMusicGenerator.load(args.load_model)
    else:
        corpus_texts = load_corpus(args.train)
        generator = AdaptiveMusicGenerator(order=args.order, alpha=args.alpha, seed=args.seed)
        generator.train(corpus_texts)
        if args.save_model:
            generator.save(args.save_model)

    score = generator.generate_score(
        bars=args.bars,
        beats_per_bar=args.beats_per_bar,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    if args.text_only:
        print(json.dumps(score, indent=2))
    else:
        midi_data = build_midi(
            [token_to_musical(token) for token in score["melody"]],
            score["accompaniment"],
            tempo_bpm=args.tempo,
        )
        Path(args.output).write_bytes(midi_data)
        print(f"Wrote MIDI file: {Path(args.output).resolve()}")
        print(f"Key: {score['key']['tonic']} {score['key']['mode']}")
        print(f"Melody tokens: {len(score['melody'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""AI-powered symbolic music generator.

This script trains a lightweight language model over musical event tokens and
uses it to generate original melodies, then exports the result as a MIDI file.

Features:
- dependency-free, standard-library only
- word/token-based music modeling with backoff n-grams
- style profiling for key, scale, rhythm, and contour
- MIDI export
- model save/load

Event format:
    C4/0.5 D4/0.5 E4/1.0 REST/0.25

Each token is a note name or REST followed by a duration in beats.
"""

import argparse
import json
import math
import random
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

TOKEN_PATTERN = re.compile(r"\w+#?\d*|REST|R|[A-Ga-g][#b]?\d*|\S+")
EVENT_PATTERN = re.compile(
    r"^(?P<pitch>[A-Ga-g][#b]?\d+|REST|R)(?:\s*[:/\-]\s*(?P<duration>\d*\.?\d+))?$"
)

START_TOKEN = "<s>"
END_TOKEN = "</s>"
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_CORPUS = [
    "C4/0.5 E4/0.5 G4/1.0 G4/0.5 A4/0.5 G4/1.0 E4/0.5 D4/0.5 C4/1.0",
    "C4/0.5 D4/0.5 E4/0.5 G4/0.5 A4/1.0 G4/0.5 E4/0.5 D4/1.0",
    "A3/0.5 C4/0.5 E4/1.0 A3/0.5 C4/0.5 E4/1.0 G3/0.5 B3/0.5 D4/1.0",
    "F3/0.5 A3/0.5 C4/1.0 G3/0.5 B3/0.5 D4/1.0 C4/0.5 E4/0.5 G4/1.0",
    "E4/0.25 G4/0.25 A4/0.5 B4/0.5 A4/0.25 G4/0.25 E4/1.0 REST/0.5",
    "D4/0.5 F4/0.5 A4/1.0 A4/0.5 G4/0.5 F4/1.0 D4/0.5 REST/0.5",
]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NOTE_TO_SEMITONE = {
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
SEMITONE_TO_NOTE = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}
SCALE_INTERVALS = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 10},
}


def tokenize_events(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text) if token.strip()]


def _parse_event_token(token: str) -> tuple[str, float]:
    normalized = token.strip()
    match = EVENT_PATTERN.match(normalized)
    if not match:
        raise ValueError(f"Invalid event token: {token}")

    pitch = match.group("pitch").upper()
    duration_text = match.group("duration")
    duration = float(duration_text) if duration_text is not None else 1.0
    if duration <= 0:
        raise ValueError(f"Duration must be positive: {token}")
    if pitch == "R":
        pitch = "REST"
    return pitch, duration


def _format_event_token(pitch: str, duration: float) -> str:
    return f"{pitch}/{duration:.4f}".rstrip("0").rstrip(".")


def _pitch_name_to_midi(pitch: str) -> int | None:
    if pitch == "REST":
        return None
    match = re.fullmatch(r"([A-G])([#B]?)(-?\d+)", pitch.upper())
    if not match:
        raise ValueError(f"Invalid pitch name: {pitch}")
    note = match.group(1)
    accidental = match.group(2)
    octave = int(match.group(3))
    name = note + accidental
    semitone = NOTE_TO_SEMITONE[name]
    return 12 * (octave + 1) + semitone


def _midi_to_pitch_name(midi_note: int) -> str:
    octave = (midi_note // 12) - 1
    note = SEMITONE_TO_NOTE[midi_note % 12]
    return f"{note}{octave}"


def _split_melodic_phrases(text: str) -> list[list[str]]:
    tokens = tokenize_events(text)
    if not tokens:
        return []
    phrases: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        current.append(token)
        if token.endswith("|"):
            phrases.append(current[:-1])
            current = []
    if current:
        phrases.append(current)
    return [phrase for phrase in phrases if phrase]


@dataclass
class StyleProfile:
    pitch_class_counts: Counter[int] = field(default_factory=Counter)
    pitch_counts: Counter[int] = field(default_factory=Counter)
    duration_counts: Counter[float] = field(default_factory=Counter)
    interval_counts: Counter[int] = field(default_factory=Counter)
    median_pitch: float = 60.0
    key_tonic_pc: int = 0
    key_mode: str = "major"

    def infer_key(self) -> tuple[int, str]:
        best_score = -float("inf")
        best_tonic = 0
        best_mode = "major"

        for tonic in range(12):
            for mode, intervals in SCALE_PATTERNS.items():
                scale = {(tonic + interval) % 12 for interval in intervals}
                in_scale = sum(self.pitch_class_counts.get(pc, 0) for pc in scale)
                tonic_bonus = self.pitch_class_counts.get(tonic, 0) * 1.3
                dominant_bonus = self.pitch_class_counts.get((tonic + 7) % 12, 0) * 0.35
                outside = sum(count for pc, count in self.pitch_class_counts.items() if pc not in scale)
                score = in_scale * 1.2 + tonic_bonus + dominant_bonus - outside * 0.2
                if score > best_score:
                    best_score = score
                    best_tonic = tonic
                    best_mode = mode

        self.key_tonic_pc = best_tonic
        self.key_mode = best_mode
        return best_tonic, best_mode

    def scale_set(self) -> set[int]:
        intervals = SCALE_PATTERNS[self.key_mode]
        return {(self.key_tonic_pc + interval) % 12 for interval in intervals}

    def common_durations(self, limit: int = 4) -> list[float]:
        return [duration for duration, _ in self.duration_counts.most_common(limit)]

    def preferred_register(self) -> int:
        if self.pitch_counts:
            return int(round(self.median_pitch))
        return 60

    def to_dict(self) -> dict[str, object]:
        return {
            "pitch_class_counts": {str(key): value for key, value in self.pitch_class_counts.items()},
            "pitch_counts": {str(key): value for key, value in self.pitch_counts.items()},
            "duration_counts": {str(key): value for key, value in self.duration_counts.items()},
            "interval_counts": {str(key): value for key, value in self.interval_counts.items()},
            "median_pitch": self.median_pitch,
            "key_tonic_pc": self.key_tonic_pc,
            "key_mode": self.key_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StyleProfile":
        profile = cls()
        profile.pitch_class_counts = Counter({int(key): value for key, value in payload.get("pitch_class_counts", {}).items()})
        profile.pitch_counts = Counter({int(key): value for key, value in payload.get("pitch_counts", {}).items()})
        profile.duration_counts = Counter({float(key): value for key, value in payload.get("duration_counts", {}).items()})
        profile.interval_counts = Counter({int(key): value for key, value in payload.get("interval_counts", {}).items()})
        profile.median_pitch = float(payload.get("median_pitch", 60.0))
        profile.key_tonic_pc = int(payload.get("key_tonic_pc", 0))
        profile.key_mode = str(payload.get("key_mode", "major"))
        return profile


@dataclass
class MusicGenerator:
    order: int = 4
    alpha: float = 0.8
    seed: int | None = None
    context_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)
    style: StyleProfile = field(default_factory=StyleProfile)
    unigram_counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be at least 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        for n in range(1, self.order + 1):
            self.context_counts.setdefault(n, defaultdict(Counter))

    def train(self, texts: Iterable[str]) -> None:
        events_seen = 0
        pitch_classes: Counter[int] = Counter()
        durations: Counter[float] = Counter()
        intervals: list[int] = []
        rests = 0
        previous_pitch: int | None = None

        for text in texts:
            for phrase in _split_melodic_phrases(text):
                padded = [START_TOKEN] * (self.order - 1) + phrase + [END_TOKEN]
                for index, token in enumerate(padded):
                    self.vocab.add(token)
                    self.unigram_counts[token] += 1
                    events_seen += 1

                    if token not in {START_TOKEN, END_TOKEN}:
                        pitch, duration = _parse_event_token(token)
                        durations[round(duration, 4)] += 1
                        if pitch == "REST":
                            rests += 1
                            previous_pitch = None
                        else:
                            midi_note = _pitch_name_to_midi(pitch)
                            assert midi_note is not None
                            pitch_classes[midi_note % 12] += 1
                            if previous_pitch is not None:
                                intervals.append(abs(midi_note - previous_pitch))
                            previous_pitch = midi_note

                    for n in range(2, self.order + 1):
                        if index < n - 1:
                            continue
                        context = tuple(padded[index - (n - 1) : index])
                        self.context_counts[n][context][token] += 1

        if not events_seen:
            raise ValueError("No training data provided")

        if pitch_classes:
            self.style.root_pitch_class = max(pitch_classes.items(), key=lambda item: item[1])[0]
            major_score = self._scale_score(pitch_classes, self.style.root_pitch_class, "major")
            minor_score = self._scale_score(pitch_classes, self.style.root_pitch_class, "minor")
            self.style.mode = "major" if major_score >= minor_score else "minor"
            self.style.scale_pitch_classes = {
                (self.style.root_pitch_class + interval) % 12
                for interval in SCALE_INTERVALS[self.style.mode]
            }
        else:
            self.style.scale_pitch_classes = SCALE_INTERVALS[self.style.mode].copy()

        if durations:
            total_duration = sum(duration * count for duration, count in durations.items())
            total_count = sum(durations.values())
            self.style.average_duration = total_duration / total_count
            self.style.duration_histogram = durations
        if intervals:
            self.style.average_interval = sum(intervals) / len(intervals)
            mean_interval = self.style.average_interval
            self.style.interval_span = math.sqrt(
                sum((interval - mean_interval) ** 2 for interval in intervals) / len(intervals)
            )
        self.style.pitch_class_histogram = pitch_classes
        self.style.rest_ratio = rests / max(events_seen - 2 * len(texts), 1)

        self.vocab.add(END_TOKEN)

    def _scale_score(self, pitch_classes: Counter[int], root: int, mode: str) -> float:
        profile = MAJOR_PROFILE if mode == "major" else MINOR_PROFILE
        scale_classes = {(root + interval) % 12 for interval in SCALE_INTERVALS[mode]}
        total = sum(pitch_classes.values()) or 1
        score = 0.0
        for pitch_class, count in pitch_classes.items():
            weight = profile[(pitch_class - root) % 12]
            if pitch_class in scale_classes:
                score += count * weight
            else:
                score -= count * (weight * 0.18)
        return score / total

    def _candidate_scores(self, context: Sequence[str]) -> dict[str, float]:
        context = tuple(context)
        scores: dict[str, float] = {}
        max_order = min(self.order, len(context) + 1)

        for n in range(max_order, 1, -1):
            window = context[-(n - 1) :] if n > 1 else ()
            bucket = self.context_counts[n].get(tuple(window))
            if not bucket:
                continue
            total = sum(bucket.values()) + self.alpha * len(self.vocab)
            weight = 1.0 + math.log1p(sum(bucket.values()))
            for token in self.vocab:
                probability = (bucket.get(token, 0) + self.alpha) / total
                scores[token] = scores.get(token, 0.0) + weight * probability

        if not scores:
            total = sum(self.unigram_counts.values()) + self.alpha * len(self.vocab)
            for token in self.vocab:
                scores[token] = (self.unigram_counts.get(token, 0) + self.alpha) / total
        else:
            total = sum(self.unigram_counts.values()) + self.alpha * len(self.vocab)
            for token in self.vocab:
                scores[token] += 0.18 * ((self.unigram_counts.get(token, 0) + self.alpha) / total)
        return scores

    def _apply_style_bias(
        self,
        token: str,
        previous_token: str | None,
        raw_score: float,
    ) -> float:
        if token in {START_TOKEN, END_TOKEN}:
            return raw_score

        pitch, duration = _parse_event_token(token)
        score = raw_score

        if pitch == "REST":
            score *= 1.0 + self.style.rest_ratio * 1.4
        else:
            midi_note = _pitch_name_to_midi(pitch)
            assert midi_note is not None
            pitch_class = midi_note % 12
            if pitch_class in self.style.scale_pitch_classes:
                score *= 1.28
            else:
                score *= 0.78

            if previous_token and previous_token not in {START_TOKEN, END_TOKEN}:
                prev_pitch, _ = _parse_event_token(previous_token)
                if prev_pitch != "REST":
                    previous_midi = _pitch_name_to_midi(prev_pitch)
                    assert previous_midi is not None
                    interval = abs(midi_note - previous_midi)
                    ideal = max(self.style.average_interval, 1.0)
                    closeness = math.exp(-abs(interval - ideal) / max(self.style.interval_span, 1.0))
                    score *= 0.82 + closeness

        duration_target = max(self.style.average_duration, 0.25)
        duration_score = math.exp(-abs(duration - duration_target) / max(duration_target * 1.6, 0.25))
        score *= 0.82 + duration_score
        return score

    def sample_next(
        self,
        context: Sequence[str],
        *,
        rng: random.Random,
        temperature: float = 0.95,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
    ) -> str:
        raw_scores = self._candidate_scores(context)
        previous_token = context[-1] if context else None
        candidates = []
        recent_counter = Counter(context[-8:])

        for token, raw_score in raw_scores.items():
            if token == START_TOKEN:
                continue
            score = self._apply_style_bias(token, previous_token, raw_score)
            if token in recent_counter:
                score /= 1 + (recent_counter[token] * 0.7)
            candidates.append((token, max(score, 1e-12)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None and top_k > 0:
            candidates = candidates[:top_k]

        if top_p is not None and 0 < top_p < 1:
            total = sum(score for _, score in candidates)
            cumulative = 0.0
            pruned: list[tuple[str, float]] = []
            for token, score in candidates:
                pruned.append((token, score))
                cumulative += score / total if total else 0.0
                if cumulative >= top_p:
                    break
            if pruned:
                candidates = pruned

        if not candidates:
            return END_TOKEN

        temperature = max(temperature, 1e-6)
        scaled = [(token, score ** (1.0 / temperature)) for token, score in candidates]
        total = sum(score for _, score in scaled)
        threshold = rng.random() * total
        cumulative = 0.0
        for token, score in scaled:
            cumulative += score
            if cumulative >= threshold:
                return token
        return scaled[-1][0]

    def generate_tokens(
        self,
        prompt: str = "",
        *,
        bars: int = 16,
        beats_per_bar: float = 4.0,
        temperature: float = 0.95,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        seed: int | None = None,
    ) -> list[str]:
        if bars < 1:
            raise ValueError("bars must be at least 1")

        rng = random.Random(self.seed if seed is None else seed)
        prompt_tokens = tokenize_events(prompt)
        prompt_tokens = [token for token in prompt_tokens if token not in {"|"}]
        generated = list(prompt_tokens)
        context = ([START_TOKEN] * max(0, self.order - 1 - len(prompt_tokens))) + prompt_tokens[-(self.order - 1) :]
        total_beats = 0.0
        for token in prompt_tokens:
            if token in {START_TOKEN, END_TOKEN}:
                continue
            try:
                _, duration = _parse_event_token(token)
                total_beats += duration
            except ValueError:
                continue

        target_beats = bars * beats_per_bar
        safety_limit = max(int(target_beats * 4), bars * 32)

        for _ in range(safety_limit):
            if total_beats >= target_beats and len(generated) >= max(len(prompt_tokens) + 4, 12):
                break
            next_token = self.sample_next(
                context,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if next_token == END_TOKEN:
                if total_beats >= target_beats * 0.75:
                    break
                continue
            generated.append(next_token)
            context = (context + [next_token])[-(self.order - 1) :]
            try:
                _, duration = _parse_event_token(next_token)
                total_beats += duration
            except ValueError:
                continue

        return generated

    def generate_melody(
        self,
        prompt: str = "",
        *,
        bars: int = 16,
        beats_per_bar: float = 4.0,
        temperature: float = 0.95,
        top_k: int | None = 40,
        top_p: float | None = 0.9,
        seed: int | None = None,
    ) -> str:
        return detokenize(self.generate_tokens(
            prompt,
            bars=bars,
            beats_per_bar=beats_per_bar,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        ))

    def generate_midi(self, tokens: Sequence[str], *, tempo_bpm: int = 120, ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT) -> bytes:
        events: list[tuple[int, int, bytes]] = []
        events.append((0, 0, _meta_tempo_event(tempo_bpm)))

        current_tick = 0
        active_pitch: int | None = None
        for token in tokens:
            if token in {START_TOKEN, END_TOKEN}:
                continue
            pitch_name, duration = _parse_event_token(token)
            duration_ticks = max(int(round(duration * ticks_per_beat)), 1)
            if active_pitch is not None:
                events.append((current_tick, 0, _note_off_event(active_pitch, channel=0)))
                active_pitch = None
            if pitch_name == "REST":
                current_tick += duration_ticks
                continue
            midi_note = _pitch_name_to_midi(pitch_name)
            assert midi_note is not None
            velocity = _velocity_from_pitch(midi_note)
            events.append((current_tick, 1, _note_on_event(midi_note, velocity=velocity, channel=0)))
            current_tick += duration_ticks
            events.append((current_tick, 2, _note_off_event(midi_note, channel=0)))

        events.append((current_tick, 3, _end_of_track_event()))
        events.sort(key=lambda item: (item[0], item[1]))
        track_data = _encode_midi_track(events)
        header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
        track_chunk = b"MTrk" + struct.pack(">I", len(track_data)) + track_data
        return header + track_chunk

    def save(self, path: str | Path) -> None:
        payload = {
            "order": self.order,
            "alpha": self.alpha,
            "seed": self.seed,
            "vocab": sorted(self.vocab),
            "unigram_counts": dict(self.unigram_counts),
            "context_counts": {
                str(order): {
                    "|".join(context): dict(counter)
                    for context, counter in contexts.items()
                }
                for order, contexts in self.context_counts.items()
            },
            "style": self.style.to_dict(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MusicGenerator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            order=int(payload["order"]),
            alpha=float(payload["alpha"]),
            seed=payload.get("seed") if payload.get("seed") is None else int(payload["seed"]),
        )
        model.vocab = set(payload.get("vocab", []))
        model.unigram_counts = Counter(payload.get("unigram_counts", {}))
        model.context_counts = {n: defaultdict(Counter) for n in range(1, model.order + 1)}
        for order_text, contexts in payload.get("context_counts", {}).items():
            order = int(order_text)
            for context_text, counter_dict in contexts.items():
                context = tuple(context_text.split("|")) if context_text else tuple()
                model.context_counts[order][context] = Counter(counter_dict)
        model.style = StyleProfile.from_dict(payload.get("style", {}))
        return model


def _velocity_from_pitch(midi_note: int) -> int:
    base = 72 + int(10 * math.sin(midi_note / 7.0))
    return max(40, min(110, base))


def _varlen(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7
    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def _note_on_event(note: int, *, velocity: int, channel: int = 0) -> bytes:
    return bytes([0x90 | (channel & 0x0F), note & 0x7F, velocity & 0x7F])


def _note_off_event(note: int, *, channel: int = 0) -> bytes:
    return bytes([0x80 | (channel & 0x0F), note & 0x7F, 0x00])


def _meta_tempo_event(tempo_bpm: int) -> bytes:
    tempo_bpm = max(20, min(300, tempo_bpm))
    microseconds_per_beat = int(round(60_000_000 / tempo_bpm))
    return b"\xFF\x51\x03" + microseconds_per_beat.to_bytes(3, byteorder="big")


def _end_of_track_event() -> bytes:
    return b"\xFF\x2F\x00"


def _encode_midi_track(events: Sequence[tuple[int, int, bytes]]) -> bytes:
    last_tick = 0
    chunks = bytearray()
    for tick, _, payload in events:
        delta = max(tick - last_tick, 0)
        chunks.extend(_varlen(delta))
        chunks.extend(payload)
        last_tick = tick
    return bytes(chunks)


def load_texts_from_paths(paths: Sequence[str | Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        if text.strip():
            texts.append(text)
    return texts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-powered symbolic music generator")
    parser.add_argument("--train", nargs="*", help="Text files containing event-token music examples")
    parser.add_argument("--load-model", type=str, help="Load a saved music model JSON file")
    parser.add_argument("--save-model", type=str, help="Save the trained model JSON file")
    parser.add_argument("--midi", type=str, default="generated_music.mid", help="Output MIDI file path")
    parser.add_argument("--prompt", type=str, default="", help="Prompt event tokens to seed generation")
    parser.add_argument("--bars", type=int, default=16, help="Target number of bars to generate")
    parser.add_argument("--tempo", type=int, default=120, help="Tempo in beats per minute")
    parser.add_argument("--order", type=int, default=4, help="N-gram order for training")
    parser.add_argument("--alpha", type=float, default=0.8, help="Add-alpha smoothing strength")
    parser.add_argument("--temperature", type=float, default=0.95, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling limit")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling threshold")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--text-output", action="store_true", help="Print the symbolic event sequence")
    parser.add_argument("--perplexity", type=str, help="Evaluate perplexity on a text file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.load_model:
        model = MusicGenerator.load(args.load_model)
    else:
        train_paths = args.train or []
        model = MusicGenerator(order=args.order, alpha=args.alpha, seed=args.seed)
        if train_paths:
            texts = load_texts_from_paths(train_paths)
        else:
            texts = DEFAULT_CORPUS
        model.train(texts)
        if args.save_model:
            model.save(args.save_model)

    if args.perplexity:
        target = Path(args.perplexity).read_text(encoding="utf-8")
        print(f"Perplexity: {model_perplexity(model, target):.4f}")

    tokens = model.generate_tokens(
        args.prompt,
        bars=args.bars,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    midi_bytes = model.generate_midi(tokens, tempo_bpm=args.tempo)
    output_path = Path(args.midi)
    output_path.write_bytes(midi_bytes)

    print(f"Wrote MIDI: {output_path.resolve()}")
    print(f"Style: root_pc={model.style.root_pitch_class}, mode={model.style.mode}, avg_dur={model.style.average_duration:.2f}")
    if args.text_output:
        print(detokenize(tokens))
    return 0


def model_perplexity(model: MusicGenerator, text: str) -> float:
    tokens = tokenize_events(text)
    if not tokens:
        return float("inf")

    padded = [START_TOKEN] * (model.order - 1) + tokens + [END_TOKEN]
    log_probability = 0.0
    steps = 0
    for index in range(model.order - 1, len(padded)):
        context = padded[index - (model.order - 1) : index]
        next_token = padded[index]
        scores = model._candidate_scores(context)
        total = sum(scores.values()) or 1.0
        probability = max(scores.get(next_token, model.alpha / max(total, 1.0)) / total, 1e-12)
        log_probability += math.log(probability)
        steps += 1
    return math.exp(-log_probability / max(steps, 1))


if __name__ == "__main__":
    raise SystemExit(main())
