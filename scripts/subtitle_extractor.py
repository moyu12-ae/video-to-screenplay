#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/subtitle_extractor.py - Adaptive Multi-Source Subtitle Extraction Engine

Implements a 3-tier hierarchy:
  Tier 1: External companion files (.srt, .ass, .vtt, .json)
  Tier 2: Embedded soft subtitle streams in MKV/MP4/TS containers via FFprobe & FFmpeg
  Tier 3: Hardcoded burnt-in subtitles fallback instruction (Wangyan OCR MCP)
"""

import sys
import os
import re
import json
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def require_ffprobe() -> bool:
    """Report whether the FFmpeg tooling needed by Tier 2 is reachable on PATH."""
    missing = [b for b in ("ffprobe", "ffmpeg") if shutil.which(b) is None]
    if missing:
        sys.stderr.write(
            f"[WARN] Missing {', '.join(missing)} on PATH; Tier 2 (embedded soft subtitles) "
            "cannot be attempted. Please install FFmpeg (macOS: 'brew install ffmpeg', Windows: 'winget install Gyan.FFmpeg' or scoop/choco).\n"
        )
        return False
    return True


def parse_timecode_to_ms(tc: str) -> int:
    """Convert SRT/VTT/ASS timecode string to milliseconds."""
    tc = tc.strip().replace(',', '.')
    parts = tc.split(':')
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s_parts = parts[2].split('.')
        s = int(s_parts[0])
        ms_str = s_parts[1] if len(s_parts) > 1 else '0'
        if len(ms_str) == 1:
            ms = int(ms_str) * 100
        elif len(ms_str) == 2:
            ms = int(ms_str) * 10
        else:
            ms = int(ms_str[:3])
        return h * 3600000 + m * 60000 + s * 1000 + ms
    elif len(parts) == 2:
        m = int(parts[0])
        s_parts = parts[1].split('.')
        s = int(s_parts[0])
        ms = int(s_parts[1][:3]) if len(s_parts) > 1 else 0
        return m * 60000 + s * 1000 + ms
    return 0


def ms_to_timecode(ms: int) -> str:
    """Format milliseconds to HH:MM:SS.mmm."""
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    rem_ms = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{rem_ms:03d}"


def extract_speaker_from_text(raw_text: str) -> Tuple[Optional[str], str]:
    """
    Extract speaker name prefix if present in text and return (speaker, clean_text).
    Handles patterns like:
      - 【罗温】你好 -> ("罗温", "你好")
      - [Rowan] Hello -> ("Rowan", "Hello")
      - 罗温：你好 -> ("罗温", "你好")
      - 罗温: 你好 -> ("罗温", "你好")
      - (旁白) 很久很久以前 -> ("旁白", "很久很久以前")
    """
    text = raw_text.strip()
    if not text:
        return None, ""

    # Bracketed speaker prefix: 【...】 or [...] or （...） or (...)
    m_bracket = re.match(r'^([【\[（\(])([^】\]）\)]+)([】\]）\)])\s*(.*)$', text)
    if m_bracket:
        candidate = m_bracket.group(2).strip()
        rest = m_bracket.group(4).strip()
        # Avoid treating sound effects or emotion cues as speaker name if rest is empty
        non_speaker_cues = {"叹气", "哭声", "笑声", "掌声", "音乐", "枪声", "爆炸", "脚步声", "深呼吸", "sigh", "laughter", "music"}
        if candidate.lower() not in non_speaker_cues and len(candidate) <= 20:
            if rest:
                return candidate, rest
            # If line is solely (旁白) or (解说), treat as speaker tag for empty dialogue
            if candidate in {"旁白", "解说", "画外音", "Narrator", "Voiceover"}:
                return candidate, ""

    # Colon speaker prefix: Name：text or Name: text
    m_colon = re.match(r'^([^：:\s\d\(\)\[\]【】]{1,16})\s*[：:]\s*(.+)$', text)
    if m_colon:
        candidate = m_colon.group(1).strip()
        rest = m_colon.group(2).strip()
        # Ignore timestamps or urls or obvious false positives
        if not re.match(r'^(http|https|ftp|\d+)$', candidate, re.IGNORECASE):
            return candidate, rest

    return None, text


def clean_ass_text(text: str) -> str:
    """Remove ASS formatting tags, positioning, overrides, and style escapes."""
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'\\[Nn]', ' ', text)
    text = re.sub(r'\\[hH]', ' ', text)
    return text.strip()


def parse_srt_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse SRT/VTT file into structured dialogue items."""
    entries = []
    if not os.path.isfile(file_path):
        return entries

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 2:
            time_line_idx = 1 if lines[0].isdigit() else 0
            if '-->' in lines[time_line_idx]:
                tc_match = re.search(r'([\d:,.]+)\s*-->\s*([\d:,.]+)', lines[time_line_idx])
                if tc_match:
                    start_ms = parse_timecode_to_ms(tc_match.group(1))
                    end_ms = parse_timecode_to_ms(tc_match.group(2))
                    text_lines = lines[time_line_idx + 1:]
                    clean_text = ' '.join(text_lines)
                    clean_text = re.sub(r'<[^>]+>', '', clean_text).strip()
                    if clean_text:
                        spk, clean_dialogue = extract_speaker_from_text(clean_text)
                        final_text = clean_dialogue if clean_dialogue else clean_text
                        entries.append({
                            "index": len(entries) + 1,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "start_timecode": ms_to_timecode(start_ms),
                            "end_timecode": ms_to_timecode(end_ms),
                            "text": final_text,
                            "speaker": spk
                        })
    return entries


def parse_ass_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse ASS/SSA dialogue events into structured dialogue items."""
    entries = []
    if not os.path.isfile(file_path):
        return entries

    in_events = False
    format_indices = {}

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[Events]'):
                in_events = True
                continue
            if in_events:
                if line.startswith('Format:'):
                    fields = [x.strip().lower() for x in line[7:].split(',')]
                    format_indices = {field: i for i, field in enumerate(fields)}
                elif line.startswith('Dialogue:'):
                    content = line[9:].strip()
                    parts = content.split(',', len(format_indices) - 1) if format_indices else content.split(',', 9)
                    start_idx = format_indices.get('start', 1)
                    end_idx = format_indices.get('end', 2)
                    text_idx = format_indices.get('text', len(parts) - 1)
                    name_idx = format_indices.get('name', None)
                    style_idx = format_indices.get('style', None)

                    if len(parts) > max(start_idx, end_idx, text_idx):
                        start_ms = parse_timecode_to_ms(parts[start_idx])
                        end_ms = parse_timecode_to_ms(parts[end_idx])
                        raw_text = parts[text_idx]
                        clean = clean_ass_text(raw_text)
                        if clean:
                            # 1. First attempt: check Name field
                            spk = None
                            if name_idx is not None and len(parts) > name_idx:
                                cand_name = parts[name_idx].strip()
                                if cand_name and cand_name.lower() not in {"default", "normal", "*default"}:
                                    spk = cand_name

                            # 2. Second attempt: extract from text prefix
                            extracted_spk, clean_dialogue = extract_speaker_from_text(clean)
                            if extracted_spk:
                                spk = extracted_spk
                                clean = clean_dialogue if clean_dialogue else clean

                            # 3. Third attempt: check Style field if looks like character name
                            if not spk and style_idx is not None and len(parts) > style_idx:
                                cand_style = parts[style_idx].strip()
                                if cand_style and cand_style.lower() not in {"default", "title", "staff", "song", "ed", "op", "note", "top"}:
                                    # If style is specific (e.g., "Lala", "Rowan")
                                    if len(cand_style) <= 15 and not any(c in cand_style for c in "0123456789"):
                                        spk = cand_style

                            entries.append({
                                "index": len(entries) + 1,
                                "start_ms": start_ms,
                                "end_ms": end_ms,
                                "start_timecode": ms_to_timecode(start_ms),
                                "end_timecode": ms_to_timecode(end_ms),
                                "text": clean,
                                "speaker": spk
                            })
    entries.sort(key=lambda x: x["start_ms"])
    for i, e in enumerate(entries):
        e["index"] = i + 1
    return entries


class SubtitleExtractor:
    """Adaptive Multi-Source Subtitle Extractor."""

    def __init__(self, video_path: str, external_sub: Optional[str] = None, lang_pref: str = "chi,zho,chs,cht,zh"):
        self.video_path = os.path.abspath(video_path)
        self.external_sub = os.path.abspath(external_sub) if external_sub else None
        self.lang_prefs = [l.strip().lower() for l in lang_pref.split(',') if l.strip()]

    def check_tier1_external(self) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """Tier 1: Check user-specified or adjacent companion subtitle files."""
        candidates: List[str] = []
        if self.external_sub and os.path.isfile(self.external_sub):
            candidates.append(self.external_sub)

        video_dir = os.path.dirname(self.video_path)
        video_stem = os.path.splitext(os.path.basename(self.video_path))[0]

        try:
            for entry_name in os.listdir(video_dir):
                full_cand = os.path.join(video_dir, entry_name)
                if not os.path.isfile(full_cand):
                    continue
                cand_stem, cand_ext = os.path.splitext(entry_name)
                if cand_ext.lower() in ['.srt', '.ass', '.ssa', '.vtt']:
                    if cand_stem == video_stem or cand_stem.startswith(f"{video_stem}."):
                        if full_cand not in candidates:
                            candidates.append(full_cand)
        except OSError:
            pass

        for cand in candidates:
            ext = os.path.splitext(cand)[1].lower()
            if ext in ['.srt', '.vtt']:
                entries = parse_srt_file(cand)
                if entries:
                    return (f"TIER_1_EXTERNAL: {os.path.basename(cand)}", entries)
            elif ext in ['.ass', '.ssa']:
                entries = parse_ass_file(cand)
                if entries:
                    return (f"TIER_1_EXTERNAL: {os.path.basename(cand)}", entries)

        return None

    def check_tier2_embedded(self) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """Tier 2: Inspect video container for embedded soft subtitle streams using ffprobe."""
        if not require_ffprobe():
            return None
        try:
            res = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", self.video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
                shell=False
            )
            if res.returncode != 0:
                return None

            data = json.loads(res.stdout)
            sub_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "subtitle"]
            if not sub_streams:
                return None

            selected_stream = None
            for pref in self.lang_prefs:
                for s in sub_streams:
                    tags = s.get("tags", {})
                    lang = str(tags.get("language", "")).lower()
                    title = str(tags.get("title", "")).lower()
                    if pref in lang or pref in title:
                        selected_stream = s
                        break
                if selected_stream:
                    break

            if not selected_stream:
                selected_stream = sub_streams[0]

            stream_idx = int(selected_stream["index"])
            codec_name = str(selected_stream.get("codec_name", "subrip"))
            track_title = str(selected_stream.get("tags", {}).get("title", f"Stream #{stream_idx}"))
            track_lang = str(selected_stream.get("tags", {}).get("language", "und"))

            out_ext = ".ass" if "ass" in codec_name else ".srt"
            temp_sub = os.path.join(tempfile.gettempdir(), f"extracted_stream_{stream_idx}{out_ext}")

            extract_res = subprocess.run(
                ["ffmpeg", "-y", "-i", self.video_path, "-map", f"0:{stream_idx}", temp_sub],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=25,
                shell=False
            )
            if extract_res.returncode == 0 and os.path.isfile(temp_sub):
                if out_ext == ".ass":
                    entries = parse_ass_file(temp_sub)
                else:
                    entries = parse_srt_file(temp_sub)
                if entries:
                    info = f"TIER_2_EMBEDDED: Stream #{stream_idx} ({codec_name}, lang={track_lang}, title='{track_title}')"
                    return (info, entries)

        except Exception as e:
            sys.stderr.write(f"[WARN] Tier 2 extraction failed: {e}\n")

        return None

    def extract(self) -> Dict[str, Any]:
        """Execute the three-tier extraction hierarchy."""
        sys.stderr.write("[1/3] Checking Tier 1 (External Subtitles)...\n")
        tier1_res = self.check_tier1_external()
        if tier1_res:
            source, items = tier1_res
            sys.stderr.write(f" -> Found external subtitle: {source} ({len(items)} lines)\n")
            return {
                "source_tier": "TIER_1_EXTERNAL",
                "source_detail": source,
                "video_path": self.video_path,
                "subtitle_count": len(items),
                "items": items
            }

        sys.stderr.write("[2/3] Checking Tier 2 (Embedded Container Subtitle Streams)...\n")
        tier2_res = self.check_tier2_embedded()
        if tier2_res:
            source, items = tier2_res
            sys.stderr.write(f" -> Extracted embedded stream: {source} ({len(items)} lines)\n")
            return {
                "source_tier": "TIER_2_EMBEDDED",
                "source_detail": source,
                "video_path": self.video_path,
                "subtitle_count": len(items),
                "items": items
            }

        sys.stderr.write("[3/3] Falling back to Tier 3 (Hardcoded Burnt-in Subtitle OCR via Wangyan OCR)...\n")
        return {
            "source_tier": "TIER_3_HARDCODED_OCR",
            "source_detail": "Wangyan OCR MCP required (http://127.0.0.1:8901)",
            "video_path": self.video_path,
            "status": "NEEDS_OCR",
            "instruction": "Invoke mcp__wangyan-ocr__wangyan_ocr_api POST /import -> POST /predet -> POST /pipeline -> POST /export."
        }


def main():
    parser = argparse.ArgumentParser(description="Adaptive Multi-Source Subtitle Extractor")
    parser.add_argument("video", nargs="?", default=None, help="Path to video file (.mp4, .mkv, etc.)")
    parser.add_argument("--subtitles", "-s", help="Optional path to external subtitle file")
    parser.add_argument("--lang", "-l", default="chi,zho,chs,cht,zh", help="Language priority list (comma-separated)")
    parser.add_argument("--workspace", "-w", default=None, help="Optional workspace root directory")
    parser.add_argument("--require-subtitles", action="store_true", help="Fail fast with error if no subtitles found")

    args = parser.parse_args()

    ws = os.path.abspath(args.workspace) if args.workspace else None

    # Resolve video path
    video_input = args.video
    if not video_input and ws:
        mat_dir = os.path.join(ws, "materials")
        if os.path.isdir(mat_dir):
            candidates = sorted(
                f for f in os.listdir(mat_dir) if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))
            )
            if len(candidates) > 1:
                sys.stderr.write(
                    f"[WARN] materials/ holds {len(candidates)} videos; auto-discovery picked "
                    f"'{candidates[0]}'. Pass the video explicitly to process another one.\n"
                )
            for f in candidates:
                video_input = os.path.join(mat_dir, f)
                break

    if not video_input:
        sys.stderr.write("Error: video file must be provided either as argument or inside workspace/materials/\n")
        sys.exit(1)

    # Resolve external subtitle path if not explicitly provided
    external_sub = args.subtitles
    if not external_sub and ws:
        mat_dir = os.path.join(ws, "materials")
        if os.path.isdir(mat_dir):
            for f in sorted(os.listdir(mat_dir)):
                if f.lower().endswith((".srt", ".ass", ".vtt")):
                    external_sub = os.path.join(mat_dir, f)
                    break

    extractor = SubtitleExtractor(video_input, external_sub, args.lang)
    res = extractor.extract()

    # If --require-subtitles is passed and subtitle_count == 0, fail fast!
    if args.require_subtitles and res.get("subtitle_count", 0) == 0:
        sys.stderr.write(
            "[FATAL] Pure visual video without subtitles is rejected. Screenplay generation requires dialogue.\n"
            "Please provide an external subtitle (.srt/.ass) in materials/ or use a video with embedded subtitles.\n"
        )
        sys.exit(5)

    # Output pure JSON to stdout (standard UNIX filter pattern)
    sys.stdout.write(json.dumps(res, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
