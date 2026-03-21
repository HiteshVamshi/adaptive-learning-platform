"""
CBSE Class X Science learn catalog built from syllabus + NCERT textbook index.

- NCERT PDFs: from `cbse_class10_science_sources.textbook_section_seed` (jesc1NN.pdf per chapter).
- Optional YouTube embeds: same env keys as math (`YOUTUBE_API_KEY` / `ADAPTIVE_LEARNING_YOUTUBE_API_KEY`).
  Without a key, article rows link to YouTube search for each chapter.

Run `python scripts/build_learn_index.py --subject science` after edits.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from adaptive_learning.config import youtube_api_key
from adaptive_learning.data.cbse_class10_science_sources import (
    SCIENCE_TEXTBOOK_ROWS,
    textbook_section_seed,
)
from adaptive_learning.data.schemas import TextbookSectionRecord
from adaptive_learning.learn.schemas import LearnResourceRecord

_SUBJECT = "science"
_NCERT_LICENSE = "NCERT (see ncert.nic.in terms)"
_YT_LICENSE = "YouTube / uploader terms vary"


def _pdf_title(rec: TextbookSectionRecord) -> str:
    return f"NCERT Class X Science — {rec.chapter_name} (PDF)"


def _unique_ncert_sections_by_pdf() -> list[TextbookSectionRecord]:
    seen: set[str] = set()
    out: list[TextbookSectionRecord] = []
    for rec in textbook_section_seed():
        url = rec.chapter_pdf_url.strip()
        if url in seen:
            continue
        seen.add(url)
        out.append(rec)
    out.sort(key=lambda r: (r.chapter_number, r.chapter_id))
    return out


def _chapter_order_and_name() -> dict[str, tuple[int, str]]:
    """chapter_id -> (NCERT chapter_number, display name)."""
    return {
        chapter_id: (chapter_number, chapter_name)
        for chapter_number, chapter_id, chapter_name, _start, _lines in SCIENCE_TEXTBOOK_ROWS
    }


def _youtube_search_embed_url(query: str, api_key: str) -> str | None:
    params = urllib.parse.urlencode(
        {
            "part": "id",
            "q": query,
            "type": "video",
            "maxResults": "1",
            "key": api_key,
            "safeSearch": "strict",
            "videoEmbeddable": "true",
        }
    )
    url = f"https://www.googleapis.com/youtube/v3/search?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "adaptive-learning-platform/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    items = data.get("items") or []
    if not items:
        return None
    vid = (items[0].get("id") or {}).get("videoId")
    if not vid:
        return None
    return f"https://www.youtube.com/embed/{vid}"


def _youtube_search_fallback_url(query: str) -> str:
    q = urllib.parse.quote_plus(query)
    return f"https://www.youtube.com/results?search_query={q}"


def _video_query(chapter_name: str) -> str:
    return f"CBSE class 10 {chapter_name} science NCERT explained"


def science_learn_resource_seed() -> list[LearnResourceRecord]:
    meta = _chapter_order_and_name()
    rows: list[LearnResourceRecord] = []
    api_key = youtube_api_key()
    pdf_seq: dict[str, int] = {}

    for rec in _unique_ncert_sections_by_pdf():
        cid = rec.chapter_id
        ch_meta = meta.get(cid)
        if ch_meta is None:
            continue
        order_base, ch_name = ch_meta
        slug = rec.chapter_pdf_url.rstrip("/").rsplit("/", 1)[-1].replace(".pdf", "")
        pdf_seq[cid] = pdf_seq.get(cid, 0) + 1
        pdf_order = order_base * 100 + pdf_seq[cid]

        rows.append(
            LearnResourceRecord(
                resource_id=f"lr_sci_ncert_{cid}_{slug}",
                subject=_SUBJECT,
                concept_id="",
                chapter_id=cid,
                chapter_name=ch_name,
                concept_name="",
                resource_type="slides",
                title=_pdf_title(rec),
                url=rec.chapter_pdf_url,
                source="ncert_pdf",
                duration_min=None,
                difficulty="standard",
                order_index=pdf_order,
                license=_NCERT_LICENSE,
            )
        )

    for chapter_number, chapter_id, chapter_name, _page, _lines in SCIENCE_TEXTBOOK_ROWS:
        query = _video_query(chapter_name)
        order_vid = chapter_number * 100 + 80
        embed = _youtube_search_embed_url(query, api_key) if api_key else None
        if embed:
            rows.append(
                LearnResourceRecord(
                    resource_id=f"lr_sci_vid_{chapter_id}",
                    subject=_SUBJECT,
                    concept_id="",
                    chapter_id=chapter_id,
                    chapter_name=chapter_name,
                    concept_name="",
                    resource_type="video",
                    title=f"{chapter_name} — video (YouTube search pick)",
                    url=embed,
                    source="youtube_api_search",
                    duration_min=None,
                    difficulty="intro",
                    order_index=order_vid,
                    license=_YT_LICENSE,
                )
            )
        else:
            rows.append(
                LearnResourceRecord(
                    resource_id=f"lr_sci_vidsearch_{chapter_id}",
                    subject=_SUBJECT,
                    concept_id="",
                    chapter_id=chapter_id,
                    chapter_name=chapter_name,
                    concept_name="",
                    resource_type="article",
                    title=f"YouTube search: {chapter_name} (Class 10 Science)",
                    url=_youtube_search_fallback_url(query),
                    source="youtube_search_link",
                    duration_min=None,
                    difficulty="intro",
                    order_index=order_vid,
                    license=_YT_LICENSE,
                )
            )

    rows.sort(key=lambda r: (r.chapter_id, r.order_index, r.resource_id))
    return rows
