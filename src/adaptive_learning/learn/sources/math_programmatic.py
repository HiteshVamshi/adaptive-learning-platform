"""
CBSE Class X Mathematics learn catalog built from syllabus + NCERT textbook index.

- NCERT PDFs: deduced from `cbse_class10_textbook_index` (official chapter_number → jemh1NN.pdf).
- Optional YouTube embeds: if YOUTUBE_API_KEY or ADAPTIVE_LEARNING_YOUTUBE_API_KEY is set, one
  search per syllabus chapter at index build time (YouTube Data API v3). Otherwise an article
  row links to a YouTube search results URL for the same query.

Run `python scripts/build_learn_index.py` after changing env or syllabus data.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from adaptive_learning.config import youtube_api_key
from adaptive_learning.data.cbse_class10_syllabus import CHAPTER_ROWS
from adaptive_learning.data.cbse_class10_textbook_index import textbook_section_seed
from adaptive_learning.data.schemas import TextbookSectionRecord
from adaptive_learning.learn.schemas import LearnResourceRecord

_SUBJECT = "math"
_NCERT_LICENSE = "NCERT (see ncert.nic.in terms)"
_YT_LICENSE = "YouTube / uploader terms vary"


def _pdf_title(rec: TextbookSectionRecord) -> str:
    if rec.chapter_id == "ch_trigonometry" and rec.chapter_number == 9:
        return "NCERT Class X Mathematics — Some Applications of Trigonometry (PDF)"
    return f"NCERT Class X Mathematics — {rec.chapter_name} (PDF)"


def _unique_ncert_sections_by_pdf() -> list[TextbookSectionRecord]:
    """One representative section row per distinct NCERT chapter PDF (URL)."""
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
    """chapter_id -> (syllabus order_index, display chapter_name)."""
    return {
        cid: (order_idx, name)
        for cid, name, _u, _m, _p, order_idx, *_rest in CHAPTER_ROWS
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


def _video_query(chapter_id: str, chapter_name: str) -> str:
    return f"CBSE class 10 {chapter_name} mathematics NCERT explained"


def math_learn_resource_seed() -> list[LearnResourceRecord]:
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
                resource_id=f"lr_math_ncert_{cid}_{slug}",
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

    for chapter_id, chapter_name, _u, _m, _p, order_idx, *_rest in CHAPTER_ROWS:
        query = _video_query(chapter_id, chapter_name)
        order_vid = order_idx * 100 + 80
        embed = _youtube_search_embed_url(query, api_key) if api_key else None
        if embed:
            rows.append(
                LearnResourceRecord(
                    resource_id=f"lr_math_vid_{chapter_id}",
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
                    resource_id=f"lr_math_vidsearch_{chapter_id}",
                    subject=_SUBJECT,
                    concept_id="",
                    chapter_id=chapter_id,
                    chapter_name=chapter_name,
                    concept_name="",
                    resource_type="article",
                    title=f"YouTube search: {chapter_name} (Class 10 Maths)",
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
