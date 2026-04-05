"""
gong_client.py
--------------
Lightweight Gong API client for fetching recent calls and transcripts.
Uses Basic Auth (access_key:access_key_secret).
"""

import os
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

GONG_API_BASE = "https://us-11211.api.gong.io/v2"


class GongClient:
    def __init__(self):
        self.access_key = os.getenv("GONG_ACCESS_KEY", "")
        self.access_key_secret = os.getenv("GONG_ACCESS_KEY_SECRET", "")
        self.base_url = os.getenv("GONG_BASE_URL", GONG_API_BASE).rstrip("/")
        # Ensure we're hitting the API endpoint, not the website
        if "api.gong.io" not in self.base_url:
            self.base_url = GONG_API_BASE
        # Ensure /v2 is included in the base URL
        if not self.base_url.endswith("/v2"):
            self.base_url = self.base_url.rstrip("/") + "/v2"

    @property
    def is_configured(self):
        return bool(self.access_key and self.access_key_secret)

    def _auth(self):
        return (self.access_key, self.access_key_secret)

    def get_recent_calls(self, days=30, limit=20):
        """Fetch recent calls from the last N days."""
        if not self.is_configured:
            return []

        from_dt = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_dt = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            resp = requests.post(
                f"{self.base_url}/calls/extensive",
                auth=self._auth(),
                json={
                    "filter": {
                        "fromDateTime": from_dt,
                        "toDateTime": to_dt,
                    },
                    "contentSelector": {
                        "exposedFields": {
                            "content": {
                                "topics": True,
                                "trackers": True,
                            },
                            "collaboration": {
                                "publicComments": True,
                            },
                            "parties": True,
                        }
                    }
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            calls = data.get("calls", [])
            logger.info(f"Gong: fetched {len(calls)} recent calls")
            return calls[:limit]
        except Exception as e:
            logger.error(f"Gong API error (calls): {e}")
            return []

    def get_transcripts(self, call_ids):
        """Fetch FULL transcripts for given call IDs."""
        if not self.is_configured or not call_ids:
            return {}

        try:
            resp = requests.post(
                f"{self.base_url}/calls/transcript",
                auth=self._auth(),
                json={"filter": {"callIds": call_ids[:5]}},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            transcripts = {}
            for ct in data.get("callTranscripts", []):
                call_id = ct.get("callId", "")
                sentences = ct.get("transcript", [])
                # Build readable transcript — FULL, not truncated
                lines = []
                for s in sentences:
                    speaker = s.get("speakerName", s.get("speakerId", "Unknown"))
                    text = " ".join(sent.get("text", "") for sent in s.get("sentences", []))
                    if text.strip():
                        lines.append(f"{speaker}: {text.strip()}")
                transcripts[call_id] = "\n".join(lines)

            logger.info(f"Gong: fetched {len(transcripts)} transcripts (full)")
            return transcripts
        except Exception as e:
            logger.error(f"Gong API error (transcripts): {e}")
            return {}

    def get_recent_call_summaries(self, days=30, limit=10):
        """Get recent calls with FULL transcripts for analysis."""
        calls = self.get_recent_calls(days=days, limit=limit)
        if not calls:
            return []

        summaries = []
        # Get FULL transcripts for the most recent calls
        call_ids = [c.get("metaData", {}).get("id", "") for c in calls if c.get("metaData", {}).get("id")]
        transcripts = self.get_transcripts(call_ids[:5])

        for call in calls:
            meta = call.get("metaData", {})
            call_id = meta.get("id", "")
            title = meta.get("title", "Untitled Call")
            started = meta.get("started", "")
            duration = meta.get("duration", 0)

            parties = call.get("parties", [])
            participants = [p.get("name", p.get("emailAddress", "Unknown")) for p in parties]

            topics = [t.get("name", "") for t in call.get("content", {}).get("topics", [])]
            trackers = [t.get("name", "") for t in call.get("content", {}).get("trackers", [])]

            # FULL transcript — no truncation
            full_transcript = transcripts.get(call_id, "")

            summaries.append({
                "title": title,
                "date": started[:10] if started else "Unknown",
                "duration_min": round(duration / 60) if duration else 0,
                "participants": participants[:10],
                "topics": topics[:10],
                "trackers": trackers[:10],
                "transcript": full_transcript,
            })

        return summaries


def get_gong_findings():
    """Get formatted Gong findings with FULL transcripts for VicSherlock system prompt."""
    client = GongClient()
    if not client.is_configured:
        return ""

    try:
        summaries = client.get_recent_call_summaries(days=30, limit=5)
        if not summaries:
            return "GONG CALLS: No recent calls found in the last 30 days."

        lines = ["RECENT GONG CALLS WITH FULL TRANSCRIPTS (last 30 days):\n"]
        total_chars = 0
        max_total = 60000  # Stay within Claude's context limits

        for s in summaries:
            call_header = f"\n{'='*60}\nCall: {s['title']} ({s['date']}, {s['duration_min']} min)"
            if s['participants']:
                call_header += f"\nParticipants: {', '.join(s['participants'])}"
            if s['topics']:
                call_header += f"\nTopics: {', '.join(s['topics'])}"
            if s['trackers']:
                call_header += f"\nTrackers: {', '.join(s['trackers'])}"

            lines.append(call_header)

            if s['transcript']:
                # Include as much transcript as possible within limits
                remaining = max_total - total_chars
                transcript_text = s['transcript']
                if len(transcript_text) > remaining:
                    transcript_text = transcript_text[:remaining] + "\n[... transcript truncated due to length ...]"
                lines.append(f"\nFULL TRANSCRIPT:\n{transcript_text}")
                total_chars += len(transcript_text)
            else:
                lines.append("\n(No transcript available for this call)")

            lines.append("")

            if total_chars >= max_total:
                lines.append("[Remaining calls omitted due to context limits]")
                break

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error getting Gong findings: {e}")
        return ""
