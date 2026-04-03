"""
guide_generator.py
------------------
Uses Anthropic Claude to generate structured step-by-step guides
from video transcripts. Maps extracted screenshots to guide steps.
"""

import anthropic
import json
import re


class GuideGenerator:

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_guide(self, transcript: str, instructions: str,
                       doc_type: str = "step-by-step") -> dict:
        """
        Generate a structured guide from a transcript.

        Returns dict with:
          - title: str
          - overview: str
          - steps: list of {number, title, description, key_points: list[str]}
          - conclusion: str
        """
        system_prompt = self._get_system_prompt(doc_type)
        user_prompt = self._build_user_prompt(transcript, instructions, doc_type)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = message.content[0].text
        return self._parse_guide(raw)

    def _get_system_prompt(self, doc_type: str) -> str:
        base = """You are VicSherlock, an expert technical writer at Vic.ai who creates
clear, structured documentation from conversation transcripts and video recordings.

You have deep knowledge of implementation processes, UAT testing, product configuration,
and customer onboarding workflows. You write in a professional but approachable tone.

When creating guides, you:
- Organize content into clear, numbered steps
- Include specific UI navigation paths (e.g., "Navigate to Admin > Payables > Bank Details")
- Call out critical warnings and prerequisites
- Note where screenshots would be helpful (mark with [SCREENSHOT])
- Use concrete, actionable language"""

        if doc_type == "step-by-step":
            return base + """

You specialize in step-by-step implementation guides. Each step should be
self-contained and actionable. Group related actions into logical phases
(e.g., Kickoff Tasks, Configuration, Go-Live)."""
        elif doc_type == "faq":
            return base + """

You specialize in FAQ documents. Extract the most common questions from the
transcript and provide clear, customer-facing answers."""
        elif doc_type == "troubleshooting":
            return base + """

You specialize in troubleshooting guides. Identify problems discussed in the
transcript and provide step-by-step resolution paths."""
        else:
            return base

    def _build_user_prompt(self, transcript: str, instructions: str, doc_type: str) -> str:
        return f"""Based on this video transcript and the instructions below, generate a structured document.

INSTRUCTIONS FROM USER:
{instructions}

DOCUMENT TYPE: {doc_type}

TRANSCRIPT:
{transcript}

---

Generate the document in the following exact format. Use this structure precisely:

TITLE: [Document title based on the instructions and content]

OVERVIEW: [2-3 sentence overview of what this guide covers]

STEP 1: [Step title]
DESCRIPTION: [Detailed description of what to do in this step. Include specific UI paths,
button names, and configuration details mentioned in the transcript. Be specific and actionable.]
KEY_POINTS:
- [Important detail or prerequisite]
- [Warning or tip]
- [Additional context]
[SCREENSHOT] [Brief description of what the screenshot should show]

STEP 2: [Step title]
DESCRIPTION: [...]
KEY_POINTS:
- [...]
[SCREENSHOT] [...]

(Continue for all major steps identified in the transcript)

CONCLUSION: [Summary of what was covered and any next steps]

Rules:
- Extract EVERY significant action or process described in the transcript
- Include specific names, paths, settings, and values mentioned
- Mark [SCREENSHOT] where a visual would help the reader
- Group related small actions into single steps
- Call out critical warnings with "CRITICAL:" or "WARNING:" prefixes
- If the transcript mentions feature flags, include their exact names
- Aim for 6-15 steps depending on content complexity"""

    def _parse_guide(self, raw: str) -> dict:
        """Parse Claude's structured output into a dict."""
        guide = {
            "title": "",
            "overview": "",
            "steps": [],
            "conclusion": ""
        }

        # Extract title
        title_match = re.search(r"TITLE:\s*(.+?)(?:\n|$)", raw)
        if title_match:
            guide["title"] = title_match.group(1).strip()

        # Extract overview
        overview_match = re.search(r"OVERVIEW:\s*(.+?)(?=\nSTEP|\n\n)", raw, re.DOTALL)
        if overview_match:
            guide["overview"] = overview_match.group(1).strip()

        # Extract conclusion
        conclusion_match = re.search(r"CONCLUSION:\s*(.+?)$", raw, re.DOTALL)
        if conclusion_match:
            guide["conclusion"] = conclusion_match.group(1).strip()

        # Extract steps
        step_pattern = r"STEP\s+(\d+):\s*(.+?)(?=STEP\s+\d+:|CONCLUSION:|$)"
        step_matches = re.finditer(step_pattern, raw, re.DOTALL)

        for match in step_matches:
            step_num = int(match.group(1))
            step_content = match.group(2).strip()

            step = {
                "number": step_num,
                "title": "",
                "description": "",
                "key_points": [],
                "screenshot_hint": ""
            }

            # Extract step title (first line)
            lines = step_content.split("\n")
            if lines:
                step["title"] = lines[0].strip()

            # Extract description
            desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?=KEY_POINTS:|$)", step_content, re.DOTALL)
            if desc_match:
                step["description"] = desc_match.group(1).strip()

            # Extract key points
            kp_match = re.search(r"KEY_POINTS:\s*(.+?)(?=\[SCREENSHOT\]|STEP|CONCLUSION|$)", step_content, re.DOTALL)
            if kp_match:
                points = kp_match.group(1).strip().split("\n")
                step["key_points"] = [
                    p.strip().lstrip("-").strip()
                    for p in points
                    if p.strip() and p.strip() != "-"
                ]

            # Extract screenshot hint
            ss_match = re.search(r"\[SCREENSHOT\]\s*(.+?)(?:\n|$)", step_content)
            if ss_match:
                step["screenshot_hint"] = ss_match.group(1).strip()

            guide["steps"].append(step)

        # Sort steps by number
        guide["steps"].sort(key=lambda s: s["number"])

        return guide

    def map_screenshots_to_steps(self, steps: list[dict], frame_paths: list[str]) -> dict:
        """
        Map extracted screenshots to guide steps.
        Distributes frames as evenly as possible across steps.

        Returns dict: {step_number: [frame_path, ...]}
        """
        if not frame_paths or not steps:
            return {}

        mapping = {}
        num_steps = len(steps)
        num_frames = len(frame_paths)

        if num_frames <= num_steps:
            # One frame per step (or fewer)
            for i, path in enumerate(frame_paths):
                step_num = steps[min(i, num_steps - 1)]["number"]
                mapping.setdefault(step_num, []).append(path)
        else:
            # Distribute frames evenly
            frames_per_step = num_frames / num_steps
            for i, step in enumerate(steps):
                start_idx = int(i * frames_per_step)
                end_idx = int((i + 1) * frames_per_step)
                # Take just the first frame from each group (cleaner docs)
                if start_idx < num_frames:
                    mapping[step["number"]] = [frame_paths[start_idx]]

        return mapping
