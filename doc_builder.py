"""
doc_builder.py
--------------
Generates a formatted Word document (.docx) from a structured guide,
with embedded screenshots at the appropriate steps.
Uses the Node.js docx library via subprocess.
"""

import subprocess
import os
import json
import base64
from datetime import datetime
from pathlib import Path


def generate_guide_document(guide: dict, frame_map: dict, output_path: str):
    """
    Generate a Vic-branded .docx from a structured guide with embedded screenshots.

    guide = {
        title: str,
        overview: str,
        steps: [{number, title, description, key_points: [str], screenshot_hint: str}],
        conclusion: str
    }
    frame_map = {step_number: [frame_path, ...]}
    """

    date_str = datetime.now().strftime("%B %d, %Y")
    title = _escape(guide.get("title", "VicSherlock Guide"))
    overview = _escape(guide.get("overview", ""))
    conclusion = _escape(guide.get("conclusion", ""))

    # Build steps JS
    steps_js = ""
    for step in guide.get("steps", []):
        step_num = step["number"]
        step_title = _escape(step.get("title", f"Step {step_num}"))
        description = _escape(step.get("description", ""))
        key_points = step.get("key_points", [])

        # Step heading
        steps_js += f"""
      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        spacing: {{ before: 400, after: 120 }},
        children: [new TextRun({{ text: "Step {step_num}: {step_title}", font: "Arial" }})]
      }}),
      new Paragraph({{
        spacing: {{ before: 80, after: 160 }},
        children: [new TextRun({{ text: "{description}", font: "Arial", size: 22 }})]
      }}),"""

        # Key points
        for point in key_points:
            escaped_point = _escape(point)
            # Highlight critical warnings
            if any(w in point.upper() for w in ["CRITICAL", "WARNING", "IMPORTANT"]):
                steps_js += f"""
      new Paragraph({{
        numbering: {{ reference: "bullets", level: 0 }},
        spacing: {{ before: 40, after: 40 }},
        children: [new TextRun({{ text: "{escaped_point}", font: "Arial", size: 22, bold: true, color: "CC0000" }})]
      }}),"""
            else:
                steps_js += f"""
      new Paragraph({{
        numbering: {{ reference: "bullets", level: 0 }},
        spacing: {{ before: 40, after: 40 }},
        children: [new TextRun({{ text: "{escaped_point}", font: "Arial", size: 22 }})]
      }}),"""

        # Embedded screenshots for this step
        frames = frame_map.get(step_num, [])
        for frame_path in frames:
            if os.path.exists(frame_path):
                abs_path = os.path.abspath(frame_path).replace("\\", "/")
                steps_js += f"""
      new Paragraph({{
        spacing: {{ before: 160, after: 160 }},
        alignment: AlignmentType.CENTER,
        children: [
          new ImageRun({{
            data: fs.readFileSync("{abs_path}"),
            transformation: {{ width: 550, height: 350 }},
          }})
        ]
      }}),"""

    script = f"""
const {{ Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
         HeadingLevel, LevelFormat, BorderStyle, PageNumber, Header, Footer }} = require('docx');
const fs = require('fs');

const doc = new Document({{
  numbering: {{
    config: [{{
      reference: "bullets",
      levels: [{{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: {{ paragraph: {{ indent: {{ left: 720, hanging: 360 }} }} }}
      }}]
    }}]
  }},
  styles: {{
    default: {{ document: {{ run: {{ font: "Arial", size: 22 }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: {{ size: 36, bold: true, font: "Arial", color: "1a1a2e" }},
        paragraph: {{ spacing: {{ before: 240, after: 120 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: {{ size: 26, bold: true, font: "Arial", color: "16213e" }},
        paragraph: {{ spacing: {{ before: 200, after: 80 }}, outlineLevel: 1 }} }},
    ]
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 12240, height: 15840 }},
        margin: {{ top: 1440, right: 1080, bottom: 1440, left: 1080 }}
      }}
    }},
    headers: {{
      default: new Header({{
        children: [new Paragraph({{
          border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: "7c3aed", space: 1 }} }},
          children: [
            new TextRun({{ text: "VicSherlock", font: "Arial", size: 20, bold: true, color: "7c3aed" }}),
            new TextRun({{ text: " — {title}", font: "Arial", size: 20, color: "888888" }})
          ]
        }})]
      }})
    }},
    footers: {{
      default: new Footer({{
        children: [new Paragraph({{
          alignment: AlignmentType.CENTER,
          border: {{ top: {{ style: BorderStyle.SINGLE, size: 6, color: "CCCCCC", space: 1 }} }},
          children: [
            new TextRun({{ text: "Page ", font: "Arial", size: 18, color: "888888" }}),
            new TextRun({{ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" }}),
            new TextRun({{ text: " | Generated by VicSherlock | {date_str}", font: "Arial", size: 18, color: "888888" }})
          ]
        }})]
      }})
    }},
    children: [
      // Title
      new Paragraph({{
        heading: HeadingLevel.HEADING_1,
        spacing: {{ before: 0, after: 80 }},
        children: [new TextRun({{ text: "{title}", bold: true, size: 44, font: "Arial", color: "1a1a2e" }})]
      }}),
      // Subtitle
      new Paragraph({{
        spacing: {{ before: 0, after: 120 }},
        children: [new TextRun({{ text: "Generated by VicSherlock · {date_str}", size: 20, color: "888888", italics: true, font: "Arial" }})]
      }}),
      // Divider
      new Paragraph({{
        border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 8, color: "7c3aed", space: 1 }} }},
        spacing: {{ before: 0, after: 320 }},
        children: []
      }}),
      // Overview
      new Paragraph({{
        spacing: {{ before: 0, after: 80 }},
        children: [new TextRun({{ text: "Overview", bold: true, size: 28, font: "Arial", color: "16213e" }})]
      }}),
      new Paragraph({{
        spacing: {{ before: 0, after: 320 }},
        children: [new TextRun({{ text: "{overview}", font: "Arial", size: 22 }})]
      }}),
      // Steps
      {steps_js}
      // Conclusion
      new Paragraph({{
        heading: HeadingLevel.HEADING_2,
        spacing: {{ before: 400, after: 120 }},
        children: [new TextRun({{ text: "Conclusion", font: "Arial" }})]
      }}),
      new Paragraph({{
        spacing: {{ before: 80, after: 200 }},
        children: [new TextRun({{ text: "{conclusion}", font: "Arial", size: 22 }})]
      }}),
    ]
  }}]
}});

Packer.toBuffer(doc).then(buffer => {{
  fs.writeFileSync("{output_path.replace(chr(92), '/')}", buffer);
  console.log("Created: {output_path.replace(chr(92), '/')}");
}}).catch(err => {{
  console.error("Error:", err.message);
  process.exit(1);
}});
"""

    # Write and execute the Node.js script
    script_path = output_path.replace(".docx", "_gen.js")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    result = subprocess.run(["node", script_path], capture_output=True, text=True)

    # Cleanup script
    try:
        os.remove(script_path)
    except Exception:
        pass

    if result.returncode != 0:
        raise RuntimeError(f"Document generation failed: {result.stderr}")


def _escape(text: str) -> str:
    """Escape text for safe use in JavaScript strings."""
    if not text:
        return ""
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", " ")
    )


# Helper for f-string backslash workaround
def chr(code):
    return __builtins__["chr"](code) if isinstance(__builtins__, dict) else __import__("builtins").chr(code)
