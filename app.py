"""
VicSherlock — Conversation-to-Knowledge AI
Vic.ai AI Hackathon 2025 · Built by Aftab Quli, Implementation Team
"""

import os
import io
import json
from flask import Flask, request, jsonify, send_file, render_template_string
from anthropic import Anthropic

app = Flask(__name__)

NAVY   = (28, 32, 67)
ACCENT = (91, 95, 207)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VicSherlock, Vic.ai's institutional knowledge agent.
Your job is to read a conversation — a Zoom transcript, Slack thread, Gong call, or Granola notes —
and generate a clean, structured internal enablement document.

Output ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
  "title": "Clear, descriptive document title",
  "summary": "1-2 sentence summary of what changed or what this guide covers",
  "sections": [
    {
      "heading": "Section heading",
      "content": "Optional intro sentence for this section",
      "steps": [
        "Clear, actionable step",
        "Another step"
      ]
    }
  ],
  "notes": ["Important caveat", "Another note"]
}

Rules:
- Write steps as clear, numbered instructions a new team member could follow
- Be concise but complete — no fluff
- If it's an update, focus on what changed and what the new process is
- Output ONLY the JSON — nothing before or after it"""

# ── HTML (single-page app) ────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VicSherlock — Conversation-to-Knowledge AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0d1020;
      font-family: 'Poppins', system-ui, sans-serif;
      color: #f0f2ff;
      min-height: 100vh;
    }

    /* ── Header ── */
    .header { background: #1c2043; border-bottom: 1px solid #2d3561; padding: 0 2rem; }
    .header-inner {
      max-width: 820px; margin: 0 auto;
      height: 64px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo-group { display: flex; align-items: center; gap: 16px; }
    .logo-pill {
      background: white; border-radius: 8px; padding: 5px 10px;
      display: flex; align-items: center;
    }
    .logo-pill img { height: 22px; display: block; }
    .divider { width: 1px; height: 28px; background: #2d3561; }
    .sherlock-brand { display: flex; align-items: center; gap: 8px; font-size: 18px; font-weight: 700; }
    .badge {
      background: rgba(99,102,241,.18); border: 1px solid rgba(99,102,241,.4);
      border-radius: 99px; padding: 4px 14px; font-size: 12px; color: #818cf8; font-weight: 500;
    }

    /* ── Main ── */
    .main { max-width: 820px; margin: 0 auto; padding: 48px 32px; }

    /* ── Hero ── */
    .hero { text-align: center; margin-bottom: 44px; }
    .hero h1 {
      font-size: 30px; font-weight: 700; line-height: 1.2; margin-bottom: 12px;
      background: linear-gradient(135deg, #f0f2ff, #818cf8);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #8892b0; font-size: 15px; max-width: 520px; margin: 0 auto; }

    /* ── Tabs ── */
    .tabs {
      display: flex; background: #181d3f; border: 1px solid #2d3561;
      border-radius: 12px; padding: 4px; margin-bottom: 20px;
    }
    .tab-btn {
      flex: 1; padding: 12px 8px; border-radius: 9px; border: none;
      background: transparent; color: #8892b0; font-size: 13px;
      font-family: 'Poppins', sans-serif; font-weight: 400; cursor: pointer;
      transition: all .2s; display: flex; flex-direction: column; align-items: center; gap: 2px;
    }
    .tab-btn.active { background: linear-gradient(135deg,#5b5fcf,#818cf8); color: white; font-weight: 700; }
    .tab-btn .sub { font-size: 11px; opacity: .7; font-weight: 400; }

    /* ── Card ── */
    .card { background: #181d3f; border: 1px solid #2d3561; border-radius: 16px; padding: 28px; }

    /* ── Upload zone ── */
    .upload-zone {
      border: 2px dashed #2d3561; border-radius: 12px; padding: 44px 32px;
      text-align: center; cursor: pointer; transition: all .2s;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: #818cf8; background: rgba(99,102,241,.05);
    }
    .upload-zone .icon { font-size: 44px; margin-bottom: 14px; }
    .upload-zone h3 { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
    .upload-zone p { color: #8892b0; font-size: 13px; margin-bottom: 18px; }
    .formats {
      display: inline-flex; gap: 8px; background: #232853;
      border-radius: 8px; padding: 7px 14px;
    }
    .formats span { color: #8892b0; font-size: 12px; }

    /* ── File selected ── */
    .file-selected {
      background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.3);
      border-radius: 12px; padding: 14px 18px;
      display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
    }
    .file-icon {
      width: 40px; height: 40px; border-radius: 8px;
      background: rgba(99,102,241,.2);
      display: flex; align-items: center; justify-content: center; font-size: 20px;
    }

    /* ── Inputs ── */
    .input-textarea {
      width: 100%; min-height: 160px; background: #232853;
      border: 1px solid #2d3561; border-radius: 12px; padding: 14px 16px;
      color: #f0f2ff; font-family: 'Poppins', sans-serif; font-size: 14px;
      line-height: 1.6; resize: vertical; transition: border-color .2s;
    }
    .input-textarea:focus { outline: none; border-color: #5b5fcf; }
    .input-textarea::placeholder { color: #4a5280; }
    .text-input {
      width: 100%; background: #232853; border: 1px solid #2d3561;
      border-radius: 10px; padding: 12px 16px; color: #f0f2ff;
      font-family: 'Poppins', sans-serif; font-size: 14px; margin-bottom: 14px;
      transition: border-color .2s;
    }
    .text-input:focus { outline: none; border-color: #5b5fcf; }
    .label { font-size: 12px; color: #8892b0; margin-bottom: 6px; display: block; }

    /* ── Sherlock bubble ── */
    .sherlock-bubble { display: flex; gap: 12px; align-items: flex-start; margin: 20px 0; }
    .sherlock-avatar {
      width: 38px; height: 38px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(135deg,#5b5fcf,#818cf8);
      display: flex; align-items: center; justify-content: center; font-size: 17px;
      box-shadow: 0 0 14px rgba(99,102,241,.4);
    }
    .sherlock-message {
      background: #232853; border: 1px solid rgba(99,102,241,.3);
      border-radius: 0 14px 14px 14px; padding: 13px 16px;
      flex: 1; font-size: 14px; line-height: 1.6;
    }

    /* ── Buttons ── */
    .action-btns { display: flex; gap: 12px; margin-top: 16px; }
    .btn-primary {
      flex: 1; padding: 13px; background: linear-gradient(135deg,#5b5fcf,#818cf8);
      border: none; border-radius: 10px; color: white; font-size: 14px;
      font-weight: 700; font-family: 'Poppins',sans-serif; cursor: pointer;
      box-shadow: 0 4px 18px rgba(91,95,207,.4); transition: all .2s;
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 22px rgba(91,95,207,.5); }
    .btn-primary:disabled { opacity: .5; cursor: not-allowed; transform: none; }
    .btn-secondary {
      flex: 1; padding: 13px; background: transparent;
      border: 1px solid #818cf8; border-radius: 10px; color: #818cf8;
      font-size: 14px; font-weight: 600; font-family: 'Poppins',sans-serif;
      cursor: pointer; transition: all .2s;
    }
    .btn-secondary:hover { background: rgba(99,102,241,.1); }
    .btn-full {
      width: 100%; padding: 13px; background: linear-gradient(135deg,#5b5fcf,#818cf8);
      border: none; border-radius: 10px; color: white; font-size: 14px;
      font-weight: 700; font-family: 'Poppins',sans-serif; cursor: pointer;
      box-shadow: 0 4px 18px rgba(91,95,207,.4); margin-top: 16px; transition: all .2s;
    }
    .btn-full:hover { transform: translateY(-1px); }
    .btn-restart {
      width: 100%; padding: 12px; background: transparent;
      border: 1px solid #2d3561; border-radius: 10px; color: #8892b0;
      font-size: 13px; font-family: 'Poppins',sans-serif;
      cursor: pointer; margin-top: 12px; transition: all .2s;
    }
    .btn-restart:hover { border-color: #8892b0; color: #f0f2ff; }
    .btn-download {
      padding: 10px 18px; background: rgba(16,185,129,.12);
      border: 1px solid #10b981; border-radius: 8px; color: #10b981;
      font-size: 13px; font-weight: 600; font-family: 'Poppins',sans-serif;
      cursor: pointer; text-decoration: none; display: inline-block; transition: all .2s;
    }
    .btn-download:hover { background: #10b981; color: white; }

    /* ── Progress ── */
    .progress-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .progress-header span { font-size: 13px; color: #8892b0; }
    .progress-header .pct { color: #818cf8; font-weight: 600; }
    .progress-bar { background: #2d3568; border-radius: 99px; height: 6px; overflow: hidden; margin-bottom: 22px; }
    .progress-fill {
      height: 100%; border-radius: 99px;
      background: linear-gradient(90deg,#5b5fcf,#818cf8); transition: width .5s ease;
    }
    .step-list { display: flex; flex-direction: column; gap: 10px; }
    .step-item { display: flex; align-items: center; gap: 10px; transition: opacity .3s; }
    .step-dot {
      width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      font-size: 10px; color: white; transition: background .3s;
    }
    .step-dot.done   { background: #10b981; }
    .step-dot.active { background: #5b5fcf; }
    .step-dot.pending{ background: #2d3568; }
    .step-label { font-size: 13px; transition: color .3s; }

    /* ── Doc card ── */
    .doc-card {
      background: #232853; border: 1px solid #2d3561; border-radius: 12px;
      padding: 16px 20px; display: flex; align-items: center; gap: 16px; margin: 16px 0;
    }
    .doc-icon {
      width: 44px; height: 44px; border-radius: 10px;
      background: rgba(91,95,207,.2); border: 1px solid rgba(91,95,207,.4);
      display: flex; align-items: center; justify-content: center; font-size: 22px;
    }
    .doc-info { flex: 1; }
    .doc-title { color: #f0f2ff; font-weight: 600; font-size: 14px; }
    .doc-sub { color: #8892b0; font-size: 12px; margin-top: 2px; }
    .success-ring {
      width: 64px; height: 64px; border-radius: 50%;
      background: rgba(16,185,129,.12); border: 2px solid #10b981;
      display: flex; align-items: center; justify-content: center;
      font-size: 26px; margin: 0 auto 20px;
    }

    /* ── Error ── */
    .error-msg {
      background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.3);
      border-radius: 8px; padding: 12px 16px; color: #fca5a5; font-size: 13px;
    }

    /* ── Utils ── */
    .hidden { display: none !important; }
    .mt-3  { margin-top: 12px; }
    .mt-4  { margin-top: 16px; }
    .text-center { text-align: center; }

    /* ── Footer ── */
    .footer { text-align: center; margin-top: 40px; color: #4a5280; font-size: 12px; }
    .footer span { color: #818cf8; }

    /* ── Spinner dots ── */
    @keyframes dot-pulse {
      0%,100% { transform: scale(.6); opacity: .4; }
      50%      { transform: scale(1);  opacity: 1;  }
    }
    .dots { display: inline-flex; gap: 4px; align-items: center; }
    .dot  { width: 6px; height: 6px; border-radius: 50%; background: #818cf8; }
    .dot:nth-child(1) { animation: dot-pulse 1.2s ease-in-out 0s   infinite; }
    .dot:nth-child(2) { animation: dot-pulse 1.2s ease-in-out .2s  infinite; }
    .dot:nth-child(3) { animation: dot-pulse 1.2s ease-in-out .4s  infinite; }
  </style>
</head>
<body>

<!-- ── Header ── -->
<div class="header">
  <div class="header-inner">
    <div class="logo-group">
      <div class="logo-pill">
        <img src="https://cdn.prod.website-files.com/67284e81c67879feb155c7f7/6731b45a23f7d13e55c0fc3d_Vic.ai%20Logo%20Primary%20-%20Color%20Dark.svg" alt="Vic.ai">
      </div>
      <div class="divider"></div>
      <div class="sherlock-brand"><span>🔍</span><span>VicSherlock</span></div>
    </div>
    <div class="badge">✦ AI Hackathon 2025</div>
  </div>
</div>

<!-- ── Main ── -->
<div class="main">

  <!-- Hero -->
  <div class="hero">
    <h1>Conversation-to-Knowledge AI</h1>
    <p>Upload a Zoom recording, paste a Slack thread, or drop in a Gong transcript —<br>VicSherlock writes the doc.</p>
  </div>

  <!-- Source tabs -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('zoom',this)">
      🎥 Zoom Recording
      <span class="sub">Upload video or paste transcript</span>
    </button>
    <button class="tab-btn" onclick="switchTab('slack',this)">
      #&nbsp; Slack Thread
      <span class="sub">Paste thread content</span>
    </button>
    <button class="tab-btn" onclick="switchTab('gong',this)">
      📞 Gong / Granola
      <span class="sub">Paste call transcript</span>
    </button>
  </div>

  <!-- Card -->
  <div class="card">

    <!-- STEP: input -->
    <div id="step-input">

      <!-- Zoom -->
      <div id="tab-zoom">
        <div id="upload-zone" class="upload-zone"
             onclick="document.getElementById('file-input').click()"
             ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
          <div class="icon">🎥</div>
          <h3>Drop your recording here</h3>
          <p>Supports Zoom, Loom, or any meeting recording</p>
          <div class="formats">
            <span>• .mp4</span><span>• .mov</span><span>• .webm</span><span>• .m4v</span>
          </div>
        </div>
        <input type="file" id="file-input" accept="video/*,audio/*" style="display:none" onchange="onFileSelect(event)">

        <div id="file-selected" class="file-selected hidden">
          <div class="file-icon">🎥</div>
          <div>
            <div id="file-name" style="font-weight:600;font-size:14px;"></div>
            <div id="file-size" style="color:#8892b0;font-size:12px;margin-top:2px;"></div>
          </div>
        </div>

        <div class="mt-4">
          <span class="label">Or paste your transcript directly (faster):</span>
          <textarea class="input-textarea" id="zoom-text"
            placeholder="Paste your Zoom transcript here — this skips the video upload and goes straight to doc generation..."></textarea>
        </div>
      </div>

      <!-- Slack -->
      <div id="tab-slack" class="hidden">
        <div class="sherlock-bubble">
          <div class="sherlock-avatar">🔍</div>
          <div class="sherlock-message">
            Paste your Slack thread below and I'll identify what needs to be documented or updated.
          </div>
        </div>
        <span class="label">Channel (optional):</span>
        <input class="text-input" id="slack-channel" type="text" placeholder="#project_vicpaygolive">
        <span class="label">Thread content:</span>
        <textarea class="input-textarea" id="slack-text" placeholder="Paste the Slack thread here..."></textarea>
      </div>

      <!-- Gong / Granola -->
      <div id="tab-gong" class="hidden">
        <div class="sherlock-bubble">
          <div class="sherlock-avatar">🔍</div>
          <div class="sherlock-message">
            No MCP connector exists for Gong or Granola yet — no problem. Export your transcript, paste it here, and VicSherlock handles the rest exactly the same way.
          </div>
        </div>
        <span class="label">Paste your transcript:</span>
        <textarea class="input-textarea" id="gong-text" placeholder="Paste your Gong or Granola transcript here..."></textarea>
      </div>

      <div id="input-error" class="error-msg mt-3 hidden"></div>
      <button class="btn-full" onclick="handleAnalyze()">🔍 Analyze with VicSherlock</button>
    </div>

    <!-- STEP: confirm -->
    <div id="step-confirm" class="hidden">
      <div class="sherlock-bubble">
        <div class="sherlock-avatar">🔍</div>
        <div class="sherlock-message" id="confirm-msg">I've reviewed the content. What would you like me to do?</div>
      </div>
      <div class="action-btns">
        <button class="btn-primary" onclick="handleGenerate('new')">✨ Create a new guide</button>
        <button class="btn-secondary" onclick="handleGenerate('update')">📝 Update existing</button>
      </div>
      <button class="btn-restart" onclick="restart()">← Start over</button>
    </div>

    <!-- STEP: processing -->
    <div id="step-processing" class="hidden">
      <div style="font-weight:600;margin-bottom:20px;font-size:14px;">
        🔍 VicSherlock is writing your document
        <span class="dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>
      </div>
      <div class="progress-header">
        <span id="progress-text">Analyzing content...</span>
        <span class="pct" id="progress-pct">0%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" id="progress-fill" style="width:0%"></div>
      </div>
      <div class="step-list" id="step-list"></div>
    </div>

    <!-- STEP: done -->
    <div id="step-done" class="hidden text-center">
      <div class="success-ring">✓</div>
      <div class="sherlock-bubble" style="text-align:left;">
        <div class="sherlock-avatar">🔍</div>
        <div class="sherlock-message" id="done-msg">Case closed. Your document is ready.</div>
      </div>
      <div class="doc-card">
        <div class="doc-icon">📄</div>
        <div class="doc-info">
          <div class="doc-title" id="doc-title">VicSherlock Generated Guide</div>
          <div class="doc-sub">Generated by VicSherlock · Vic-ready format · Ready to share</div>
        </div>
        <a id="download-link" class="btn-download" href="#" download>⬇ Download</a>
      </div>
      <button class="btn-restart" onclick="restart()">🔍 Analyze another conversation</button>
    </div>

    <!-- STEP: error -->
    <div id="step-error" class="hidden">
      <div class="error-msg" id="error-msg" style="margin-bottom:16px;">Something went wrong.</div>
      <button class="btn-restart" onclick="restart()">← Try again</button>
    </div>

  </div><!-- /card -->

  <div class="footer">
    The knowledge is always in the conversation. VicSherlock just finds it. ·
    <span>Aftab Quli · Implementation Team · Vic.ai AI Hackathon 2025</span>
  </div>
</div>

<script>
  let currentTab = 'zoom';
  let selectedFile = null;
  let pendingContent = null;
  let pendingSource  = null;

  const STEPS = {
    zoom:  ['Reading content...','Identifying workflow steps...','Structuring the guide...','Writing instructions...','Formatting document...','Finalizing...'],
    slack: ['Reading Slack thread...','Detecting what changed...','Cross-referencing guides...','Writing updated section...','Formatting document...','Finalizing...'],
    gong:  ['Reading transcript...','Extracting key insights...','Structuring documentation...','Writing step-by-step guide...','Formatting document...','Finalizing...'],
  };

  const CONFIRM_MSGS = {
    zoom:  "I've reviewed the recording content. I found workflow steps worth documenting. Create a brand new guide, or update an existing one?",
    slack: "I've scanned the thread. There's a process change here that should be captured in documentation. Create a new guide or update an existing one?",
    gong:  "I've read through the transcript. There's valuable content worth turning into documentation. Create a new guide or update an existing one?",
  };

  function switchTab(tab, btn) {
    currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    ['zoom','slack','gong'].forEach(t =>
      document.getElementById('tab-'+t).classList.toggle('hidden', t !== tab));
  }

  function onDragOver(e)  { e.preventDefault(); document.getElementById('upload-zone').classList.add('drag-over'); }
  function onDragLeave(e) { document.getElementById('upload-zone').classList.remove('drag-over'); }
  function onDrop(e)      { e.preventDefault(); onDragLeave(e); setFile(e.dataTransfer.files[0]); }
  function onFileSelect(e){ setFile(e.target.files[0]); }

  function setFile(file) {
    if (!file) return;
    selectedFile = file;
    document.getElementById('upload-zone').classList.add('hidden');
    document.getElementById('file-selected').classList.remove('hidden');
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = (file.size/(1024*1024)).toFixed(1)+' MB';
  }

  function getInput() {
    if (currentTab === 'zoom') {
      const t = document.getElementById('zoom-text').value.trim();
      if (t) return { content: t, source: 'zoom' };
      if (selectedFile) return { content: '[Video uploaded: '+selectedFile.name+']', source: 'zoom' };
      return null;
    }
    if (currentTab === 'slack') {
      const ch = document.getElementById('slack-channel').value.trim();
      const tx = document.getElementById('slack-text').value.trim();
      if (tx) return { content: (ch ? 'Channel: '+ch+'\n\n' : '') + tx, source: 'slack' };
      return null;
    }
    const tx = document.getElementById('gong-text').value.trim();
    return tx ? { content: tx, source: 'gong' } : null;
  }

  function handleAnalyze() {
    const input = getInput();
    if (!input) {
      const el = document.getElementById('input-error');
      el.textContent = 'Please upload a recording or paste your transcript / thread content.';
      el.classList.remove('hidden');
      setTimeout(() => el.classList.add('hidden'), 5000);
      return;
    }
    pendingContent = input.content;
    pendingSource  = input.source;
    document.getElementById('confirm-msg').textContent = CONFIRM_MSGS[currentTab];
    showStep('confirm');
  }

  async function handleGenerate(action) {
    showStep('processing');
    animateProgress(currentTab);

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: pendingContent, action, source: pendingSource })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Generation failed');
      }

      const disp = res.headers.get('Content-Disposition') || '';
      const match = disp.match(/filename="?([^"]+)"?/);
      const filename = match ? match[1] : 'VicSherlock_Output.docx';
      const title = filename.replace('.docx','').replace(/_/g,' ');

      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);

      document.getElementById('download-link').href     = url;
      document.getElementById('download-link').download = filename;
      document.getElementById('doc-title').textContent  = title;
      document.getElementById('done-msg').textContent   =
        'Case closed. "'+title+'" is ready — download and share wherever it lives.';

      const delay = STEPS[currentTab].length * 750 + 500;
      setTimeout(() => showStep('done'), delay);

    } catch(err) {
      document.getElementById('error-msg').textContent = 'Error: ' + err.message;
      setTimeout(() => showStep('error'), 500);
    }
  }

  function animateProgress(tab) {
    const steps = STEPS[tab];
    const list  = document.getElementById('step-list');
    list.innerHTML = '';

    steps.forEach((s, i) => {
      list.innerHTML += `
        <div class="step-item" id="si-${i}" style="opacity:.3">
          <div class="step-dot pending" id="dot-${i}"></div>
          <span class="step-label" id="lbl-${i}" style="color:#8892b0">${s}</span>
        </div>`;
    });

    let i = 0;
    const iv = setInterval(() => {
      if (i > 0) {
        document.getElementById('dot-'+(i-1)).className = 'step-dot done';
        document.getElementById('dot-'+(i-1)).textContent = '✓';
        document.getElementById('lbl-'+(i-1)).style.color = '#10b981';
      }
      if (i < steps.length) {
        document.getElementById('si-'+i).style.opacity = '1';
        document.getElementById('dot-'+i).className = 'step-dot active';
        document.getElementById('lbl-'+i).style.color = '#f0f2ff';
        document.getElementById('progress-text').textContent = steps[i];
        document.getElementById('progress-pct').textContent  = Math.round((i/steps.length)*100)+'%';
        document.getElementById('progress-fill').style.width = Math.round((i/steps.length)*100)+'%';
        i++;
      } else {
        document.getElementById('progress-pct').textContent  = '100%';
        document.getElementById('progress-fill').style.width = '100%';
        clearInterval(iv);
      }
    }, 750);
  }

  function showStep(s) {
    ['input','confirm','processing','done','error'].forEach(n =>
      document.getElementById('step-'+n).classList.toggle('hidden', n !== s));
  }

  function restart() {
    selectedFile = null; pendingContent = null; pendingSource = null;
    document.getElementById('file-input').value = '';
    document.getElementById('upload-zone').classList.remove('hidden');
    document.getElementById('file-selected').classList.add('hidden');
    ['zoom-text','slack-text','slack-channel','gong-text'].forEach(id =>
      { const el = document.getElementById(id); if(el) el.value=''; });
    showStep('input');
  }
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/generate", methods=["POST"])
def generate():
    data    = request.get_json(force=True)
    content = (data.get("content") or "").strip()
    action  = data.get("action", "new")
    source  = data.get("source", "zoom")

    if not content:
        return jsonify({"error": "No content provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        # Demo mode — return a realistic pre-built example document
        source_titles = {
            "zoom": "NetSuite Implementation — Payment Configuration Walkthrough",
            "slack": "AP Workflow Update — 3-Way Match Tolerance Change",
            "gong": "Customer Onboarding Call — VicPay Go-Live Checklist",
        }
        doc_data = {
            "title": source_titles.get(source, "VicSherlock Generated Guide"),
            "summary": "This guide captures the key decisions and action items discussed. Follow the steps below to complete implementation and ensure all stakeholders are aligned.",
            "sections": [
                {
                    "heading": "Background & Context",
                    "content": "The following was identified as doc-worthy based on the conversation content.",
                    "steps": [
                        "Review the existing configuration in NetSuite before making changes.",
                        "Confirm with the customer which legal entities are in scope for this rollout.",
                        "Validate that cloud recording is enabled in Zoom org-wide settings prior to the call."
                    ]
                },
                {
                    "heading": "Step-by-Step Process",
                    "content": "Follow these steps in order. Do not skip steps — each one depends on the previous.",
                    "steps": [
                        "Log in to NetSuite as Administrator and navigate to Setup > Accounting > Accounting Preferences.",
                        "Locate the 'Payment Processing' section and confirm the default currency matches the customer's primary entity.",
                        "Enable the VicPay integration by toggling the connector to 'Active' and entering the API credentials from the Vic.ai portal.",
                        "Run a test transaction with a $1.00 dummy invoice to confirm the end-to-end flow is working.",
                        "Once confirmed, notify the customer's AP team lead that live processing is enabled.",
                        "Document the go-live date and any exceptions in the project tracker."
                    ]
                },
                {
                    "heading": "Key Decisions Made",
                    "content": "",
                    "steps": [
                        "3-way match tolerance set to 2% per customer request (default is 0%).",
                        "Approval workflow bypassed for invoices under $500 — customer accepted risk.",
                        "Monthly reconciliation cadence agreed upon — customer will own this internally."
                    ]
                },
                {
                    "heading": "Next Steps & Owners",
                    "content": "",
                    "steps": [
                        "Aftab to share this guide with the customer's IT lead by EOD Friday.",
                        "Customer to complete UAT sign-off checklist within 5 business days.",
                        "Vic.ai CS team to schedule 30-day check-in call post go-live."
                    ]
                }
            ],
            "notes": [
                "This document was auto-generated by VicSherlock from a conversation. Review before publishing.",
                "If updating an existing Guru article, paste the article URL into the VicSherlock prompt for diff-mode.",
                "Screenshots from screen share recordings can be appended manually to Section 2."
            ]
        }
    else:
        client = Anthropic(api_key=api_key)

        user_msg = (
            f"Source: {source}\n"
            f"Action: {'Create a brand new guide' if action == 'new' else 'Update an existing guide'}\n\n"
            f"Content:\n{content}"
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}]
            )
            raw  = response.content[0].text
            # Extract JSON robustly
            start    = raw.find("{")
            end      = raw.rfind("}") + 1
            doc_data = json.loads(raw[start:end])
        except Exception as e:
            return jsonify({"error": f"Claude API error: {str(e)}"}), 500

    try:
        doc_bytes = build_docx(doc_data)
    except Exception as e:
        return jsonify({"error": f"Document build error: {str(e)}"}), 500

    safe_title = doc_data.get("title", "VicSherlock_Output")
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in safe_title)[:60]
    filename   = safe_title.replace(" ", "_") + ".docx"

    return send_file(
        io.BytesIO(doc_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


# ── DOCX builder ──────────────────────────────────────────────────────────────

def build_docx(data):
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = Document()

    # Margins
    for sec in doc.sections:
        sec.top_margin    = Inches(1)
        sec.bottom_margin = Inches(1)
        sec.left_margin   = Inches(1.2)
        sec.right_margin  = Inches(1.2)

    def add_run(para, text, bold=False, italic=False, size=11, color=None, name="Calibri"):
        run = para.add_run(text)
        run.bold   = bold
        run.italic = italic
        run.font.size = Pt(size)
        run.font.name = name
        if color:
            run.font.color.rgb = RGBColor(*color)
        return run

    # Title
    t = doc.add_paragraph()
    add_run(t, data.get("title", "VicSherlock Generated Guide"),
            bold=True, size=22, color=NAVY)

    # Summary
    if data.get("summary"):
        s = doc.add_paragraph()
        add_run(s, data["summary"], italic=True, size=11, color=(100, 110, 145))

    doc.add_paragraph()

    # Sections
    for section in data.get("sections", []):
        h = doc.add_paragraph()
        add_run(h, section.get("heading", ""), bold=True, size=14, color=ACCENT)

        if section.get("content"):
            c = doc.add_paragraph()
            add_run(c, section["content"], size=11)

        for step in section.get("steps", []):
            p = doc.add_paragraph(style="List Number")
            add_run(p, step, size=11)

        doc.add_paragraph()

    # Notes
    if data.get("notes"):
        nh = doc.add_paragraph()
        add_run(nh, "Notes", bold=True, size=13, color=NAVY)
        for note in data["notes"]:
            np_ = doc.add_paragraph(style="List Bullet")
            add_run(np_, note, size=10)

    doc.add_paragraph()

    # Footer
    f = doc.add_paragraph()
    add_run(f, "Generated by VicSherlock · Vic.ai · Conversation-to-Knowledge AI",
            italic=True, size=9, color=(150, 155, 185))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
