# Deploy VicSherlock — Full Setup Guide

## Step 1 — Create the Slack App (10 minutes)

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**
2. Name it **VicSherlock**, select your Vic.ai workspace
3. Upload a profile picture (use the VicSherlock logo)

### OAuth Scopes (Bot Token)
Go to **OAuth & Permissions** → **Bot Token Scopes** and add:
- `channels:history` — read public channel messages
- `channels:read` — list channels
- `groups:history` — read private channel messages
- `groups:read` — list private channels
- `im:history` — read DMs
- `im:read` — list DMs
- `im:write` — open DM conversations
- `chat:write` — send messages
- `files:write` — upload .docx files
- `users:read` — look up users
- `users:read.email` — find users by email (for Zoom webhook)
- `mpim:history` — read group DMs

### Event Subscriptions
Go to **Event Subscriptions** → enable → set Request URL to:
`https://your-render-url.onrender.com/slack/events`

Subscribe to these **Bot Events**:
- `message.channels`
- `message.groups`
- `message.im`
- `app_mention`

### Install the App
Go to **OAuth & Permissions** → **Install to Workspace** → copy the **Bot User OAuth Token**

Go to **Basic Information** → copy the **Signing Secret**

---

## Step 2 — Deploy to Render (5 minutes)

1. Push this folder to a GitHub repo
2. Go to https://render.com → **New Web Service** → connect your repo
3. Set **Start Command** to: `gunicorn slack_bot:flask_app`
4. Add these environment variables:

| Key | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your key from https://console.anthropic.com |
| `SLACK_BOT_TOKEN` | Bot User OAuth Token from Step 1 |
| `SLACK_SIGNING_SECRET` | Signing Secret from Step 1 |
| `ZOOM_API_TOKEN` | Zoom Server-to-Server OAuth token (for Zoom webhook) |

5. Deploy — copy your public Render URL

---

## Step 3 — Connect Zoom Webhook (optional, for auto-trigger on every call)

1. Go to https://marketplace.zoom.us → **Develop** → **Build App** → **Webhook Only**
2. Add Event Subscription:
   - Event: `recording.completed`
   - Notification URL: `https://your-render-url.onrender.com/zoom/webhook`
3. Save — Zoom will now ping VicSherlock every time a cloud recording finishes

---

## Step 4 — Test it

1. Go to your Slack workspace → find VicSherlock in Apps
2. DM it: *"Create a FAQ from this"* and paste any transcript
3. It will respond, ask for doc type, generate the .docx, and send it back

VicSherlock is also now watching every channel. Post a process change and wait for the DM.
