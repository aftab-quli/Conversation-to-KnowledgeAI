"""
slack_bot.py
----------
VicSherlock Slack Bot — Scans Slack channels for documentation-worthy conversations.
Uses Claude to identify tutorials, troubleshooting guides, and process explanations.
Sends DMs to users when it finds something worth documenting.

Features:
- Scans public and private channels
- Analyzes conversations with Claude AI
- Identifies documentation-worthy threads
- Sends intelligent DM notifications with thread links
"""

import os
import logging
from typing import Optional
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackBotScanner:
    """Scans Slack channels for documentation-worthy conversations."""

    def __init__(self, slack_bot_token: str, anthropic_api_key: str):
        """
        Initialize the Slack bot scanner.

        Args:
            slack_bot_token: Slack bot token (xoxb-...)
            anthropic_api_key: Anthropic API key for Claude
        """
        self.slack_client = WebClient(token=slack_bot_token)
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
        self.bot_user_id = self._get_bot_user_id()

    def _get_bot_user_id(self) -> str:
        """Get the bot's user ID from Slack."""
        try:
            response = self.slack_client.auth_test()
            return response["user_id"]
        except SlackApiError as e:
            logger.error(f"Error getting bot user ID: {e}")
            raise

    def scan_channels(self, limit_channels: Optional[int] = None) -> dict:
        """
        Scan public and private channels for documentation-worthy conversations.

        Args:
            limit_channels: Limit number of channels to scan (for testing)

        Returns:
            Dictionary with scan results and documentation-worthy threads
        """
        logger.info("Starting channel scan...")
        scan_results = {
            "timestamp": datetime.now().isoformat(),
            "channels_scanned": 0,
            "threads_analyzed": 0,
            "documentation_worthy": [],
            "errors": [],
        }

        try:
            channels = self._get_all_channels(limit_channels)
            scan_results["channels_found"] = len(channels)
            logger.info(f"Found {len(channels)} channels to scan")

            for channel in channels:
                try:
                    channel_id = channel["id"]
                    channel_name = channel["name"]
                    logger.info(f"Scanning channel: #{channel_name}")

                    threads = self._get_recent_threads(channel_id)
                    scan_results["threads_analyzed"] += len(threads)

                    for thread in threads:
                        if self._is_documentation_worthy(thread, channel_name):
                            doc_entry = {
                                "channel_id": channel_id,
                                "channel_name": channel_name,
                                "thread_ts": thread["ts"],
                                "user_id": thread["user"],
                                "user_name": thread.get("username", "Unknown"),
                                "topic": thread.get("topic", ""),
                                "message_count": thread.get("message_count", 0),
                                "preview": thread.get("text", "")[:200],
                                "confidence": thread.get("confidence", 0),
                            }
                            scan_results["documentation_worthy"].append(doc_entry)
                            logger.info(f"Found documentation-worthy thread in #{channel_name}")

                    scan_results["channels_scanned"] += 1

                except SlackApiError as e:
                    error_msg = f"Error scanning channel {channel.get('name', 'unknown')}: {e}"
                    logger.error(error_msg)
                    scan_results["errors"].append(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error scanning channel: {e}"
                    logger.error(error_msg)
                    scan_results["errors"].append(error_msg)

            logger.info(
                f"Scan complete: {scan_results['channels_scanned']} channels, "
                f"{len(scan_results['documentation_worthy'])} documentation-worthy threads found"
            )
            return scan_results

        except Exception as e:
            logger.error(f"Fatal error during channel scan: {e}")
            scan_results["errors"].append(f"Fatal error: {str(e)}")
            return scan_results

    def _get_all_channels(self, limit: Optional[int] = None) -> list:
        """Get all public and private channels the bot has access to."""
        channels = []
        cursor = None

        try:
            # Get public channels
            while True:
                response = self.slack_client.conversations_list(
                    cursor=cursor,
                    limit=100,
                    exclude_archived=True,
                    types="public_channel,private_channel",
                )

                channels.extend(response["channels"])
                cursor = response.get("response_metadata", {}).get("next_cursor")

                if limit and len(channels) >= limit:
                    channels = channels[:limit]
                    break

                if not cursor:
                    break

            return channels

        except SlackApiError as e:
            logger.error(f"Error fetching channels: {e}")
            return []

    def _get_recent_threads(self, channel_id: str, days_back: int = 7) -> list:
        """
        Get recent messages and threads from a channel.

        Args:
            channel_id: Channel ID to scan
            days_back: How many days back to look

        Returns:
            List of message/thread data
        """
        threads = []
        cursor = None

        try:
            while True:
                response = self.slack_client.conversations_history(
                    channel=channel_id,
                    cursor=cursor,
                    limit=50,
                )

                messages = response.get("messages", [])

                for msg in messages:
                    # Get message metadata
                    msg_data = {
                        "ts": msg["ts"],
                        "text": msg.get("text", ""),
                        "user": msg.get("user", ""),
                        "username": msg.get("username", ""),
                        "reply_count": msg.get("reply_count", 0),
                        "message_count": msg.get("reply_count", 0) + 1,
                    }

                    # Get thread replies if this is a threaded message
                    if msg.get("thread_ts"):
                        try:
                            thread_replies = self.slack_client.conversations_replies(
                                channel=channel_id,
                                ts=msg["thread_ts"],
                                limit=100,
                            )
                            replies = thread_replies.get("messages", [])
                            msg_data["thread_messages"] = replies
                            msg_data["full_thread_text"] = " ".join(
                                [m.get("text", "") for m in replies]
                            )
                        except SlackApiError:
                            msg_data["thread_messages"] = []
                            msg_data["full_thread_text"] = msg.get("text", "")

                    threads.append(msg_data)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            return threads

        except SlackApiError as e:
            logger.error(f"Error fetching threads from {channel_id}: {e}")
            return []

    def _is_documentation_worthy(self, thread: dict, channel_name: str) -> bool:
        """
        Use Claude to determine if a thread is documentation-worthy.

        Args:
            thread: Thread data
            channel_name: Name of the channel

        Returns:
            True if documentation-worthy, False otherwise
        """
        # Skip very short messages
        text = thread.get("full_thread_text", thread.get("text", ""))
        if len(text) < 100:
            return False

        # Skip if no replies
        if thread.get("message_count", 0) < 2:
            return False

        try:
            prompt = f"""Analyze this Slack conversation and determine if it's worth documenting as a guide, tutorial, or process documentation.

Channel: #{channel_name}
Conversation:
{text[:2000]}

Consider:
1. Is someone explaining a process or how-to?
2. Are there troubleshooting steps that could help others?
3. Is it an implementation guide or technical explanation?
4. Does it contain valuable knowledge that should be preserved?

Respond with:
- A single line: YES or NO
- One line explanation (max 50 words)
- A suggested title for the documentation (max 60 chars)

Format: YES/NO | Explanation | Title"""

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            result = response.content[0].text.strip()
            is_worthy = result.upper().startswith("YES")

            if is_worthy:
                # Parse the response to extract topic
                lines = result.split("|")
                if len(lines) >= 3:
                    thread["topic"] = lines[2].strip()
                    thread["confidence"] = 0.95 if "YES" in result[:5] else 0.75
                else:
                    thread["topic"] = "Documentation"
                    thread["confidence"] = 0.75

            return is_worthy

        except Exception as e:
            logger.error(f"Error analyzing thread with Claude: {e}")
            return False

    def send_channel_notification(
        self, target_channel: str, source_channel_name: str, thread_ts: str, topic: str,
        preview: str = "", author_name: str = ""
    ) -> bool:
        """
        Post a notification to the #vicsherlock channel about a documentation-worthy thread.

        Args:
            target_channel: Channel ID or name to post to (e.g. #vicsherlock)
            source_channel_name: Name of channel where thread was found
            thread_ts: Thread timestamp
            topic: Topic/title of the documentation
            preview: Short preview of the conversation
            author_name: Name of the person who posted

        Returns:
            True if message sent successfully
        """
        try:
            author_line = f" by *{author_name}*" if author_name else ""

            message = (
                f":mag: *New finding in #{source_channel_name}*{author_line}\n\n"
                f"*{topic}*\n\n"
            )

            if preview:
                snippet = preview[:300] + ("..." if len(preview) > 300 else "")
                message += f"> {snippet}\n\n"

            message += (
                "Would you like me to:\n"
                "• *Create a new doc* from this conversation?\n"
                "• *Update an existing doc* — just send me the current doc and I'll revise it!\n\n"
                "_Reply in this thread to let me know!_"
            )

            response = self.slack_client.chat_postMessage(
                channel=target_channel, text=message, mrkdwn=True
            )

            logger.info(f"Notification posted to {target_channel} about thread in #{source_channel_name}")
            return True

        except SlackApiError as e:
            logger.error(f"Error posting to {target_channel}: {e}")
            return False

    def send_dm_notification(
        self, user_id: str, channel_name: str, thread_ts: str, topic: str,
        preview: str = "", author_name: str = ""
    ) -> bool:
        """Legacy DM method — now wraps send_channel_notification for backward compat."""
        return self.send_channel_notification(
            target_channel=user_id,
            source_channel_name=channel_name,
            thread_ts=thread_ts,
            topic=topic,
            preview=preview,
            author_name=author_name,
        )

    def _find_vicsherlock_channel(self) -> Optional[str]:
        """Find the #vicsherlock channel ID."""
        try:
            response = self.slack_client.conversations_list(
                limit=200, exclude_archived=True, types="public_channel,private_channel"
            )
            for ch in response.get("channels", []):
                if ch["name"] == "vicsherlock":
                    return ch["id"]
        except SlackApiError as e:
            logger.error(f"Error finding #vicsherlock channel: {e}")
        return None

    def notify_documentation_worthy_threads(
        self, documentation_worthy: list
    ) -> dict:
        """
        Post notifications to #vicsherlock channel for all documentation-worthy threads found.

        Args:
            documentation_worthy: List of documentation-worthy thread entries

        Returns:
            Dictionary with notification results
        """
        results = {
            "total": len(documentation_worthy),
            "sent": 0,
            "failed": 0,
            "skipped": 0,
        }

        # Find #vicsherlock channel
        target_channel = self._find_vicsherlock_channel()
        if not target_channel:
            logger.error("Could not find #vicsherlock channel — falling back to scanning user DMs")
            # If no channel found, we can't notify
            results["errors"] = ["#vicsherlock channel not found"]
            return results

        for entry in documentation_worthy:
            try:
                if results["sent"] >= 10:  # Safety limit per scan
                    results["skipped"] += 1
                    continue

                # Look up author name
                author_name = entry.get("user_name", "")
                if not author_name and entry.get("user_id"):
                    try:
                        user_info = self.slack_client.users_info(user=entry["user_id"])
                        author_name = user_info["user"].get("real_name", user_info["user"].get("name", ""))
                    except Exception:
                        pass

                success = self.send_channel_notification(
                    target_channel=target_channel,
                    source_channel_name=entry["channel_name"],
                    thread_ts=entry["thread_ts"],
                    topic=entry.get("topic", "Documentation-worthy conversation"),
                    preview=entry.get("preview", ""),
                    author_name=author_name,
                )

                if success:
                    results["sent"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"Error notifying about thread: {e}")
                results["failed"] += 1

        return results

    def perform_full_scan_and_notify(
        self, limit_channels: Optional[int] = None
    ) -> dict:
        """
        Perform a complete scan and send notifications.

        Args:
            limit_channels: Limit number of channels to scan

        Returns:
            Combined results from scan and notifications
        """
        logger.info("Starting full scan and notification process...")

        # Scan channels
        scan_results = self.scan_channels(limit_channels)

        # Send notifications
        notification_results = self.notify_documentation_worthy_threads(
            scan_results.get("documentation_worthy", [])
        )

        # Combine results
        scan_results["notifications"] = notification_results

        return scan_results


def create_slack_bot_scanner() -> Optional[SlackBotScanner]:
    """Factory function to create a SlackBotScanner with environment variables."""
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not slack_token:
        logger.warning("SLACK_BOT_TOKEN not found in environment variables")
        return None

    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        return None

    return SlackBotScanner(slack_token, anthropic_key)
