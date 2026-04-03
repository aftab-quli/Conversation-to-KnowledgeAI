"""
slack_bot_demo.py
----------------
Demo script for testing the VicSherlock Slack bot scanner.
Shows how to use the SlackBotScanner to find documentation-worthy conversations.

Run with:
    python slack_bot_demo.py [--limit 5]

Environment variables needed:
    SLACK_BOT_TOKEN - Slack bot token (xoxb-...)
    ANTHROPIC_API_KEY - Anthropic API key for Claude
"""

import sys
import argparse
import logging
import json
from slack_bot import create_slack_bot_scanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Scan Slack channels for documentation-worthy conversations"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of channels to scan (default: 5)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send DM notifications for found documentation",
    )

    args = parser.parse_args()

    logger.info("Starting VicSherlock Slack Bot Demo...")
    logger.info(f"Scanning up to {args.limit} channels")

    # Create scanner
    bot = create_slack_bot_scanner()
    if not bot:
        logger.error(
            "Failed to initialize Slack bot. Check SLACK_BOT_TOKEN and ANTHROPIC_API_KEY."
        )
        sys.exit(1)

    # Perform scan
    logger.info("Scanning channels for documentation-worthy conversations...")
    results = bot.scan_channels(limit_channels=args.limit)

    # Print results
    print("\n" + "=" * 80)
    print("SCAN RESULTS")
    print("=" * 80)
    print(f"\nChannels scanned: {results['channels_scanned']}")
    print(f"Threads analyzed: {results['threads_analyzed']}")
    print(f"Documentation-worthy conversations found: {len(results['documentation_worthy'])}")

    if results.get("errors"):
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")

    if results["documentation_worthy"]:
        print("\n" + "-" * 80)
        print("DOCUMENTATION-WORTHY CONVERSATIONS FOUND:")
        print("-" * 80)

        for i, doc in enumerate(results["documentation_worthy"], 1):
            print(f"\n{i}. {doc['topic']}")
            print(f"   Channel: #{doc['channel_name']}")
            print(f"   User: {doc['user_name']} ({doc['user_id']})")
            print(f"   Messages: {doc['message_count']}")
            print(f"   Confidence: {doc['confidence']*100:.0f}%")
            print(f"   Preview: {doc['preview'][:100]}...")
            print(f"   Thread: {doc['thread_ts']}")

        # Send notifications if requested
        if args.notify:
            print("\n" + "=" * 80)
            print("SENDING NOTIFICATIONS...")
            print("=" * 80)

            notification_results = bot.notify_documentation_worthy_threads(
                results["documentation_worthy"]
            )

            print(f"\nNotifications sent: {notification_results['sent']}/{notification_results['total']}")
            print(f"Failed: {notification_results['failed']}")
            print(f"Skipped: {notification_results['skipped']}")
    else:
        print("\nNo documentation-worthy conversations found in this scan.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
