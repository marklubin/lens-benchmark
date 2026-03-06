#!/usr/bin/env python3
import sys
import os
import json
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..lib.agent import Agent


def print_help():
    """Print help message."""
    help_text = """
KV Agent CLI - Persistent agent with key-value store memory

Commands:
  help                    Show this help message
  quit, exit              Exit the agent
  mode [history|kv_only]  Set interaction mode
  kv show                 Show all KV store contents
  kv search <pattern>     Search KV store for pattern
  history                 Show conversation history
  clear                   Clear conversation history (in history mode)

In any mode, type your query to get a response from the agent.
"""
    print(help_text)


def main():
    """Main CLI loop."""
    # Initialize agent
    try:
        agent = Agent()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set the KIMI_API_KEY environment variable.")
        sys.exit(1)

    print("Welcome to KV Agent CLI!")
    print("Type 'help' for available commands.")
    print("Enter your queries below:\n")

    while True:
        try:
            try:
                user_input = input("> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                if user_input.lower() == "help":
                    print_help()
                    continue
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Parse command
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command in ["quit", "exit"]:
                print("Goodbye!")
                break
            elif command == "help":
                print_help()
            elif command == "mode":
                if args in ["history", "kv_only"]:
                    agent.set_mode(args)
                    print(f"Mode set to {args}")
                else:
                    print("Mode must be 'history' or 'kv_only'")
            elif command == "kv":
                kv_parts = args.split(" ", 1)
                kv_command = kv_parts[0].lower() if kv_parts else ""
                kv_args = kv_parts[1] if len(kv_parts) > 1 else ""

                if kv_command == "show":
                    contents = agent.get_kv_store_contents()
                    if contents:
                        print(f"KV Store contents ({len(contents)} items):")
                        for item in contents:
                            print(f"  {item['key']}: {item['value']}")
                    else:
                        print("KV Store is empty")
                elif kv_command == "search" and kv_args:
                    results = agent.search_kv_store(kv_args)
                    if results:
                        print(f"Search results for '{kv_args}':")
                        for item in results:
                            print(f"  {item['key']}: {item['value']} ({item['updated_at']})")
                    else:
                        print(f"No results found for '{kv_args}'")
                else:
                    print("KV commands: 'show', 'search <pattern>'")
            elif command == "history":
                if agent.history:
                    print("Conversation history:")
                    for msg in agent.history:
                        print(f"  {msg['role']}: {msg['content']}")
                else:
                    print("No conversation history")
            elif command == "clear":
                if agent.mode == "history":
                    agent.history = []
                    print("Conversation history cleared")
                else:
                    print("Clear command only available in history mode")
            else:
                # Treat as query to agent
                response = agent.respond(user_input)
                print(f"{response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
