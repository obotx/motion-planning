import sys
import os
import json
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.pubsub import IPCPubSub

def parse_value(value_str, value_type):
    """Parse string value based on type"""
    if value_type == "bool":
        return value_str.lower() in ("true", "1", "yes", "on")
    elif value_type == "float":
        return float(value_str)
    elif value_type == "int":
        return int(value_str)
    elif value_type == "list":
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            cleaned = value_str.strip()
            if cleaned.startswith('[') and cleaned.endswith(']'):
                items = cleaned[1:-1].split(',')
                return [float(x.strip()) for x in items if x.strip()]
            else:
                raise ValueError(f"Cannot parse list: {value_str}")
    else:
        return value_str 

def main():
    parser = argparse.ArgumentParser(description="CLI publisher for IPC topics")
    parser.add_argument("command", choices=["pub"], help="Command (only 'pub' supported)")
    parser.add_argument("topic", help="Topic name (e.g., target_base)")
    parser.add_argument("type", choices=["bool", "float", "int", "list", "str"], help="Message type")
    parser.add_argument("value", help="Message value as string")
    args = parser.parse_args()

    try:
        message = parse_value(args.value, args.type)
        print(f"Publishing to '{args.topic}': {message} (type: {type(message).__name__})")

        ipc = IPCPubSub()
        pub = ipc.create_publisher()
        pub.publish(args.topic, message)
        print("✓ Published!")

    except Exception as e:
        print(f" Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()