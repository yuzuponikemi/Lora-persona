#!/usr/bin/env python3
"""
Slack Data Processor for LoRA Fine-tuning
==========================================

This script processes Slack export data and creates a standardized dataset
for LLM fine-tuning, specifically for persona modeling.

Features:
- Loads Slack export directories
- Extracts user metadata and conversations
- Creates conversation context and user responses
- Outputs in multiple formats (CSV, JSON, JSONL)
- Follows industry standards for instruction-tuning datasets

Author: Generated for LoRA Persona Training
"""

import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import argparse

class SlackDataProcessor:
    """Process Slack export data for LLM fine-tuning"""
    
    def __init__(self, slack_export_path: str):
        self.slack_export_path = Path(slack_export_path)
        self.users = {}
        self.channels = {}
        self.conversations = []
        self.processed_data = []
        
    def load_metadata(self):
        """Load users and channels metadata"""
        print("Loading Slack metadata...")
        
        # Load users
        users_file = self.slack_export_path / "users.json"
        if users_file.exists():
            with open(users_file, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
                self.users = {user['id']: user for user in users_data}
                print(f"Loaded {len(self.users)} users")
        else:
            print("Warning: users.json not found")
            
        # Load channels
        channels_file = self.slack_export_path / "channels.json"
        if channels_file.exists():
            with open(channels_file, 'r', encoding='utf-8') as f:
                channels_data = json.load(f)
                self.channels = {channel['id']: channel for channel in channels_data}
                print(f"Loaded {len(self.channels)} channels")
        else:
            print("Warning: channels.json not found")
    
    def get_user_display_name(self, user_id: str) -> str:
        """Get user display name or real name"""
        if user_id in self.users:
            user = self.users[user_id]
            return user.get('profile', {}).get('display_name') or \
                   user.get('profile', {}).get('real_name') or \
                   user.get('name', user_id)
        return user_id
    
    def get_channel_name(self, channel_id: str) -> str:
        """Get channel name"""
        if channel_id in self.channels:
            return self.channels[channel_id].get('name', channel_id)
        return channel_id
    
    def load_conversations(self):
        """Load all conversation files"""
        print("Loading conversations...")
        
        conversation_count = 0
        for channel_dir in self.slack_export_path.iterdir():
            if channel_dir.is_dir() and not channel_dir.name.startswith('.'):
                channel_name = channel_dir.name
                print(f"Processing channel: {channel_name}")
                
                for json_file in channel_dir.glob("*.json"):
                    if json_file.name != "canvas_in_the_conversation.json":
                        with open(json_file, 'r', encoding='utf-8') as f:
                            try:
                                messages = json.load(f)
                                for message in messages:
                                    message['channel'] = channel_name
                                    message['date'] = json_file.stem
                                    self.conversations.append(message)
                                    conversation_count += 1
                            except json.JSONDecodeError as e:
                                print(f"Error reading {json_file}: {e}")
        
        print(f"Loaded {conversation_count} messages from {len(self.conversations)} total")
        
        # Sort by timestamp
        self.conversations.sort(key=lambda x: float(x.get('ts', 0)))
    
    def analyze_users(self) -> Dict[str, Dict]:
        """Analyze user activity and return statistics"""
        user_stats = defaultdict(lambda: {
            'message_count': 0,
            'channels': set(),
            'dates': set(),
            'avg_message_length': 0,
            'total_chars': 0
        })
        
        for msg in self.conversations:
            user_id = msg.get('user')
            if user_id and msg.get('text'):
                stats = user_stats[user_id]
                stats['message_count'] += 1
                stats['channels'].add(msg.get('channel', 'unknown'))
                stats['dates'].add(msg.get('date', 'unknown'))
                stats['total_chars'] += len(msg.get('text', ''))
        
        # Calculate averages and convert sets to counts
        for user_id, stats in user_stats.items():
            if stats['message_count'] > 0:
                stats['avg_message_length'] = stats['total_chars'] / stats['message_count']
                stats['channel_count'] = len(stats['channels'])
                stats['active_days'] = len(stats['dates'])
                stats['display_name'] = self.get_user_display_name(user_id)
                # Remove sets for JSON serialization
                del stats['channels']
                del stats['dates']
        
        return dict(user_stats)
    
    def select_target_user(self, user_stats: Dict[str, Dict]) -> Optional[str]:
        """Interactive user selection"""
        print("\n" + "="*80)
        print("USER SELECTION FOR PERSONA TRAINING")
        print("="*80)
        
        # Filter users with sufficient activity
        active_users = {
            uid: stats for uid, stats in user_stats.items() 
            if stats['message_count'] >= 10  # Minimum threshold
        }
        
        if not active_users:
            print("No users found with sufficient activity (minimum 10 messages)")
            return None
        
        # Sort by message count
        sorted_users = sorted(
            active_users.items(), 
            key=lambda x: x[1]['message_count'], 
            reverse=True
        )
        
        print(f"\nFound {len(sorted_users)} users with sufficient activity:\n")
        print(f"{'#':<3} {'User ID':<15} {'Display Name':<20} {'Messages':<8} {'Channels':<8} {'Days':<6} {'Avg Length':<10}")
        print("-" * 80)
        
        for i, (user_id, stats) in enumerate(sorted_users[:20], 1):  # Show top 20
            print(f"{i:<3} {user_id:<15} {stats['display_name']:<20} "
                  f"{stats['message_count']:<8} {stats['channel_count']:<8} "
                  f"{stats['active_days']:<6} {stats['avg_message_length']:<10.1f}")
        
        if len(sorted_users) > 20:
            print(f"... and {len(sorted_users) - 20} more users")
        
        print("\nRecommendation: Choose a user with high message count and good average length (50+ chars)")
        
        while True:
            try:
                choice = input("\nEnter user number (1-{}) or 'q' to quit: ".format(len(sorted_users)))
                if choice.lower() == 'q':
                    return None
                    
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(sorted_users):
                    selected_user_id = sorted_users[choice_idx][0]
                    selected_stats = sorted_users[choice_idx][1]
                    
                    print(f"\nSelected user: {selected_stats['display_name']} ({selected_user_id})")
                    print(f"Messages: {selected_stats['message_count']}")
                    print(f"Active in {selected_stats['channel_count']} channels over {selected_stats['active_days']} days")
                    
                    confirm = input("Confirm selection? (y/n): ")
                    if confirm.lower() == 'y':
                        return selected_user_id
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number or 'q'.")
    
    def create_conversation_threads(self, target_user_id: str) -> List[Dict]:
        """Create conversation threads with context and target user responses"""
        print(f"\nCreating conversation threads for user: {self.get_user_display_name(target_user_id)}")
        
        threads = []
        
        # Group messages by channel and thread
        channel_threads = defaultdict(lambda: defaultdict(list))
        
        for msg in self.conversations:
            channel = msg.get('channel', 'general')
            thread_ts = msg.get('thread_ts', msg.get('ts'))
            channel_threads[channel][thread_ts].append(msg)
        
        # Process each thread
        for channel_name, channel_data in channel_threads.items():
            for thread_ts, messages in channel_data.items():
                if len(messages) < 2:  # Skip single-message threads
                    continue
                
                # Sort messages by timestamp
                messages.sort(key=lambda x: float(x.get('ts', 0)))
                
                # Find target user responses
                for i, msg in enumerate(messages):
                    if (msg.get('user') == target_user_id and 
                        msg.get('text') and 
                        len(msg.get('text', '').strip()) > 10):  # Minimum length
                        
                        # Get context (previous messages in thread)
                        context_messages = messages[:i]
                        if not context_messages:
                            continue
                        
                        # Create context string
                        context_parts = []
                        for ctx_msg in context_messages[-5:]:  # Last 5 messages for context
                            ctx_user = self.get_user_display_name(ctx_msg.get('user', 'Unknown'))
                            ctx_text = ctx_msg.get('text', '').strip()
                            if ctx_text:
                                context_parts.append(f"{ctx_user}: {ctx_text}")
                        
                        if context_parts:
                            context = "\n".join(context_parts)
                            response = msg.get('text', '').strip()
                            
                            thread_data = {
                                'channel': channel_name,
                                'thread_ts': thread_ts,
                                'message_ts': msg.get('ts'),
                                'context': context,
                                'response': response,
                                'date': msg.get('date'),
                                'user_id': target_user_id,
                                'user_name': self.get_user_display_name(target_user_id)
                            }
                            threads.append(thread_data)
        
        print(f"Created {len(threads)} conversation threads")
        return threads
    
    def create_training_dataset(self, threads: List[Dict], format_type: str = "alpaca") -> List[Dict]:
        """Create standardized training dataset"""
        print(f"Creating training dataset in {format_type} format...")
        
        dataset = []
        
        for thread in threads:
            if format_type == "alpaca":
                # Alpaca format (instruction, input, output)
                data_point = {
                    "instruction": "以下の対話の文脈に続いて、あなたらしく返信を生成してください。",
                    "input": thread['context'],
                    "output": thread['response']
                }
            elif format_type == "chat":
                # Chat format (messages array)
                data_point = {
                    "messages": [
                        {"role": "system", "content": "あなたは親しみやすく協力的なアシスタントです。以下の会話の文脈を理解し、自然で一貫性のある返信をしてください。"},
                        {"role": "user", "content": thread['context']},
                        {"role": "assistant", "content": thread['response']}
                    ]
                }
            elif format_type == "completion":
                # Completion format (prompt, completion)
                data_point = {
                    "prompt": f"対話の文脈:\n{thread['context']}\n\n返信:",
                    "completion": thread['response']
                }
            else:
                # Custom format with metadata
                data_point = {
                    "instruction": "以下の対話の文脈に続いて、あなたらしく返信を生成してください。",
                    "input": thread['context'],
                    "output": thread['response'],
                    "metadata": {
                        "channel": thread['channel'],
                        "date": thread['date'],
                        "user_name": thread['user_name']
                    }
                }
            
            dataset.append(data_point)
        
        print(f"Created {len(dataset)} training examples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_dir: str, target_user_id: str):
        """Save dataset in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        user_name = self.get_user_display_name(target_user_id)
        safe_user_name = "".join(c for c in user_name if c.isalnum() or c in ('-', '_'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_filename = f"slack_persona_{safe_user_name}_{timestamp}"
        
        # Save as JSON
        json_file = output_path / f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON dataset: {json_file}")
        
        # Save as JSONL (common for LLM training)
        jsonl_file = output_path / f"{base_filename}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved JSONL dataset: {jsonl_file}")
        
        # Save as CSV
        csv_file = output_path / f"{base_filename}.csv"
        df = pd.DataFrame(dataset)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved CSV dataset: {csv_file}")
        
        # Save statistics
        stats_file = output_path / f"{base_filename}_stats.json"
        stats = {
            "user_id": target_user_id,
            "user_name": user_name,
            "total_examples": len(dataset),
            "avg_context_length": sum(len(item.get('input', '')) for item in dataset) / len(dataset),
            "avg_response_length": sum(len(item.get('output', '')) for item in dataset) / len(dataset),
            "created_at": datetime.now().isoformat(),
            "format": "alpaca",
            "files": {
                "json": str(json_file.name),
                "jsonl": str(jsonl_file.name),
                "csv": str(csv_file.name)
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Saved statistics: {stats_file}")
        
        return {
            "json": json_file,
            "jsonl": jsonl_file,
            "csv": csv_file,
            "stats": stats_file
        }
    
    def process(self, output_dir: str = "./processed_datasets") -> Optional[Dict]:
        """Main processing pipeline"""
        print("Starting Slack data processing...")
        
        # Load metadata
        self.load_metadata()
        
        # Load conversations
        self.load_conversations()
        
        if not self.conversations:
            print("No conversations found. Please check your Slack export path.")
            return None
        
        # Analyze users
        user_stats = self.analyze_users()
        
        # Select target user
        target_user_id = self.select_target_user(user_stats)
        if not target_user_id:
            print("No user selected. Exiting.")
            return None
        
        # Create conversation threads
        threads = self.create_conversation_threads(target_user_id)
        
        if not threads:
            print("No suitable conversation threads found for the selected user.")
            return None
        
        # Create training dataset
        dataset = self.create_training_dataset(threads, format_type="alpaca")
        
        # Save dataset
        saved_files = self.save_dataset(dataset, output_dir, target_user_id)
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"User: {self.get_user_display_name(target_user_id)}")
        print(f"Training examples: {len(dataset)}")
        print(f"Output directory: {output_dir}")
        print("\nFiles created:")
        for format_name, file_path in saved_files.items():
            print(f"  {format_name.upper()}: {file_path}")
        
        return saved_files

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Process Slack export data for LoRA fine-tuning')
    parser.add_argument('slack_path', help='Path to Slack export directory')
    parser.add_argument('-o', '--output', default='./processed_datasets', 
                       help='Output directory for processed datasets')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.slack_path):
        print(f"Error: Slack export path '{args.slack_path}' does not exist.")
        sys.exit(1)
    
    processor = SlackDataProcessor(args.slack_path)
    result = processor.process(args.output)
    
    if result:
        print("\nYou can now use the generated CSV file in your LoRA training notebook!")
        print(f"CSV file: {result['csv']}")
    else:
        print("Processing failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()