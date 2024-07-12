# pylint:disable-all
import json
import tiktoken
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

class Metadata(BaseModel):
    is_visually_hidden_from_conversation: Optional[bool] = None
    request_id: Optional[str] = None
    message_source: Optional[Any] = None
    timestamp_: Optional[str] = None
    message_type: Optional[Any] = None
    model_slug: Optional[str] = None
    default_model_slug: Optional[str] = None
    parent_id: Optional[str] = None
    citations: Optional[List[Any]] = None
    gizmo_id: Optional[Any] = None
    finish_details: Optional[Dict[str, Any]] = None
    is_complete: Optional[bool] = None
    pad: Optional[str] = None

class Content(BaseModel):
    content_type: str
    parts: Optional[List[str]] = Field(default_factory=list)

class Author(BaseModel):
    role: str
    name: Optional[Any]
    metadata: Metadata

class Message(BaseModel):
    id: str
    author: Author
    create_time: Optional[float] = None
    update_time: Optional[float] = None
    content: Optional[Content] = None
    status: str
    end_turn: Optional[bool] = None
    weight: float
    metadata: Metadata
    recipient: str
    channel: Optional[Any]

class MappingItem(BaseModel):
    id: str
    message: Optional[Message] = None
    parent: Optional[str] = None
    children: List[str] = []

class Conversation(BaseModel):
    title: str
    create_time: float
    update_time: float
    mapping: Dict[str, MappingItem]
    moderation_results: List[Any]
    current_node: str
    plugin_ids: Optional[Any] = None
    conversation_id: str
    conversation_template_id: Optional[Any] = None
    gizmo_id: Optional[Any] = None
    is_archived: bool
    safe_urls: List[Any]
    default_model_slug: str
    conversation_origin: Optional[Any] = None
    voice: Optional[Any] = None
    id: str

# Load the JSON data from the file
with open('conversations.json', 'r') as file:
    json_data = json.load(file)

# Convert parsed JSON to Pydantic model instances
conversations = []
for item in json_data:
    try:
        conversations.append(Conversation(**item))
    except Exception as e:
        print(f"Error parsing item: {item}")
        print(e)

# Initialize arrays for storing user, assistant messages, token counts, message dates
user_messages = []
assistant_messages = []
user_token_counts = []
assistant_token_counts = []
message_dates = []

# Extract messages authored by users and assistants, and message dates
for conversation in conversations:
    for mapping_item in conversation.mapping.values():
        if mapping_item.message:
            # Extract content parts based on author role
            if mapping_item.message.content:
                if mapping_item.message.author.role == 'user':
                    user_messages.extend(mapping_item.message.content.parts or [])
                elif mapping_item.message.author.role == 'assistant':
                    assistant_messages.extend(mapping_item.message.content.parts or [])
            
            # Extract message date
            if mapping_item.message.create_time:
                create_time = mapping_item.message.create_time
                date = datetime.fromtimestamp(create_time, tz=timezone.utc).date()
                message_dates.append(date)

tokenizer = tiktoken.get_encoding("o200k_base")

# Calculate tokens for user messages
for message in user_messages:
    if message:  # Ensure the message is not empty
        tokens = tokenizer.encode(message)
        user_token_counts.append(len(tokens))

# Calculate tokens for assistant messages
for message in assistant_messages:
    if message:  # Ensure the message is not empty
        tokens = tokenizer.encode(message)
        assistant_token_counts.append(len(tokens))

# Number of unique days with messages
unique_dates = set(message_dates)
num_unique_days = len(unique_dates)

# Sum all token counts for each array
total_user_tokens = sum(user_token_counts)
total_assistant_tokens = sum(assistant_token_counts)
input_cost = total_user_tokens * 5/1000000
output_cost = total_assistant_tokens * 15/1000000
total_cost = input_cost + output_cost
average_cost_per_day = total_cost / num_unique_days
average_cost_per_month = average_cost_per_day * 365 / 12

# Display the total for each array
print("Total Input Tokens:", total_user_tokens)
print("Total Output Tokens:", total_assistant_tokens)
print("Total Input Cost:", input_cost)
print("Total Output Cost:", output_cost)
print("Total Cost:", total_cost)
print("Days of Use:", num_unique_days)
print("Average Cost Per Day:",average_cost_per_day)
print("Average Cost Per Month:", average_cost_per_month)