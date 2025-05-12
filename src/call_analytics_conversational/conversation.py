"""
Conversation analysis module for Call Analytics - Conversational Intelligence.
Provides functions for analyzing call conversations using Azure Language Service.
"""

import json
from typing import Dict, List, Optional

from .settings import (
    PATH_AIRCALL_SENTIMENTS,
    PATH_AIRCALL_SUMMARIES,
    PATH_AIRCALL_TOPICS,
    PATH_AIRCALL_TRANSCRIPTIONS,
)


class Conversation:
    """
    A class to load and store conversation data from multiple JSON files.
    Redacts PII from the transcription using Azure Text Analytics.
    """

    def __init__(
        self,
        convo_id,
    ):
        """
        Initialize a Conversation instance by loading JSON files.

        Parameters:
            convo_id (str): Unique conversation identifier.
            azure_key (str, optional): Azure Text Analytics subscription key.
            azure_endpoint (str, optional): Azure Text Analytics endpoint.
        """
        self.convo_id = convo_id
        self.summary = self._load_json(f"data/aircall/summary/{convo_id}.json")
        self.topics = self._load_json(f"data/aircall/topics/{convo_id}.json")
        self.transcription = self._load_json(
            f"data/aircall/transcription/{convo_id}.json"
        )
        self.sentiments = self._load_json(f"data/aircall/sentiments/{convo_id}.json")

        self.summary_str = self._format_summary()
        self.topics_str = self._format_topics()
        self.transcription_str = self._format_transcription()
        self.sentiments_str = self._format_sentiments()

    def _load_json(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": f"Could not load {file_path}"}

    def _format_summary(self):
        return self.summary.get("summary", {}).get("content", "")

    def _format_topics(self):
        topics_list = self.topics.get("topic", {}).get("content", [])
        return "\n".join(topics_list) if topics_list else ""

    def _format_sentiments(self):
        participants = self.sentiments.get("sentiment", {}).get("participants", [])
        return participants[0].get("value", "") if participants else ""

    def _format_transcription(self):
        utterances = (
            self.transcription.get("transcription", {})
            .get("content", {})
            .get("utterances", [])
        )
        if not utterances:
            return ""

        external_participants = {}
        external_counter = 2

        def get_participant_label(utterance):
            nonlocal external_counter
            participant_type = utterance.get("participant_type")

            if participant_type == "internal":
                return "AGENT_1"

            if participant_type == "external":
                phone_number = utterance.get("phone_number")
                if phone_number not in external_participants:
                    label = (
                        "CUSTOMER"
                        if "CUSTOMER" not in external_participants.values()
                        else f"AGENT_{external_counter}"
                    )
                    external_participants[phone_number] = label
                    if label != "CUSTOMER":
                        external_counter += 1
                return external_participants[phone_number]

            return "UNKNOWN"

        consolidated = []
        last_speaker, buffer = None, []

        for utterance in utterances:
            speaker = get_participant_label(utterance)
            text = utterance.get("text", "")

            if speaker == last_speaker:
                buffer.append(text)
            else:
                if buffer:
                    consolidated.append(f"{last_speaker}:\n" + " ".join(buffer))
                buffer = [text]
                last_speaker = speaker

        if buffer:
            consolidated.append(f"{last_speaker}:\n" + " ".join(buffer))

        return "\n\n".join(consolidated)
