import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class TelegramBot:
    credential_file: str = None

    def __init__(self) -> None:
        if Path(self.credential_file).exists():
            with open(self.credential_file) as file:
                auth_json = json.load(file)
            self.bot_token, self.chat_id = auth_json["token"], auth_json["chat_id"]

    def send_telegram_message(self, message: str) -> None:
        if not hasattr(self, "bot_token"):
            logger.warning(f"Credential not found! Message not sent.")
            return
        api_url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        try:
            response = requests.post(api_url, {'chat_id': self.chat_id, 'text': message})
            response.raise_for_status()
            logger.debug(f"Telegram message sent. Response: {response.text}")
        except Exception as error:
            logger.exception(f"An error occurred while sending message to telegram bot! Error: {error}")
