import json
import logging

import requests

logger = logging.getLogger(__name__)


class TelegramBot:
    credential_file = None  # Path file.

    def __init__(self) -> None:
        if self.credential_file:
            auth_json = json.loads(self.credential_file.read_text())
            self.bot_token = auth_json["token"]
            self.chat_id = auth_json["chat_id"]

    def send_telegram_message(self, message: str) -> None:
        api_url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        data = {'chat_id': self.chat_id, 'text': message}
        try:
            response = requests.post(api_url, data=data)
            response.raise_for_status()
            logger.debug(f"Telegram message sent. Response: {response.text}")
        except Exception as error:
            logger.exception(f"An error occurred while sending message to telegram bot! Error: {error}")
