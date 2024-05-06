from channels.generic.websocket import AsyncWebsocketConsumer
import json
from .models import TrainingSession


class TrainingStatusConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        await self.accept()
        await self.send_status_update()

    async def disconnect(self, close_code):
        pass

    async def send_status_update(self):
        session = TrainingSession.objects.get(id=self.session_id)
        await self.send(
            text_data=json.dumps(
                {"details": session.details, "progress": session.progress}
            )
        )
