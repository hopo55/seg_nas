from django.urls import path
from myproject.myapp.consumers import TrainingStatusConsumer

websocket_urlpatterns = [
    path("ws/status/<int:session_id>/", TrainingStatusConsumer.as_asgi()),
]
