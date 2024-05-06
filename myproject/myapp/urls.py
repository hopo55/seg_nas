from django.urls import path
from . import views

urlpatterns = [
    path("status/", views.training_status, name="training_status"),
    path(
        "start-training/", views.start_training, name="start_training"
    ),  # 학습 시작 URL
]
