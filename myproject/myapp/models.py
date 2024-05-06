from django.db import models

# Create your models here.


class TrainingSession(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("in_progress", "In Progress"),
        ("completed", "Completed"),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    progress = models.FloatField(default=0.0)  # 학습 진행률
    details = models.TextField(blank=True)  # 학습 세부 정보

    def __str__(self):
        return f"Session {self.id} - {self.status}"
