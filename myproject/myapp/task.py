from celery import shared_task
from .models import TrainingSession


@shared_task
def train(session_id):
    session = TrainingSession.objects.get(id=session_id)
    session.status = "in_progress"
    session.save()

    # 예시 학습 과정
    for epoch in range(5):  # 5 epoch 동안 학습을 시행한다고 가정
        time.sleep(1)  # 실제 학습 대신 간단한 대기 시간
        accuracy = epoch * 20  # 예시 정확도
        iou = epoch * 0.1  # 예시 IoU

        # 학습 상태 업데이트
        session.details = f"Epoch: {epoch + 1}, Accuracy: {accuracy}, IoU: {iou}"
        session.progress = (epoch + 1) / 5 * 100  # 진행률 업데이트
        session.save()

    session.status = "completed"
    session.save()


@shared_task
def start_training(session_id):
    session = TrainingSession.objects.get(id=session_id)
    session.status = "in_progress"
    session.save()
    # 여기에 머신러닝 학습 코드를 추가
    session.progress = 100.0  # 예시 진행률 업데이트
    session.status = "completed"
    session.save()
