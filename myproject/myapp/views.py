from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from myapp.models import TrainingSession
from myapp.task import train
from django.shortcuts import render

def training_status(request):
    # 여기에 로직 구현
    return render(request, 'myapp/status.html')

@csrf_exempt
def start_training(request):
    if request.method == "POST":
        # 새로운 학습 세션을 생성하고, 학습 작업을 시작합니다.
        session = TrainingSession.objects.create()
        train.delay(session.id)  # Celery를 사용하여 비동기적으로 학습을 시작합니다.
        return JsonResponse({"session_id": session.id})
    return JsonResponse({"error": "Invalid request"}, status=400)
