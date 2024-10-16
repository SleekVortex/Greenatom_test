from django.urls import path
from api.views import TextPredictionAPIView

urlpatterns = [
    path('predict/', TextPredictionAPIView.as_view(), name='text-predict'),
]