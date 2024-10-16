from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
# from api.serializers import TextPredictionSerializer
from api.models import TextClassifier  # Обновите путь импортов
from api.serializers import TextPredictionSerializer
import os

# Путь к вашим моделям
RATING_MODEL_PATH = os.path.join('./model_params/custom_nn_rating.h5')  # Замените на свой путь
TONALITY_MODEL_PATH = os.path.join('./model_params/custom_nn_sentiment.h5')  # Замените на свой путь
TOKENIZER_PATH = os.path.join('./tokenizer_encoder/tokenizer.json')
ENCODER_PATH = os.path.join('./tokenizer_encoder/label_encoder.pkl')
MAX_LEN = 100
THRESHOLD = 0.5

# Инициализируем классификатор в начале
classifier = TextClassifier(
    RATING_MODEL_PATH,
    TONALITY_MODEL_PATH,
    TOKENIZER_PATH,
    ENCODER_PATH,
    MAX_LEN,
    THRESHOLD
    )

class TextPredictionAPIView(APIView):
    def post(self, request):
        serializer = TextPredictionSerializer(data=request.data)
        print(serializer)
        if serializer.is_valid():
            text = str(serializer.validated_data['text'])
            predicted_rating = classifier.predict_rating(text)
            predicted_tonality = classifier.predict_tonality(text)

            return Response({
                "text": text, 
                "predicted_rating": predicted_rating,
                "predicted_tonality": predicted_tonality
            }, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)