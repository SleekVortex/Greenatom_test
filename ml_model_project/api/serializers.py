from rest_framework import serializers

class TextPredictionSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)
    predicted_rating = serializers.IntegerField(read_only=True)
    predicted_tonality = serializers.IntegerField(read_only=True)