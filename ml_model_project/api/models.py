import pickle
import numpy as np

from django.db import models
from keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences


class TextClassifier:
    def __init__(
            self,
            rating_model_path, tonality_model_path,
            tokenizer_path, encoder_path, max_len,
            threshold
            ):
        # Загрузка моделей
        self.rating_model = load_model(rating_model_path)
        self.tonality_model = load_model(tonality_model_path)

        self.max_len = max_len
        self.threshold = threshold

        # Загрузка токенизатора из JSON
        with open(tokenizer_path, 'r') as f:
            tokenizer_json = f.read()
        self.tokenizer = tokenizer_from_json(tokenizer_json)

        with open(encoder_path, 'rb') as file:
            self.encoder = pickle.load(file)
    
    def preprocess_text(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        pad_seq = pad_sequences(seq, maxlen=self.max_len)
        return pad_seq  # Измените это на реальную предобработку

    def predict_rating(self, text):
        # Предобработка текста для первой модели и получение предсказания
        # Например, преобразование текста в вектор или последовательность
        processed_text = self.preprocess_text(text)
        prediction = self.rating_model.predict(processed_text)
        predicted_class = np.argmax(prediction, axis=1)[0]
        prediction_transformed = self.encoder.inverse_transform(
            [predicted_class]
            )[0]
        # Получите класс с наибольшей вероятностью
        return prediction_transformed
        # Предполагается, что вы возвращаете индекс класса

    def predict_tonality(self, text):
        # Аналогично, для второй модели
        processed_text = self.preprocess_text(text)
        prediction = self.tonality_model.predict(processed_text)
        prediction = (prediction > self.threshold).astype(int)[0][0]
        return prediction
