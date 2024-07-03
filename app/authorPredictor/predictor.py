import tensorflow as tf
import json


class Predictor:
    __stop_words = None
    __user_names = None
    model = None
    tokenizer = None

    def __init__(self, model_path, tokenizer_json_path, stopwords_list, usernames_json_list):
        self.__load_model(model_path)
        self.__load_tokenizer_json(tokenizer_json_path)
        self.__load_stopwords(stopwords_list)
        self.__load_usernames_json(usernames_json_list)

    def predict(self, text):
        text = self.__clean_text(text)

        sequences = self.tokenizer.texts_to_sequences([text])

        padded = tf.keras.utils.pad_sequences(sequences, maxlen=250)

        prediction = self.model.predict(padded)
        preds = []

        for i, user in enumerate(self.__user_names):
            preds.append([user, prediction[0][i] * 100])

        return preds

    def predict_anon(self, text):
        text = self.__clean_text(text)

        sequences = self.tokenizer.texts_to_sequences([text])

        padded = tf.keras.utils.pad_sequences(sequences, maxlen=250)

        prediction = self.model.predict(padded)

        return prediction

    def getUsersCount(self):
        return len(self.__user_names)

    def getUsersList(self):
        return self.__user_names

    def __load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def __load_tokenizer_json(self, tokenizer_path):
        with open(tokenizer_path, 'rb') as tokenizerFile:
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                tokenizerFile.read())

    def __load_stopwords(self, stopwords):
        self.__stop_words = stopwords

    def __load_usernames_json(self, usernames_path):
        with open(usernames_path) as users_file:
            temp = json.load(users_file)
            self.__user_names = temp['participants']

    def __clean_text(self, message):
        message = message.replace(".", "")
        message = message.replace(",", "")
        message = message.replace(";", "")
        message = message.replace("!", "")
        message = message.replace("?", "")
        message = message.replace("(", "")
        message = message.replace(")", "")
        message = message.replace("\\", "")
        message = message.replace("\"", "")

        # remove discord effects
        message = message.replace("*", "")
        message = message.replace("_", "")
        message = message.replace("~", "")
        message = message.replace("`", "")
        message = message.replace(">", "")
        message = message.replace("<", "")
        message = message.replace("||", "")
        message = message.replace("```", "")
        message = message.replace("~~", "")
        message = message.replace(":", "")
        message = message.replace("#", "")
        message = message.replace("@", "")

        # remove stopwords
        message = message.lower()
        message = ' '.join([word for word in message.split(
        ) if word not in self.__stop_words or word in self.__user_names])

        return message
