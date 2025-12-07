import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from base_classifier import BaseClassifier

class CNNClassifier(BaseClassifier):
    def __init__(self, image_size=(224, 224), batch_size=16, epochs=5, learning_rate=0.001, 
                 dropout_rate=0.5, class_names=None):
        super().__init__('cnn', image_size, class_names)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model = None
        
    def create_cnn_model(self):
        weights_path = os.path.expanduser('./weights/vgg16_weights_imagenet.h5')
        if os.path.exists(weights_path):
            base_model = VGG16(weights=weights_path, 
                              include_top=False, 
                              input_shape=(self.image_size[0], self.image_size[1], 3))
        else:
            base_model = VGG16(weights='imagenet', 
                              include_top=False, 
                              input_shape=(self.image_size[0], self.image_size[1], 3))
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        return model

    def preprocess_image_cnn(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype('float32') / 255.0
        
        return image

    def train(self, train_file, images_dir="."):
        from sklearn.metrics import accuracy_score
        
        train_paths, train_labels, label_ids = self.load_data(train_file, images_dir)
        
        if len(train_paths) == 0:
            print("Ошибка: нет данных для обучения!")
            return 0
        
        X_train = []
        y_train = []
        
        for (path, label_id) in zip(train_paths, label_ids):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X_train.append(image)
                y_train.append(label_id)
        
        X_train = np.array(X_train)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(self.class_names))
        
        self.model = self.create_cnn_model()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
        
        train_predictions = np.argmax(self.model.predict(X_train), axis=1)
        train_true_labels = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        
        print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
        return train_accuracy

    def test(self, test_file, images_dir="."):
        from sklearn.metrics import accuracy_score, classification_report
        
        test_paths, test_labels, label_ids = self.load_data(test_file, images_dir)
        
        if len(test_paths) == 0:
            print("Ошибка: нет данных для тестирования!")
            return 0
        
        X_test = []
        y_test = []
        
        for (path, label_id) in zip(test_paths, label_ids):
            image = self.preprocess_image_cnn(path)
            if image is not None:
                X_test.append(image)
                y_test.append(label_id)
        
        X_test = np.array(X_test)
        y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(self.class_names))
        
        test_predictions_proba = self.model.predict(X_test)
        test_predictions = np.argmax(test_predictions_proba, axis=1)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        test_predictions_labels = [self.class_names[pred] for pred in test_predictions]
        
        print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        print("\nДетальный отчет по классификации:")
        print(classification_report(test_labels, test_predictions_labels))
        
        return test_accuracy

    def save_model(self, model_dir="models"):
        super().save_model(model_dir)
        self.model.save(os.path.join(model_dir, 'cnn_model.h5'))

    def load_model(self, model_dir="models"):
        super().load_model(model_dir)
        self.model = tf.keras.models.load_model(os.path.join(model_dir, 'cnn_model.h5'))
        return True
