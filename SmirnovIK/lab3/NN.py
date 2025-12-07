from tensorflow.data import Dataset
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
class NN:
    def __init__(self, model_name=None):
        self.base_model = None
        if model_name == 'VGG':
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    def train_nn(self,train_images,train_labels):
        num_classes = len(np.unique(train_labels))
        train_labels_oh = to_categorical(train_labels, num_classes)
        data_train = Dataset.from_tensor_slices((train_images, train_labels_oh))
        data_train = data_train.shuffle(buffer_size=len(train_images)).batch(32)

        for layer in self.base_model.layers:
            layer.trainable = False

        num_classes = len(np.unique(train_labels))
        x = Flatten()(self.base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=self.base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(data_train, epochs=5)
        model.save('model.keras')
    def load(self, model_path):
        self.base_model = load_model(model_path)
    def test_nn(self,test_images, test_labels=None):
        data_test = Dataset.from_tensor_slices((test_images)).batch(32)
        preds = np.argmax(self.base_model.predict(data_test),axis=1)
        if test_labels is not None:
            acc = accuracy_score(test_labels, preds)
            return acc, preds
        return preds