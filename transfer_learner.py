'''https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub'''
from functools import lru_cache
import numpy as np
import tensorflow as tf
class TransferLearner:

    def __init__(self):
        pass

    @property
    def preimage_layer(self):
        '''
        A loaded resnet model, use resnet_v2_152
        https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4
        '''

        url="https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4"
        if self._preimage_layer is None:
            self._preimage_layer = hub.KerasLayer(url, input_shape=(224, 224, 3), trainable=False)
        return self._preimage_layer
    
    @property
    def specific_network_layer(self):
        '''The network being trained, if not loaded from disk an undtrained dense network will be produced'''
        if self._specific_layer is None:
            self._specific_layer=self.init_specific_network()
        return self._specific_layer
        

    def build_full_model(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential([
        preimage_layer,
        self.specific_network_layer
        ])
        return model
    
    @property
    def full_model(self)-> tf.keras.Sequential:
        '''Combine the pretrained and transfer learning model into one network'''
        if self._full_model is None:
            self._full_model=self.build_full_model()
        return self._full_model

    def init_specific_network(self):
        num_classes = len(self.output_layer_vector)
        return tf.keras.layers.Dense(num_classes)


    def load_specific_network(self,path):
        '''The load the specific_network_layer being trained from file'''
        raise NotImplementedError()

    def checkpoint_specific_network(self)->none:
        '''Checkpoint the network being trained'''
        raise NotImplementedError()

    def save_specific_network(self,path):
        '''Extract only the specific_network_layer portion from full_model, then write a Keras-H5'''
        raise NotImplementedError()

    def compile_trainer_full_model(self):
        return self.full_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    def train_model(self,epochs=50):
        self._full_model_trainer.fit(self.dataset_provider, epochs=epochs)

    def evaluate_single_input(self,input_thumbnail) -> list:
        '''
        Returns an a non strict subset of ouput vector that the network thinks applies to the input thumbnail, and confidence.
        That is a list of the form [(tag,confidence)]
        '''
        raise NotImplementedError

    
    @lru_cache(1024)
    def input_thumbnail_for_image_key(self,image_key:str)->np.array:
        '''A thumbnail suitable for resnet'''
        raise NotImplementedError()

    @lru_cache(10240)
    def tags_for_image_key(self):
        raise NotImplementedError()

    def dataset_proivider(self):
        raise NotImplementedError()
        batch_size = 32
        img_height = 224
        img_width = 224

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


class ImageDatasetSource(tf.data.Dataset):
    #Fancier more intelligent image serving, I don't really know how to use this interface yet.
    def image_file_as_numpy_array(self,image_path:str) -> np.array:
        raise NotImplementedError()

    def path_for_image_key(self,image_key)->str:
        '''The filesystem path for the image tied to the specified key'''
        raise NotImplementedError()

    def output_layer_vector(self):
        '''Returns a fixed vector of strings that correspond to output excitations'''
        raise NotImplementedError()

    def validation_split(self):
        '''
        What perrcent of input items will be excluded from training input to be used for verifying how
        correctly the network evaluates
        '''
        return .2
