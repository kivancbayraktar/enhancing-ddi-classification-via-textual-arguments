import tensorflow as tf
from tensorflow import keras
# from keras import metrics
# from keras.models import Model, Sequential
# # from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
# from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization,InputLayer,Conv1D,MaxPooling1D,Flatten

event_num = 65
droprate = 0.3


def DNN(**params):
    input_shape = params.get('input_shape')
    num_classes = params.get('num_classes', event_num) 
    vector_size = input_shape[1]
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(vector_size,), name='Inputlayer'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(droprate))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(droprate))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
# ['accuracy',metrics.Precision() ]
    return model


def CNN(**params):
    # Set default values or get from params
    input_shape = params.get('input_shape')
    sequence_length = input_shape[1]
    embedding_dim = input_shape[2]
    # sequence_length = params.get('sequence_length', 2)  # default to 2
    # embedding_dim = params.get('embedding_dim', 2048)  # default to 2048
    num_classes = params.get('num_classes', event_num)  # default to 65 classes

    # Ensure parameters are correct types
    if not isinstance(sequence_length, int) or not isinstance(embedding_dim, int) or not isinstance(num_classes, int):
        raise ValueError(
            "sequence_length, embedding_dim, and num_classes must be integers")

    # Create the CNN model
    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, embedding_dim)),
        keras.layers.Conv1D(128, 1, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=1),
        keras.layers.Flatten(),
        # keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def CNN2(**params):
    # Set default values or get from params
    input_shape = params.get('input_shape')
    sequence_length = input_shape[1]
    embedding_dim = input_shape[2]
    # sequence_length = params.get('sequence_length', 2)  # default to 2
    # embedding_dim = params.get('embedding_dim', 2048)  # default to 2048
    num_classes = params.get('num_classes', event_num)  # default to 65 classes

    # Ensure parameters are correct types
    if not isinstance(sequence_length, int) or not isinstance(embedding_dim, int) or not isinstance(num_classes, int):
        raise ValueError(
            "sequence_length, embedding_dim, and num_classes must be integers")

    # Create the CNN model
    model = keras.Sequential([
        keras.layers.Input(
            shape=(sequence_length, embedding_dim)),  # Input layer

        # First Conv1D Layer
        # Larger kernel size and 'same' padding
        keras.layers.Conv1D(128, 3, activation='relu', padding='same'),

        # BatchNormalization for stable training
        keras.layers.BatchNormalization(),

        # MaxPooling1D layer
        # Increased pool size for better downsampling
        keras.layers.MaxPooling1D(pool_size=2),

        # Second Conv1D Layer (with dropout for regularization)
        # Increased number of filters
        keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),  # Dropout layer to avoid overfitting

        # Third Conv1D Layer
        # More filters for richer feature learning
        keras.layers.Conv1D(512, 3, activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        # Another max pooling to reduce spatial dimensions
        keras.layers.MaxPooling1D(pool_size=2),

        # Flatten layer to prepare for Dense layers
        keras.layers.Flatten(),

        # Fully connected layer (Dense)
        # Added a dense layer for better representation
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),  # Additional dropout for regularization

        # Output layer
        # Output layer with softmax for multi-class classification
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# def DNN_new(vector_size):
#     model = Sequential()
#     model.add(Input(shape=(vector_size,), name='Inputlayer'))
#     _mean = int((vector_size + 512) / 2)
#     model.add(Dense(_mean, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dense(512, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(droprate))
#     model.add(Dense(256, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(droprate))
#     model.add(Dense(event_num))
#     model.add(Activation('softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # ['accuracy',metrics.Precision() ]
#     return model
