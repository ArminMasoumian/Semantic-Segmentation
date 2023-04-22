import tensorflow as tf
from model import SemanticSegmentationModel
from utils import load_data

# Load the data
x_train, y_train, x_val, y_val = load_data()

# Define the model
input_shape = x_train.shape[1:]
num_classes = y_train.shape[-1]
model = SemanticSegmentationModel(input_shape, num_classes)

# Compile the model
model.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# Save the model
model.model.save('model.h5')
