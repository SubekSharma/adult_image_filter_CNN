import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from keras.preprocessing.image import ImageDataGenerator

# Assuming you have your trained model and test data generator
model = tf.keras.models.load_model('model.h5')
test_dir = "test_data" # '../test_data'

test_datagen = ImageDataGenerator(rescale=1./255)

test_data_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(240, 320),
    batch_size=64, 
    class_mode='binary' ) 

# Retrieve class labels and indices from the data generator
class_labels = list(test_data_generator.class_indices.keys())
class_indices = test_data_generator.class_indices

# Define the threshold for classification
threshold = 0.5

# Obtain the true labels from the file paths in the generator
true_labels = []
for image_path in test_data_generator.filenames:
    class_name = image_path.split('/')[0]  # Assuming the directory structure is class_name/image.jpg
    true_labels.append(class_indices[class_name])

# Convert true labels to numpy array
true_labels = tf.convert_to_tensor(true_labels)

# Make predictions on the test data
predictions = model.predict(test_data_generator)
predicted_labels = tf.where(predictions >= threshold, 1, 0)

# Calculate precision, recall, and F1 score
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
