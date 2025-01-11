# Import libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import os
import pandas as pd

# Set paths for the dataset
train_dir = 'dataset/dataset-norm2/train'
val_dir = 'dataset/dataset-norm2/validation'
test_dir = 'dataset/dataset-norm2/test'
save_dir = 'ACCEPTED MODELS' # Modify this path to your liking
os.makedirs(save_dir, exist_ok=True)

# Parameters
image_size = (128, 128)
batch_size = 16
num_classes = 2
num_folds = 5  # Number of folds for cross-validation

# Data generators
datagen = ImageDataGenerator(rescale=1.0 / 255) 

train_gen = datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
val_gen = datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
test_gen = datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# Load training data to use with KFold
train_data, train_labels = train_gen.next()
for _ in range(len(train_gen) - 1):
    data, labels = train_gen.next()
    train_data = np.vstack((train_data, data))
    train_labels = np.vstack((train_labels, labels))

# Load validation data (for fixed validation across folds)
val_data, val_labels = val_gen.next()
for _ in range(len(val_gen) - 1):
    data, labels = val_gen.next()
    val_data = np.vstack((val_data, data))
    val_labels = np.vstack((val_labels, labels))

# Learning rate scheduler and early stopping
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define a function to create the transfer learning model
def create_transfer_learning_model():
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze base model layers
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)  # Output layer
    
    # Create model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize a model for training
model = create_transfer_learning_model()

# Perform K-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_num = 1
fold_metrics = []

for train_index, val_index in kf.split(train_data):
    print(f"\nFold {fold_num}:")

    # Split data for the current fold
    X_train, X_val_fold = train_data[train_index], train_data[val_index]
    y_train, y_val_fold = train_labels[train_index], train_labels[val_index]

    # Train the model on the fold
    history = model.fit(
        X_train, y_train,
        validation_data=(val_data, val_labels),  # fixed validation set
        epochs=30,  # Adjust the number of epochs as needed
        batch_size=batch_size,
        callbacks=[lr_schedule, early_stopping],
        verbose=1
    )

    # Predict on fold validation set and calculate metrics
    y_val_pred = np.argmax(model.predict(X_val_fold), axis=-1)
    y_val_true = np.argmax(y_val_fold, axis=-1)
    fold_report = classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys(), output_dict=True)

    # Print classification report for the fold
    print(f"\nValidation Classification Report for Fold {fold_num}:\n")
    print(classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys()))

    # Append metrics for averaging later
    fold_metrics.append(fold_report)
    fold_num += 1

# Calculate and display average metrics across all folds
avg_report = {}
for label in train_gen.class_indices.keys():
    avg_report[label] = {}
    for metric in ['precision', 'recall', 'f1-score']:
        avg_report[label][metric] = np.mean([fold[label][metric] for fold in fold_metrics])

avg_report['accuracy'] = {"precision": np.mean([fold["accuracy"] for fold in fold_metrics])}

# Save averaged metrics for cross-validation
df_report = pd.DataFrame.from_dict(avg_report, orient='index')
print("\nAveraged Classification Report over all folds:\n")
print(df_report)

# Save the final model after cross-validation
model_path = os.path.join(save_dir, "Transfer_Learn_VGG.h5")
model.save(model_path)
print(f"Final model saved to {model_path}")

# Uncomment the lines below if you prefer to save in SavedModel format instead of .h5
# model_path_savedmodel = os.path.join(save_dir, "Transfer_Learn_SavedModel")
# model.save(model_path_savedmodel, save_format="tf")
# print(f"Final model saved to {model_path_savedmodel} in SavedModel format")

# Final test evaluation on the last trained model
print("\nFinal Test Evaluation:")
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Generate classification report and confusion matrix for the test set
test_gen.reset()
y_test_pred = np.argmax(model.predict(test_gen), axis=-1)
y_test_true = test_gen.classes
print("\nTest Classification Report:\n", classification_report(y_test_true, y_test_pred, target_names=test_gen.class_indices.keys()))

# Display the final confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test_true, y_test_pred)
print("\nTest Confusion Matrix:\n", conf_matrix_test)
