# Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set paths
train_dir = 'dataset2/dataset-morph/train'
val_dir = 'dataset2/dataset-morph/validation'
test_dir = 'dataset2/dataset-morph/test'
save_dir = 'FIX-Values/ANN2' # Modify this path to your liking
os.makedirs(save_dir, exist_ok=True)

# Parameters
image_size = (224, 224)
batch_size = 16
num_classes = 2
num_folds = 5  # Single 5-fold cross-validation

# Data generator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load data for cross-validation
train_gen = datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
test_gen = datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# Extract features and labels from the training set
train_data, train_labels = train_gen.next()
for _ in range(len(train_gen) - 1):
    data, labels = train_gen.next()
    train_data = np.vstack((train_data, data))
    train_labels = np.vstack((train_labels, labels))

# Load fixed validation data
val_gen = datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

val_data, val_labels = val_gen.next()
for _ in range(len(val_gen) - 1):
    data, labels = val_gen.next()
    val_data = np.vstack((val_data, data))
    val_labels = np.vstack((val_labels, labels))


# Placeholder to store metrics across folds
fold_metrics = []
fold_train_losses = []
fold_val_losses = []
fold_metrics = []

# Run a single round of 5-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_num = 1

for train_index, val_index in kf.split(train_data):
    print(f"\nFold {fold_num}:")

    # Split data for the current fold
    X_train, X_val_fold = train_data[train_index], train_data[val_index]
    y_train, y_val_fold = train_labels[train_index], train_labels[val_index]
    
    # Split data for the current fold
    # X_train, _ = train_data[train_index], train_data[val_index]  # Use only training data
    # y_train, _ = train_labels[train_index], train_labels[val_index]

    
    # Create the ANN model for the fold
    def create_ann_model(input_shape=(224, 224, 3), num_classes=2):
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=3e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    ann_model = create_ann_model(input_shape=(224, 224, 3), num_classes=num_classes)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model with fixed validation data
    history = ann_model.fit(
        X_train, y_train,
        validation_data=(X_val_fold, y_val_fold ), # Change with X_val_fold, Y_val_fold to change back to validating with training data
        epochs=30,
        callbacks=[early_stopping],
        batch_size=batch_size,
        verbose=1
    )

    # Store fold-wise training and validation losses
    fold_train_losses.append(history.history['loss'])
    fold_val_losses.append(history.history['val_loss'])
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.suptitle('ANN Model Performance (Fold ' + str(fold_num) + ')', fontsize=16)
    plt.plot(history.history['loss'], label='Training Loss (Fold ' + str(fold_num) + ')')
    plt.plot(history.history['val_loss'], label='Validation Loss (Fold ' + str(fold_num) + ')') 
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(save_dir, f'fold_{fold_num}_learning_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning curve for Fold {fold_num} saved at {plot_path}")


    # # Plot training & validation accuracy values
    # plt.figure(figsize=(12, 5))
# 
    # # Plot accuracy
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='Train Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title(f'Fold {fold_num} - Model Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='lower right')
# 
    # # Plot loss
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title(f'Fold {fold_num} - Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc='upper right')
# 
    # plt.show()

    # Predict on fold validation set and calculate metrics
    # y_val_pred = np.argmax(ann_model.predict(X_val_fold), axis=-1)
    # y_val_true = np.argmax(y_val_fold, axis=-1)
    # fold_report = classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys(), output_dict=True)
# 
    # # Print classification report and store metrics for averaging
    # print(f"\nValidation Classification Report for Fold {fold_num}:\n")
    # print(classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys()))
    # fold_metrics.append(fold_report)
# 
    # fold_num += 1
    
    # Predict on validation set
    # y_val_pred = np.argmax(ann_model.predict(val_data), axis=-1)
    # y_val_true = np.argmax(val_labels, axis=-1)
    
    # Predict on fold validation set and calculate metrics
    y_val_pred = np.argmax(ann_model.predict(X_val_fold), axis=-1)
    y_val_true = np.argmax(y_val_fold, axis=-1)
    fold_report = classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys(), output_dict=True)

    # Calculate and print classification metrics
    fold_report = classification_report(y_val_true, y_val_pred, target_names=val_gen.class_indices.keys(), output_dict=True)
    print(f"\nValidation Classification Report for Fold {fold_num}:\n")
    print(classification_report(y_val_true, y_val_pred, target_names=val_gen.class_indices.keys()))
    fold_metrics.append(fold_report)

    fold_num += 1


# Calculate overall average learning curves
max_epochs = max([len(losses) for losses in fold_train_losses])
avg_train_losses = np.zeros(max_epochs)
avg_val_losses = np.zeros(max_epochs)

# Aggregate losses for each epoch across all folds
for fold in range(num_folds):
    for epoch in range(len(fold_train_losses[fold])):
        avg_train_losses[epoch] += fold_train_losses[fold][epoch]
        avg_val_losses[epoch] += fold_val_losses[fold][epoch]

# Divide by the number of folds to get the average
avg_train_losses /= num_folds
avg_val_losses /= num_folds

# Plot the overall learning curve
plt.figure(figsize=(10, 6))
plt.suptitle('ANN Model Performance Summary', fontsize=16)
plt.plot(avg_train_losses, label='Average Training Loss')
plt.plot(avg_val_losses, label='Average Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Overall Learning Curve (Averaged Across Folds)')
plt.legend()
plt.grid()
overall_plot_path = os.path.join(save_dir, 'overall_learning_curve.png')
plt.savefig(overall_plot_path)
plt.close()
print(f"Overall learning curve saved at {overall_plot_path}")

# Calculate average metrics across all folds
avg_report = {}
for label in train_gen.class_indices.keys():
    avg_report[label] = {}
    for metric in ['precision', 'recall', 'f1-score']:
        avg_report[label][metric] = np.mean([fold[label][metric] for fold in fold_metrics])

# Display the average classification report across folds
df_report = pd.DataFrame.from_dict(avg_report, orient='index')
print("\nAveraged Classification Report over all folds:\n")
print(df_report)

# Save the trained model
model_path = os.path.join(save_dir, 'ANN-test-morph.h5')
ann_model.save(model_path)
print(f"Final model saved to {model_path}")

# Final test evaluation and plots
print("\nFinal Test Evaluation:")
history_test = ann_model.evaluate(test_gen, return_dict=True)
print(f"\nTest Loss: {history_test['loss']:.4f}, Test Accuracy: {history_test['accuracy']:.4f}")

# Generate classification report and plot confusion matrix for the test set
test_gen.reset()
y_test_pred = np.argmax(ann_model.predict(test_gen), axis=-1)
y_test_true = test_gen.classes
print("\nTest Classification Report:\n", classification_report(y_test_true, y_test_pred, target_names=test_gen.class_indices.keys()))
