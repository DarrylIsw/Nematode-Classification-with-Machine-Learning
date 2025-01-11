import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
dataset_types = ['morph']  # Dataset variations
save_dir = 'FIX-Values/ANN-FIX'  # Directory to save models and reports
os.makedirs(save_dir, exist_ok=True)

image_size = (224, 224)
batch_size = 16
num_classes = 2
num_folds = 5  # Number of folds for cross-validation

# Data generator
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Placeholder to store metrics across folds
fold_metrics = []
fold_train_losses = []
fold_val_losses = []

# Function to create the ANN model
def create_ann_model(input_shape=(224, 224, 3), num_classes=2):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Process each dataset type
for dataset_type in dataset_types:
    print(f"\nProcessing dataset-{dataset_type}...\n")

    # Set paths for the current dataset
    train_dir = f'dataset2/dataset-{dataset_type}/train'
    val_dir = f'dataset2/dataset-{dataset_type}/validation'
    test_dir = f'dataset2/dataset-{dataset_type}/test'

    # Load data generators
    train_gen = datagen.flow_from_directory(
        train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
    val_gen = datagen.flow_from_directory(
        val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
    test_gen = datagen.flow_from_directory(
        test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

    # Extract features and labels from the training set
    train_data, train_labels = train_gen.next()
    for _ in range(len(train_gen) - 1):
        data, labels = train_gen.next()
        train_data = np.vstack((train_data, data))
        train_labels = np.vstack((train_labels, labels))
    
    # Extract features and labels from the validation set (fixed)
    val_data, val_labels = val_gen.next()
    for _ in range(len(val_gen) - 1):
        data, labels = val_gen.next()
        val_data = np.vstack((val_data, data))
        val_labels = np.vstack((val_labels, labels))

    # Placeholder to store metrics across folds
    fold_metrics = []

    # K-Fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_num = 1

    for train_index, val_index in kf.split(train_data):
        print(f"\nFold {fold_num}:")

        # Split data for the current fold
        X_train, X_val_fold = train_data[train_index], train_data[val_index]
        y_train, y_val_fold = train_labels[train_index], train_labels[val_index]

        # Create model
        ann_model = create_ann_model(input_shape=(224, 224, 3), num_classes=num_classes)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train model
        history = ann_model.fit(
            X_train, y_train,
            validation_data=(val_data, val_labels),
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

        # Predict and evaluate
        y_val_pred = np.argmax(ann_model.predict(X_val_fold), axis=-1)
        y_val_true = np.argmax(y_val_fold, axis=-1)
        fold_report = classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys(), output_dict=True)

        print(f"\nValidation Classification Report for Fold {fold_num}:\n")
        print(classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys()))

        fold_metrics.append(fold_report)
        fold_num += 1

    # Calculate average metrics across folds
    avg_report = {}
    for label in ['graminicola', 'incognita']:
        avg_report[label] = {}
        for metric in ['precision', 'recall', 'f1-score']:
            avg_report[label][metric] = np.mean([fold[label][metric] for fold in fold_metrics])
    
    # Calculate and add averaged accuracy across folds
    avg_accuracy = np.mean([fold["accuracy"] for fold in fold_metrics])
    avg_report['average_accuracy'] = avg_accuracy  # Add the averaged accuracy to the report

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    history_test = ann_model.evaluate(test_gen, return_dict=True)
    test_accuracy = history_test['accuracy']
    test_loss = history_test['loss']

    y_test_pred = np.argmax(ann_model.predict(test_gen), axis=-1)
    y_test_true = test_gen.classes
    test_report = classification_report(y_test_true, y_test_pred, target_names=test_gen.class_indices.keys())
    conf_matrix_test = confusion_matrix(y_test_true, y_test_pred)

    # Save metrics to file
    report_file = os.path.join(save_dir, f"dataset-{dataset_type}-report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Dataset: dataset-{dataset_type}\n")
        f.write(f"\nAverage Metrics Across Folds:\n")
        for label, metrics in avg_report.items():
            if label == 'average_accuracy':
                f.write(f"Average Accuracy Across Folds: {metrics:.4f}\n")
            else:
                f.write(f"{label}: {metrics}\n")
        f.write(f"\nTest Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"\nTest Classification Report:\n{test_report}\n")
        f.write(f"\nTest Confusion Matrix:\n{conf_matrix_test}\n")
    
    # Calculate overall average learning curves
    max_epochs = max([len(losses) for losses in fold_train_losses])  # Find the maximum number of epochs across folds
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
    
    # Plot and save the overall learning curve
    plt.figure(figsize=(10, 6))
    plt.suptitle('Overall Learning Curve (Averaged Across Folds)', fontsize=16)
    plt.plot(avg_train_losses, label='Average Training Loss')
    plt.plot(avg_val_losses, label='Average Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Overall Learning Curve')
    plt.legend()
    plt.grid()
    overall_plot_path = os.path.join(save_dir, f'dataset-{dataset_type}-overall_learning_curve.png')
    plt.savefig(overall_plot_path)
    plt.close()
    print(f"Overall learning curve saved at {overall_plot_path}")
    
    
        # Save the trained model in .h5 format
    model_path = os.path.join(save_dir, f"ANN-{dataset_type}.h5")
    ann_model.save(model_path)
    print(f"Model for dataset-{dataset_type} saved to {model_path}")
