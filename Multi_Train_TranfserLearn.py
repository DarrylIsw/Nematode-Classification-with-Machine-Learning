import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet201, ResNet152V2, EfficientNetV2L, VGG19, ConvNeXtLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
image_size = (224, 224)
batch_size = 16
num_classes = 2
num_folds = 5
save_dir = 'MODELS-TO-BE-SENT'
os.makedirs(save_dir, exist_ok=True)

# Dataset variations
dataset_variations = ['norm', 'brigh', 'blur', 'noise', 'flip', 'cont', 'overall']
fold_metrics = []
fold_train_losses = []
fold_val_losses = []

# Learning rate scheduler and early stopping
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# List of models to iterate through
model_list = {
    #'EfficientNetV2L': EfficientNetV2L,
    #'ResNet152V2': ResNet152V2,
    #'VGG19': VGG19,
    'DenseNet201': DenseNet201,
    #'ConvNeXtLarge': ConvNeXtLarge
}

# Function to create a transfer learning model based on the base model
def create_transfer_learning_model(base_model_func):
    base_model = base_model_func(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model layers

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Automate training for each model and dataset variation
for model_name, base_model_func in model_list.items():
    print(f"\nTraining with base model: {model_name}\n")

    for dataset_type in dataset_variations:
        print(f"\nTraining on dataset-{dataset_type} with {model_name}...\n")

        # Set dataset paths
        train_dir = f'dataset2/dataset-{dataset_type}/train'
        val_dir = f'dataset2/dataset-{dataset_type}/validation'
        test_dir = f'dataset2/dataset-{dataset_type}/test'

        # Data generators
        datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_gen = datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
        val_gen = datagen.flow_from_directory(val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
        test_gen = datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

        # Load training data for KFold
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

        # Initialize a model for training
        model = create_transfer_learning_model(base_model_func)

        # Perform K-fold cross-validation
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_num = 1
        fold_metrics = []

        for train_index, val_index in kf.split(train_data):
            print(f"\nFold {fold_num} with {model_name}:")
            X_train, X_val_fold = train_data[train_index], train_data[val_index]
            y_train, y_val_fold = train_labels[train_index], train_labels[val_index]

            # Train the model on the fold
            history = model.fit(
                X_train, y_train,
                validation_data=(val_data, val_labels),  # Fixed validation set
                epochs=30,
                batch_size=batch_size,
                callbacks=[lr_schedule, early_stopping],
                verbose=1
            )

            # Store training and validation losses for this fold
            fold_train_losses.append(history.history['loss'])
            fold_val_losses.append(history.history['val_loss'])

            # Plot and save fold-specific learning curve
            plt.figure(figsize=(10, 6))
            plt.suptitle(f'Learning Curve for Fold {fold_num} ({model_name}, dataset-{dataset_type})', fontsize=16)
            plt.plot(history.history['loss'], label=f'Training Loss (Fold {fold_num})')
            plt.plot(history.history['val_loss'], label=f'Validation Loss (Fold {fold_num})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Learning Curve for Fold {fold_num}')
            plt.legend()
            plt.grid()

            # Save the plot with model and dataset-specific naming
            fold_plot_path = os.path.join(save_dir, f"{model_name}-dataset-{dataset_type}-fold_{fold_num}_learning_curve.png")
            plt.savefig(fold_plot_path)
            plt.close()
            print(f"Learning curve for Fold {fold_num} ({model_name}) saved at {fold_plot_path}")

            # Predict on fold validation set
            y_val_pred = np.argmax(model.predict(X_val_fold), axis=-1)
            y_val_true = np.argmax(y_val_fold, axis=-1)
            fold_report = classification_report(y_val_true, y_val_pred, target_names=train_gen.class_indices.keys(), output_dict=True)

            fold_metrics.append(fold_report)
            fold_num += 1


        # Calculate the maximum number of epochs across all folds
        max_epochs = max([len(losses) for losses in fold_train_losses])

        # Initialize arrays to store average losses
        avg_train_losses = np.zeros(max_epochs)
        avg_val_losses = np.zeros(max_epochs)

        # Aggregate losses for each epoch across all folds
        for fold in range(len(fold_train_losses)):
            for epoch in range(len(fold_train_losses[fold])):
                avg_train_losses[epoch] += fold_train_losses[fold][epoch]
                avg_val_losses[epoch] += fold_val_losses[fold][epoch]

        # Normalize by the number of folds
        avg_train_losses /= num_folds
        avg_val_losses /= num_folds

        # Plot and save the overall learning curve
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Overall Learning Curve ({model_name}, dataset-{dataset_type})', fontsize=16)
        plt.plot(avg_train_losses, label='Average Training Loss')
        plt.plot(avg_val_losses, label='Average Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Overall Learning Curve')
        plt.legend()
        plt.grid()

        # Save the plot with model and dataset-specific naming
        overall_plot_path = os.path.join(save_dir, f"{model_name}-dataset-{dataset_type}-overall_learning_curve.png")
        plt.savefig(overall_plot_path)
        plt.close()
        print(f"Overall learning curve for {model_name} saved at {overall_plot_path}")

        avg_report = {}
        for label in train_gen.class_indices.keys():
            avg_report[label] = {}
            for metric in ['precision', 'recall', 'f1-score']:
                avg_report[label][metric] = np.mean([fold[label][metric] for fold in fold_metrics])
        avg_report['accuracy'] = {"value": np.mean([fold["accuracy"] for fold in fold_metrics])}

        # Test on the final model
        test_loss, test_accuracy = model.evaluate(test_gen)
        y_test_pred = np.argmax(model.predict(test_gen), axis=-1)
        y_test_true = test_gen.classes
        conf_matrix_test = confusion_matrix(y_test_true, y_test_pred)

        # Save results to text file with model-specific naming
        report_path = os.path.join(save_dir, f"dataset-{dataset_type}-{model_name}-report.txt")
        with open(report_path, "w") as report_file:
            report_file.write(f"Dataset: dataset-{dataset_type}\n")
            report_file.write(f"Model: {model_name}\n\n")
            report_file.write(f"Averaged Metrics:\n")
            for label, metrics in avg_report.items():
                if isinstance(metrics, dict):
                    report_file.write(f"  {label}:\n")
                    for metric_name, metric_value in metrics.items():
                        report_file.write(f"    {metric_name}: {metric_value:.4f}\n")
            report_file.write(f"\nTest Accuracy: {test_accuracy:.4f}\n\n")
            report_file.write(f"Confusion Matrix:\n")
            for row in conf_matrix_test:
                report_file.write(" ".join(map(str, row)) + "\n")
        
        model_path = os.path.join(save_dir, f"{model_name}-{dataset_type}.h5")
        model.save(model_path)
        print(f"Model for dataset-{dataset_type} saved to {model_path}")
        print(f"Results for dataset-{dataset_type} with {model_name} saved to {report_path}")