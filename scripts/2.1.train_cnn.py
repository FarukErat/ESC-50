import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Disable GPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Add XLA fallback flag
os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

# Read metadata from CSV
meta_path = 'meta/esc50.csv'
df = pd.read_csv(meta_path)

# Split dataset using copy to avoid SettingWithCopyWarning
test_fold = 5
train_df = df[df['fold'] != test_fold].copy()
test_df = df[df['fold'] == test_fold].copy()

# Split training data
train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['target'],
    random_state=42
)

# Generate file paths for spectrogram images
def get_spectrogram_path(filename):
    return os.path.join('spectrograms', filename.replace('.wav', '_spectrogram.png'))

# Convert target to string for categorical mode
for dataframe in [train_df, val_df, test_df]:
    dataframe['image_path'] = dataframe['filename'].apply(get_spectrogram_path)
    dataframe['target'] = dataframe['target'].astype(str)  # Convert to string

# Configure data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
batch_size = 32
target_size = (224, 224)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='target',
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='target',
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col='target',
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(50, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest Accuracy: {test_acc:.2%}')
