
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# setting datapath
dataset_path = 'C:\\Users\\user\\Desktop\\Python\\Machine Learning\\Cow dataset'  # e.g., 'C:/Users/Lenny/Desktop/animal_dataset'

# creating image generator to load images & split into train/val sets

datagen = ImageDataGenerator(
    rescale=1./255,   # normalizing image pixel values
    validation_split=0.2 # use 20% of data for validation
)

# create training generator

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

print("Class indices:", train_generator.class_indices)
print("Total training samples:", train_generator.samples)


# create validation generator

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

print("Total validation samples:", val_generator.class_indices)
print("Total validation samples:", val_generator.samples)



# ................................................................
#  Build Resnet using transfer learning
# ................................................................

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Loading base ResNet50 model without classification layer 

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)  # ✅ Correct shape
)


# # Freezing base model layers

# for layer in base_model.layers:
#     layer.trainable = False

# Unfreeze last few layers
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True


# Add custom classification layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Creating and compiling the model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#........................................
# train the model
#.................................

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)

#........................................
# Evaluate and Save the Model
#.................................

# Evaluate on the validation set (or create a separate test set if you want)
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation accuracy: {val_acc:.2f}")

# Save the trained model
model.save("livestock_disease_model.keras")  # modern format


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predict on val set
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(val_generator.classes, y_pred))

print('Classification Report')
target_names = list(val_generator.class_indices.keys())
print(classification_report(val_generator.classes, y_pred, target_names=target_names))
