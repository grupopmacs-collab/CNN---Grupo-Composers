# Fase 4: Modelado con CNN para Detección de Fraude - VERSIÓN OPTIMIZADA

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("🚀 FASE 4 OPTIMIZADA - ENTRENAMIENTO RÁPIDO")
print(f"TensorFlow version: {tf.__version__}")

# Cargar dataset MNIST
print("📥 Cargando dataset MNIST...")
(X_train, y_train_original), (X_test, y_test_original) = keras.datasets.mnist.load_data()

# Mapear a fraude (0-4: legítimo, 5-9: fraude)
y_train = (y_train_original >= 5).astype(int)
y_test = (y_test_original >= 5).astype(int)

# PREPROCESAMIENTO OPTIMIZADO
print("\n🔄 Preprocesamiento optimizado...")

# Reducir tamaño de imagen para mayor velocidad (de 64x64 a 32x32)
def preprocess_images_fast(images):
    images = images.astype('float32') / 255.0
    images_resized = tf.image.resize(images[..., tf.newaxis], [32, 32])
    return images_resized.numpy()

X_train_processed = preprocess_images_fast(X_train)
X_test_processed = preprocess_images_fast(X_test)

# Tomar una muestra más pequeña para entrenamiento rápido
def get_balanced_sample(X, y, sample_size=10000):
    legit_indices = np.where(y == 0)[0]
    fraud_indices = np.where(y == 1)[0]
    
    n_each = sample_size // 2
    legit_sampled = np.random.choice(legit_indices, n_each, replace=False)
    fraud_sampled = np.random.choice(fraud_indices, n_each, replace=False)
    
    balanced_indices = np.concatenate([legit_sampled, fraud_sampled])
    np.random.shuffle(balanced_indices)
    
    return X[balanced_indices], y[balanced_indices]

print("⚖️ Creando dataset balanceado y más pequeño...")
X_balanced, y_balanced = get_balanced_sample(X_train_processed, y_train, sample_size=10000)

print(f"📐 Dimensiones optimizadas:")
print(f"Entrenamiento: {X_balanced.shape}")
print(f"Prueba: {X_test_processed.shape}")

# MODELO MÁS SIMPLE Y RÁPIDO
print("\n🧠 Construyendo modelo CNN optimizado...")

def create_fast_cnn_model(input_shape):
    model = keras.Sequential([
        # Bloque convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Bloque convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Bloque convolucional 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Capas fully connected reducidas
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Salida
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Crear modelo optimizado
input_shape = (32, 32, 1)
model = create_fast_cnn_model(input_shape)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("✅ Modelo optimizado construido:")
model.summary()

# CALLBACKS MEJORADOS
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Menos paciencia para terminar antes
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# DIVISIÓN Y ENTRENAMIENTO RÁPIDO
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"\n📊 Divisiones optimizadas:")
print(f"Entrenamiento: {X_train_final.shape}")
print(f"Validación: {X_val.shape}")
print(f"Prueba: {X_test_processed.shape}")

# ENTRENAMIENTO ACELERADO
print("\n⚡ INICIANDO ENTRENAMIENTO ACELERADO...")

# Usar batch size más grande para mayor velocidad
history = model.fit(
    X_train_final, y_train_final,
    batch_size=128,  # Batch size más grande
    epochs=15,       # Menos épocas
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("✅ Entrenamiento completado!")

# EVALUACIÓN RÁPIDA
print("\n📊 Evaluación rápida del modelo...")

test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    X_test_processed, y_test, verbose=0, batch_size=256
)

print("🎯 RESULTADOS OBTENIDOS:")
print("=" * 40)
print(f"Exactitud: {test_accuracy:.4f}")
print(f"Precisión: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"AUC: {test_auc:.4f}")

# Predicciones rápidas
y_pred_proba = model.predict(X_test_processed, verbose=0, batch_size=256)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Métricas rápidas
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)

print(f"F1-Score: {f1:.4f}")

# GUARDAR MODELO
model.save('fraud_detection_fast_model.h5')
print("\n💾 Modelo guardado: 'fraud_detection_fast_model.h5'")

# ANÁLISIS RÁPIDO DE RESULTADOS
print("\n🔍 ANÁLISIS RÁPIDO:")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Transacciones analizadas: {len(y_test):,}")
print(f"Fraudes detectados: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"Falsas alarmas: {fp}/{fp+tn} ({fp/(fp+tn)*100:.4f}%)")

if test_recall >= 0.85 and test_precision >= 0.80:
    print("🎉 ¡MODELO ÓPTIMO PARA PRODUCCIÓN!")
else:
    print("✅ Modelo funcional - Puede mejorarse con más tiempo")

print("\n⏱️  TIEMPO ESTIMADO DE ENTRENAMIENTO:")
print("   - Original: ~7 horas (30 épocas)")
print("   - Optimizado: ~5-15 minutos")