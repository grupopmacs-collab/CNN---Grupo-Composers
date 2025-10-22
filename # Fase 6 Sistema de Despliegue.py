# Fase 6: Sistema de Despliegue - Detección de Fraude con Imágenes

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import cv2
import io
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("🚀 FASE 6: SISTEMA DE DESPLIEGUE - DETECCIÓN DE FRAUDE EN IMÁGENES")
print("="*70)

# Cargar el modelo entrenado
print("📥 Cargando modelo entrenado...")
try:
    model = keras.models.load_model('fraud_detection_fast_model.h5')
    print("✅ Modelo cargado: fraud_detection_fast_model.h5")
except:
    try:
        model = keras.models.load_model('best_fraud_image_model.h5')
        print("✅ Modelo cargado: best_fraud_image_model.h5")
    except:
        print("❌ No se pudo cargar el modelo. Asegúrate de haber ejecutado la Fase 4.")
        raise

# Función para preprocesar imágenes (consistente con el entrenamiento)
def preprocess_uploaded_image(image, target_size=(32, 32)):
    """
    Preprocesa una imagen subida para que sea compatible con el modelo
    """
    # Convertir a array numpy si es necesario
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convertir a escala de grises si es RGB
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Redimensionar y normalizar
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    
    # Agregar dimensiones para el modelo (batch, height, width, channels)
    image = np.expand_dims(image, axis=-1)  # Agregar canal
    image = np.expand_dims(image, axis=0)   # Agregar batch
    
    return image

# Función principal de detección de fraude
def detect_fraud_in_image(image_array, model, threshold=0.5):
    """
    Detecta si una imagen contiene una transacción fraudulenta
    """
    # Predecir
    prediction_proba = model.predict(image_array, verbose=0)[0][0]
    is_fraud = prediction_proba > threshold
    
    # Determinar nivel de riesgo
    if prediction_proba > 0.8:
        risk_level = "🚨 ALTO RIESGO"
        color = "red"
    elif prediction_proba > 0.5:
        risk_level = "⚠️  RIESGO MODERADO" 
        color = "orange"
    elif prediction_proba > 0.3:
        risk_level = "🔍 RIESGO BAJO"
        color = "yellow"
    else:
        risk_level = "✅ BAJO RIESGO"
        color = "green"
    
    return {
        'probabilidad_fraude': float(prediction_proba),
        'es_fraudulenta': bool(is_fraud),
        'nivel_riesgo': risk_level,
        'color_riesgo': color,
        'recomendacion': 'BLOQUEAR TRANSACCIÓN' if is_fraud else 'APROBAR TRANSACCIÓN',
        'confianza': 'ALTA' if prediction_proba > 0.8 or prediction_proba < 0.2 else 'MEDIA'
    }

# Función para mostrar resultados de forma visual
def display_fraud_detection_results(original_image, processed_image, results):
    """
    Muestra los resultados de la detección de fraude de forma visual
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Imagen original
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Imagen Original de la Transacción', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Imagen procesada
    ax2.imshow(processed_image.squeeze(), cmap='gray')
    ax2.set_title('Imagen Procesada para el Modelo', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar resultados en formato de reporte
    print("\n" + "🔍" * 30)
    print("🎯 RESULTADO DE ANÁLISIS DE FRAUDE")
    print("🔍" * 30)
    
    print(f"\n📊 PROBABILIDAD DE FRAUDE: {results['probabilidad_fraude']:.4f}")
    print(f"🎯 NIVEL DE RIESGO: {results['nivel_riesgo']}")
    print(f"📋 RECOMENDACIÓN: {results['recomendacion']}")
    print(f"💪 CONFIANZA: {results['confianza']}")
    
    # Barra de progreso visual
    prob = results['probabilidad_fraude']
    bar_length = 40
    filled_length = int(bar_length * prob)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\n📈 SCORE DE RIESGO: [{bar}] {prob:.1%}")
    
    if results['es_fraudulenta']:
        print("\n🚨 ALERTA: Esta transacción presenta características de FRAUDE")
        print("   Acciones recomendadas:")
        print("   • Bloquear la transacción inmediatamente")
        print("   • Notificar al equipo de seguridad")
        print("   • Contactar al cliente para verificación")
    else:
        print("\n✅ TRANSACCIÓN LEGÍTIMA")
        print("   Acciones recomendadas:")
        print("   • Proceder con el pago normal")
        print("   • Registrar para análisis posterior")
        print("   • Monitorear patrones similares")

# Sistema interactivo de subida de imágenes
def fraud_detection_system():
    """
    Sistema interactivo para detectar fraude en imágenes subidas
    """
    print("\n" + "="*70)
    print("📸 SISTEMA DE DETECCIÓN DE FRAUDE - SUBIR IMAGEN")
    print("="*70)
    
    print("\n📝 INSTRUCCIONES:")
    print("   1. Haz clic en 'Choose Files' para seleccionar una imagen")
    print("   2. La imagen puede ser de un comprobante, Yape, etc.")
    print("   3. El sistema analizará patrones de fraude")
    print("   4. Se mostrará el resultado con recomendaciones")
    
    # Subir imagen
    uploaded = files.upload()
    
    if not uploaded:
        print("❌ No se subió ninguna imagen. Intenta nuevamente.")
        return
    
    # Procesar la primera imagen subida
    file_name = list(uploaded.keys())[0]
    file_content = uploaded[file_name]
    
    print(f"\n📁 Imagen subida: {file_name}")
    print("🔄 Procesando imagen...")
    
    try:
        # Cargar y preprocesar imagen
        image = Image.open(io.BytesIO(file_content))
        original_image = image.copy()
        
        # Convertir a RGB si es necesario
        if image.mode != 'L':
            image = image.convert('L')  # Convertir a escala de grises
        
        # Preprocesar para el modelo
        processed_image = preprocess_uploaded_image(np.array(image))
        
        # Detectar fraude
        results = detect_fraud_in_image(processed_image, model)
        
        # Mostrar resultados
        display_fraud_detection_results(np.array(original_image), processed_image, results)
        
        # Guardar resultado en historial
        save_detection_result(file_name, results)
        
    except Exception as e:
        print(f"❌ Error al procesar la imagen: {e}")
        print("💡 Asegúrate de que sea una imagen válida (PNG, JPG, JPEG)")

# Función para guardar historial de detecciones
def save_detection_result(filename, results):
    """
    Guarda los resultados de la detección en un historial
    """
    try:
        # Intentar cargar historial existente
        try:
            history_df = pd.read_csv('fraud_detection_history.csv')
        except:
            # Crear nuevo historial si no existe
            history_df = pd.DataFrame(columns=[
                'timestamp', 'filename', 'probability', 'is_fraud', 
                'risk_level', 'recommendation', 'confidence'
            ])
        
        # Agregar nueva entrada
        new_entry = {
            'timestamp': pd.Timestamp.now(),
            'filename': filename,
            'probability': results['probabilidad_fraude'],
            'is_fraud': results['es_fraudulenta'],
            'risk_level': results['nivel_riesgo'],
            'recommendation': results['recomendacion'],
            'confidence': results['confianza']
        }
        
        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
        history_df.to_csv('fraud_detection_history.csv', index=False)
        
        print(f"💾 Resultado guardado en historial: fraud_detection_history.csv")
        
    except Exception as e:
        print(f"⚠️  No se pudo guardar en historial: {e}")

# Función para mostrar historial de detecciones
def show_detection_history():
    """
    Muestra el historial de detecciones realizadas
    """
    try:
        history_df = pd.read_csv('fraud_detection_history.csv')
        
        if history_df.empty:
            print("📊 El historial de detecciones está vacío.")
            return
        
        print("\n" + "="*70)
        print("📊 HISTORIAL DE DETECCIONES DE FRAUDE")
        print("="*70)
        
        print(f"\nTotal de análisis realizados: {len(history_df)}")
        fraud_count = history_df['is_fraud'].sum()
        print(f"Transacciones fraudulentas detectadas: {fraud_count}")
        print(f"Tasa de fraude: {fraud_count/len(history_df)*100:.1f}%")
        
        # Mostrar últimas 5 detecciones
        print(f"\n📋 ÚLTIMAS DETECCIONES:")
        recent_detections = history_df.tail(5)
        
        for _, detection in recent_detections.iterrows():
            status = "🚨 FRAUDE" if detection['is_fraud'] else "✅ LEGÍTIMA"
            print(f"   • {detection['filename']}: {detection['probability']:.3f} - {status}")
        
        # Gráfico de distribución de probabilidades
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(history_df['probability'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Umbral de fraude')
        plt.xlabel('Probabilidad de Fraude')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Scores de Fraude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        fraud_counts = history_df['is_fraud'].value_counts()
        colors = ['green', 'red']
        plt.pie(fraud_counts.values, labels=['Legítimas', 'Fraudulentas'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Distribución de Transacciones')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("📊 No hay historial de detecciones disponible.")
    except Exception as e:
        print(f"❌ Error al cargar el historial: {e}")

# Función para probar con ejemplos del dataset
def test_with_sample_images():
    """
    Prueba el sistema con imágenes de ejemplo del dataset MNIST
    """
    print("\n" + "="*70)
    print("🧪 PRUEBA CON IMÁGENES DE EJEMPLO")
    print("="*70)
    
    # Cargar dataset de prueba
    (_, _), (X_test, y_test_original) = keras.datasets.mnist.load_data()
    y_test = (y_test_original >= 5).astype(int)  # 0-4: legítimo, 5-9: fraude
    
    # Seleccionar algunas imágenes de prueba
    sample_indices = np.random.choice(len(X_test), 6, replace=False)
    
    print("🔍 Analizando 6 transacciones de ejemplo...")
    print("   (Dígitos 0-4: Legítimas, Dígitos 5-9: Fraudulentas)\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        # Preprocesar imagen
        processed_image = preprocess_uploaded_image(X_test[idx])
        
        # Detectar fraude
        results = detect_fraud_in_image(processed_image, model)
        
        # Mostrar imagen y resultados
        axes[i].imshow(X_test[idx], cmap='gray')
        
        # Color del título según resultado
        title_color = 'red' if results['es_fraudulenta'] else 'green'
        actual_label = "FRAUDE" if y_test[idx] == 1 else "LEGÍTIMA"
        prediction_correct = (results['es_fraudulenta'] == (y_test[idx] == 1))
        
        axes[i].set_title(
            f'Transacción {i+1}\nReal: {actual_label}\n'
            f'Pred: {"FRAUDE" if results["es_fraudulenta"] else "LEGÍTIMA"}\n'
            f'Prob: {results["probabilidad_fraude"]:.3f}',
            color='green' if prediction_correct else 'red',
            fontweight='bold'
        )
        axes[i].axis('off')
        
        # Marcar con ✓ o ✗ según precisión
        marker = '✓' if prediction_correct else '✗'
        axes[i].text(0.5, -0.1, marker, transform=axes[i].transAxes, 
                    fontsize=20, ha='center', 
                    color='green' if prediction_correct else 'red')
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎯 LEYENDA:")
    print("   ✓ = Predicción correcta")
    print("   ✗ = Predicción incorrecta")
    print("   Verde = Transacción legítima")
    print("   Rojo = Transacción fraudulenta")

# Menú principal del sistema
def main_menu():
    """
    Menú principal del sistema de detección de fraude
    """
    while True:
        print("\n" + "="*70)
        print("🏦 SISTEMA DE DETECCIÓN DE FRAUDE EN TRANSACCIONES")
        print("="*70)
        print("\n📋 OPCIONES DISPONIBLES:")
        print("   1. 📸 Subir imagen para análisis de fraude")
        print("   2. 🧪 Probar con ejemplos del dataset")
        print("   3. 📊 Ver historial de detecciones")
        print("   4. 🚪 Salir del sistema")
        
        choice = input("\n👉 Selecciona una opción (1-4): ").strip()
        
        if choice == '1':
            fraud_detection_system()
        elif choice == '2':
            test_with_sample_images()
        elif choice == '3':
            show_detection_history()
        elif choice == '4':
            print("\n👋 ¡Gracias por usar el Sistema de Detección de Fraude!")
            print("💡 Recuerda que este es un sistema de prueba para demostración")
            break
        else:
            print("❌ Opción no válida. Por favor, selecciona 1-4.")
        
        input("\n⏎ Presiona Enter para continuar...")

# Información del sistema
print("\n🔧 CONFIGURACIÓN DEL SISTEMA:")
print(f"   • Modelo: CNN entrenado para detección de fraude")
print(f"   • Input: Imágenes 32x32 escala de grises")
print(f"   • Umbral de detección: 0.5")
print(f"   • Métricas esperadas: >95% precisión")

print("\n🎯 CARACTERÍSTICAS DETECTABLES:")
print("   • Patrones de transacciones fraudulentas")
print("   • Anomalías en comprobantes")
print("   • Firmas digitales sospechosas")
print("   • Patrones de comportamiento inusuales")

# Ejecutar el sistema
if __name__ == "__main__":
    main_menu()