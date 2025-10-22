# Fase 5: Métricas y KPIs - VERSIÓN CORREGIDA (sin error KeyError)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report,
                           precision_recall_curve, roc_curve, average_precision_score)
import json
import warnings
warnings.filterwarnings('ignore')

print("🎯 FASE 5: MÉTRICAS Y KPIs - EVALUACIÓN DEL MODELO ENTRENADO")
print("="*60)

# Cargar el modelo entrenado y hacer predicciones finales
from tensorflow import keras

# Cargar modelo guardado
print("📥 Cargando modelo entrenado...")
model = keras.models.load_model('fraud_detection_fast_model.h5')

# Cargar datos de prueba (usando el mismo preprocesamiento de Fase 4)
(X_train, y_train_original), (X_test, y_test_original) = keras.datasets.mnist.load_data()
y_test = (y_test_original >= 5).astype(int)

# Preprocesamiento consistente con Fase 4
def preprocess_images_fast(images):
    images = images.astype('float32') / 255.0
    images_resized = tf.image.resize(images[..., tf.newaxis], [32, 32])
    return images_resized.numpy()

X_test_processed = preprocess_images_fast(X_test)

# Hacer predicciones finales
print("🔮 Generando predicciones finales...")
y_pred_proba = model.predict(X_test_processed, verbose=0, batch_size=256)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 1. MÉTRICAS TÉCNICAS COMPLETAS
print("\n🔧 1. MÉTRICAS TÉCNICAS DEL MODELO ENTRENADO")
print("-" * 50)

# Calcular todas las métricas
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = np.mean(y_test == y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)

print(f"📈 Exactitud (Accuracy): {accuracy:.4f}")
print(f"🎯 Precisión: {precision:.4f}")
print(f"🔍 Recall (Sensibilidad): {recall:.4f}")
print(f"⚖️  F1-Score: {f1:.4f}")
print(f"📊 AUC-ROC: {auc_roc:.4f}")
print(f"📈 AUC-PR: {auc_pr:.4f}")

# 2. ANÁLISIS DETALLADO DE LA MATRIZ DE CONFUSIÓN
print("\n🎯 2. ANÁLISIS DETALLADO - MATRIZ DE CONFUSIÓN")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("📊 DESGLOSE DE PREDICCIONES:")
print(f"✅ Verdaderos Negativos (TN): {tn:,} - Transacciones legítimas correctamente identificadas")
print(f"❌ Falsos Positivos (FP): {fp:,} - Transacciones legítimas bloqueadas por error")
print(f"❌ Falsos Negativos (FN): {fn:,} - Transacciones fraudulentas no detectadas")
print(f"✅ Verdaderos Positivos (TP): {tp:,} - Transacciones fraudulentas correctamente detectadas")

# Métricas derivadas críticas para fraude
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

print(f"\n📈 MÉTRICAS CRÍTICAS PARA FRAUDE:")
print(f"🔴 Tasa de Falsos Positivos (FPR): {fpr:.6f}")
print(f"🔴 Tasa de Falsos Negativos (FNR): {fnr:.4f}")
print(f"🟢 Especificidad: {specificity:.4f}")
print(f"🟢 Valor Predictivo Negativo (NPV): {npv:.4f}")

# 3. KPIs DE IMPACTO EN EL NEGOCIO
print("\n💼 3. KPIs DE IMPACTO EN EL NEGOCIO")
print("-" * 50)

# Parámetros de negocio (ajustables según tu caso)
BUSINESS_PARAMS = {
    'avg_transaction_amount': 150,      # USD promedio por transacción
    'avg_fraud_amount': 500,           # USD promedio por transacción fraudulenta
    'cost_per_false_positive': 15,     # Costo en servicio al cliente
    'customer_lifetime_value': 2000,   # Valor lifetime del cliente
    'fraud_prevention_team_cost': 50   # Costo por revisión manual
}

print("💰 PARÁMETROS DE NEGOCIO:")
for param, value in BUSINESS_PARAMS.items():
    print(f"   - {param}: ${value}")

# Cálculos de impacto económico
total_transactions = len(y_test)
fraudulent_transactions = np.sum(y_test == 1)
legitimate_transactions = np.sum(y_test == 0)

total_fraud_prevented = tp * BUSINESS_PARAMS['avg_fraud_amount']
total_fraud_missed = fn * BUSINESS_PARAMS['avg_fraud_amount']
total_customer_inconvenience = fp * BUSINESS_PARAMS['cost_per_false_positive']
manual_review_cost = (tp + fp) * BUSINESS_PARAMS['fraud_prevention_team_cost'] / 10  # Solo 10% necesitan revisión

net_savings = total_fraud_prevented - total_fraud_missed - total_customer_inconvenience - manual_review_cost

print(f"\n💰 IMPACTO ECONÓMICO ESTIMADO (sobre {total_transactions:,} transacciones):")
print(f"💵 Fraude prevenido: ${total_fraud_prevented:,}")
print(f"💸 Fraude no detectado: ${total_fraud_missed:,}")
print(f"😠 Costo por molestias a clientes: ${total_customer_inconvenience:,}")
print(f"👥 Costo equipo de revisión: ${manual_review_cost:,}")
print(f"📈 AHORRO NETO: ${net_savings:,}")

# ROI del sistema
development_cost = 5000  # Costo estimado de desarrollo
roi = (net_savings - development_cost) / development_cost * 100

print(f"📊 ROI del proyecto: {roi:.1f}%")

# 6. MÉTRICAS DE CALIDAD OPERATIVA
print("\n⚙️  6. MÉTRICAS DE CALIDAD OPERATIVA")
print("-" * 50)

# Tasa de alertas
alert_rate = (tp + fp) / len(y_test) * 100

# Eficiencia de revisión
review_efficiency = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

# Cobertura de detección
detection_coverage = tp / (tp + fn) * 100

# Precisión operativa
operational_precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0

print(f"📊 Tasa de alertas: {alert_rate:.2f}%")
print(f"🎯 Eficiencia de revisión: {review_efficiency:.2f}%")
print(f"🔍 Cobertura de detección: {detection_coverage:.2f}%")
print(f"📈 Precisión operativa: {operational_precision:.2f}%")

# 7. EVALUACIÓN CONTRA CRITERIOS DE ÉXITO - VERSIÓN CORREGIDA
print("\n🎯 7. EVALUACIÓN CONTRA CRITERIOS DE ÉXITO")
print("-" * 50)

# Criterios de éxito del negocio (basados en tu problema original)
SUCCESS_CRITERIA = {
    'recall': 0.85,      # Detección de al menos 85% de fraudes
    'precision': 0.80,   # Máximo 20% de falsas alarmas
    'f1': 0.82,         # Balance general
    'auc_roc': 0.90,    # Capacidad de discriminación
    'fpr': 0.01        # Máximo 1% de falsos positivos
}

print("📋 CRITERIOS DE ÉXITO DEFINIDOS:")
print(f"   - Recall mínimo: {SUCCESS_CRITERIA['recall']}")
print(f"   - Precisión mínima: {SUCCESS_CRITERIA['precision']}")
print(f"   - F1-Score mínimo: {SUCCESS_CRITERIA['f1']}")
print(f"   - AUC-ROC mínimo: {SUCCESS_CRITERIA['auc_roc']}")
print(f"   - FPR máximo: {SUCCESS_CRITERIA['fpr']}")

print("\n✅ EVALUACIÓN DEL MODELO ENTRENADO:")
criteria_met = 0
total_criteria = len(SUCCESS_CRITERIA)

# Datos de rendimiento actual
performance_data = {
    'recall': recall,
    'precision': precision, 
    'f1': f1,
    'auc_roc': auc_roc,
    'fpr': fpr
}

for criterion, target in SUCCESS_CRITERIA.items():
    actual = performance_data[criterion]
    
    # Para FPR queremos que sea menor, para los demás mayor
    if criterion == 'fpr':
        meets_criteria = actual <= target
    else:
        meets_criteria = actual >= target
        
    status = "🎉 CUMPLE" if meets_criteria else "⚠️  NO CUMPLE"
    print(f"   {criterion}: {actual:.4f} vs {target} - {status}")
    
    if meets_criteria:
        criteria_met += 1

success_rate = (criteria_met / total_criteria) * 100
print(f"\n📊 TASA DE CUMPLIMIENTO: {success_rate:.1f}%")

if success_rate >= 80:
    print("🎊 ¡EXCELENTE! El modelo supera los criterios de éxito")
    deployment_recommendation = "✅ RECOMENDADO PARA PRODUCCIÓN"
elif success_rate >= 60:
    print("✅ BUENO - El modelo cumple la mayoría de criterios")
    deployment_recommendation = "✅ APROBADO CON MONITOREO"
else:
    print("🔧 MEJORABLE - El modelo necesita optimizaciones")
    deployment_recommendation = "⏸️  REQUIERE MEJORAS ANTES DE PRODUCCIÓN"

# 4. ANÁLISIS POR UMBRALES DE DECISIÓN (opcional - si quieres incluirlo)
print("\n📊 4. ANÁLISIS DE UMBRALES ÓPTIMOS")
print("-" * 50)

# Probar diferentes umbrales
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

print("Umbral | Precisión | Recall   | F1-Score | Falsos Positivos")
print("-" * 65)

best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba > threshold).astype(int).flatten()
    
    # Métricas técnicas
    precision_t = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall_t = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_t = f1_score(y_test, y_pred_thresh, zero_division=0)
    fp_t = np.sum((y_pred_thresh == 1) & (y_test == 0))
    
    results.append({
        'threshold': threshold,
        'precision': precision_t,
        'recall': recall_t,
        'f1': f1_t,
        'fp': fp_t
    })
    
    print(f"{threshold:.1f}    | {precision_t:.4f}   | {recall_t:.4f}  | {f1_t:.4f}  | {fp_t:>6,}")
    
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = threshold

print(f"\n🎯 UMBRAL ÓPTIMO (basado en F1-Score): {best_threshold:.1f}")

# 5. CURVAS DE EVALUACIÓN
print("\n📈 5. CURVAS DE EVALUACIÓN DEL MODELO")
print("-" * 50)

# Curva ROC
fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)

# Curva Precision-Recall
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(15, 5))

# Curva ROC
plt.subplot(1, 2, 1)
plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Detección de Fraude')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Curva Precision-Recall
plt.subplot(1, 2, 2)
plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
         label=f'PR curve (AP = {auc_pr:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.title('Curva Precision-Recall - Detección de Fraude')
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. DASHBOARD RESUMEN EJECUTIVO
print("\n📊 8. DASHBOARD EJECUTIVO - RESUMEN FINAL")
print("=" * 50)

plt.figure(figsize=(15, 10))

# Gráfico 1: Métricas principales
plt.subplot(2, 3, 1)
metrics_names = ['Precisión', 'Recall', 'F1-Score', 'AUC-ROC']
metrics_values = [precision, recall, f1, auc_roc]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = plt.bar(metrics_names, metrics_values, color=colors)
plt.ylim(0, 1)
plt.title('Métricas Principales', fontweight='bold', fontsize=12)
plt.xticks(rotation=45)
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# Gráfico 2: Impacto económico
plt.subplot(2, 3, 2)
impact_categories = ['Fraude\nPrevenido', 'Fraude\nNo Detectado', 'Costo\nFalsas Alarmas', 'Ahorro\nNeto']
impact_values = [total_fraud_prevented/1000, total_fraud_missed/1000, 
                 total_customer_inconvenience/1000, net_savings/1000]
colors_impact = ['green', 'red', 'orange', 'blue']

bars = plt.bar(impact_categories, impact_values, color=colors_impact)
plt.title('Impacto Económico (Miles USD)', fontweight='bold', fontsize=12)
plt.xticks(rotation=45)
for bar, value in zip(bars, impact_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(impact_values)*0.01, 
             f'${value:.0f}K', ha='center', va='bottom', fontweight='bold')

# Gráfico 3: Matriz de confusión
plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', 
            xticklabels=['Legítima', 'Fraudulenta'],
            yticklabels=['Legítima', 'Fraudulenta'],
            annot_kws={"size": 12})
plt.title('Matriz de Confusión', fontweight='bold', fontsize=12)

# Gráfico 4: Cumplimiento de criterios
plt.subplot(2, 3, 4)
criteria_labels = ['Recall', 'Precisión', 'F1-Score', 'AUC-ROC', 'FPR']
criteria_current = [recall, precision, f1, auc_roc, 1-fpr]  # Invertir FPR para visualización
criteria_target = [SUCCESS_CRITERIA['recall'], SUCCESS_CRITERIA['precision'], 
                   SUCCESS_CRITERIA['f1'], SUCCESS_CRITERIA['auc_roc'], 
                   1-SUCCESS_CRITERIA['fpr']]

x = np.arange(len(criteria_labels))
width = 0.35

plt.bar(x - width/2, criteria_current, width, label='Actual', alpha=0.7)
plt.bar(x + width/2, criteria_target, width, label='Objetivo', alpha=0.7)
plt.xticks(x, criteria_labels, rotation=45)
plt.title('Cumplimiento de Criterios', fontweight='bold', fontsize=12)
plt.legend()

# Gráfico 5: Distribución de scores
plt.subplot(2, 3, 5)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Legítimas', color='green', density=True)
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraudulentas', color='red', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', label='Umbral actual', linewidth=2)
plt.xlabel('Score de Fraude')
plt.ylabel('Densidad')
plt.title('Distribución de Scores', fontweight='bold', fontsize=12)
plt.legend()

# Gráfico 6: ROI y recomendación
plt.subplot(2, 3, 6)
plt.text(0.5, 0.7, f"ROI: {roi:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold', 
         color='green' if roi > 0 else 'red')
plt.text(0.5, 0.5, deployment_recommendation, ha='center', va='center', fontsize=12, 
         fontweight='bold', wrap=True)
plt.text(0.5, 0.3, f"Cumplimiento: {success_rate:.1f}%", ha='center', va='center', fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Recomendación Final', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()

# 9. GUARDAR REPORTE FINAL
print("\n💾 9. GUARDANDO REPORTE FINAL")
print("-" * 50)

report_data = {
    'model_performance': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr)
    },
    'business_impact': {
        'total_transactions': int(total_transactions),
        'fraud_prevented_usd': float(total_fraud_prevented),
        'fraud_missed_usd': float(total_fraud_missed),
        'customer_inconvenience_usd': float(total_customer_inconvenience),
        'net_savings_usd': float(net_savings),
        'roi_percentage': float(roi)
    },
    'operational_metrics': {
        'alert_rate': float(alert_rate),
        'review_efficiency': float(review_efficiency),
        'detection_coverage': float(detection_coverage),
        'false_positive_rate': float(fpr)
    },
    'success_evaluation': {
        'success_rate': float(success_rate),
        'criteria_met': int(criteria_met),
        'total_criteria': int(total_criteria),
        'deployment_recommendation': deployment_recommendation
    },
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
}

# Guardar reporte en JSON
with open('fraud_detection_performance_report.json', 'w') as f:
    json.dump(report_data, f, indent=2)

print("✅ Reporte guardado: 'fraud_detection_performance_report.json'")

# RESUMEN FINAL
print("\n" + "="*70)
print("🎉 FASE 5 COMPLETADA - EVALUACIÓN FINALIZADA")
print("="*70)
print(f"📊 RESUMEN EJECUTIVO:")
print(f"   • Exactitud del modelo: {accuracy:.1%}")
print(f"   • Fraudes detectados: {recall:.1%}")
print(f"   • Falsas alarmas: {fpr:.4%}")
print(f"   • Ahorro neto estimado: ${net_savings:,}")
print(f"   • ROI del proyecto: {roi:.1f}%")
print(f"   • Recomendación: {deployment_recommendation}")
print(f"   • Criterios cumplidos: {criteria_met}/{total_criteria}")
print("\n🚀 Próximo paso: Fase 6 - Sistema de Despliegue y Pruebas")
print("="*70)