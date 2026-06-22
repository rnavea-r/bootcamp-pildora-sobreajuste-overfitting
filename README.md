# bootcamp-pildora-sobreajuste-overfitting
El problema: cuando el modelo aprende demasiado 
## 📌 ¿De qué trata?

El **overfitting** o sobreajuste ocurre cuando un modelo de Machine Learning es tan complejo, o se entrena tanto, que se aprende de memoria el ruido y los detalles específicos de los datos de entrenamiento, en lugar de aprender la tendencia general.

Dicho de otra forma: el modelo explica perfectamente el pasado, pero falla prediciendo el futuro.

Presentación en [Gamma](https://gamma.app/docs/sjq31huhrhuydn7)
---
## 📂 Contenido del repositorio

```
📁 bootcamp-pildora-sobreajuste-overfitting/
├── 💻 codigo/
│   └── overfitting_ejemplo.py
└── 📄 README.md
```
> 🔗 La presentación completa está en [Gamma]([https://gamma.app](https://gamma.app/docs/sjq31huhrhuydn7))
---
## 🎯 Conceptos clave

### Los 3 escenarios

| Escenario | Complejidad | Error Training | Error Test |
|-----------|-------------|----------------|------------|
| Underfitting | Muy baja | Alto | Alto |
| ✅ Ajuste justo | Óptima | Bajo | Bajo |
| Overfitting | Muy alta | Muy bajo | Alto |

### Tradeoff Sesgo–Varianza


- **Alto sesgo** → modelo demasiado simple → Underfitting
- **Alta varianza** → modelo demasiado complejo → Overfitting
- **Objetivo** → encontrar el mínimo de la curva en U
---

## 🔍 Cómo detectarlo

Monitorea las métricas de rendimiento (Accuracy, RMSE) durante el entrenamiento:
```
Error
 │
 │ ╲  Training
 │  ╲___________
 │       Validación ╲___/‾‾‾‾
 │
 └─────────────────────────── Épocas
              ↑
        Punto de overfitting
```
**Señal de alerta:** la curva de validación toca su mínimo y empieza a subir mientras la de training sigue bajando.

---

## 🛠️ Cómo solucionarlo

| Técnica | Descripción | En Python |
|---------|-------------|-----------|
| Más datos | Más muestras = menos memorización | — |
| K-Fold CV | Evaluar en K particiones distintas | `cross_val_score(..., cv=5)` |
| Regularización L1 | Elimina features irrelevantes (Lasso) | `Lasso(alpha=0.1)` |
| Regularización L2 | Penaliza coeficientes grandes (Ridge) | `Ridge(alpha=0.1)` |
| Reducir complejidad | Limitar profundidad del árbol | `max_depth=3` |
| Early Stopping | Parar en el mínimo de validación | `callbacks` en Keras |

---

## 💻 Ejemplo de código

```python
pythonfrom sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# ❌ Sin control → Overfitting
modelo_overfit = DecisionTreeClassifier()
modelo_overfit.fit(X_train, y_train)
# Train accuracy: 99% | Test accuracy: 54% ⚠️

# ✅ Con control → Generaliza
modelo_ok = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(modelo_ok, X, y, cv=5)
# Train accuracy: 88% | Test accuracy: 82% ✅
```
---
## 📊 Ejemplo práctico: Predicción de precios de pisos en Málaga

| Modelo | Training      | Test |    Veredicto |
|--------|--------------|-----------|-----------|
| Modelo Lineal | 72% | 70% | ✅ Bien |
| Árbol prof. 3 | 88% | 82% | ✅ Bien |
| Árbol sin límite | 99% | 54% | ⚠️ Overfitting |


> 💡 Un 99% en training no es motivo de celebración. Es una señal de alerta.

---

## 🎮 Kahoot

El quiz de esta píldora tiene 9 preguntas sobre overfitting, detección y soluciones.

👉 [Jugar al Kahoot](https://create.kahoot.it/share/memorizas-o-aprendes/632772f4-eed4-4e8f-8927-9fe9448eb0e2)


📚 Recursos para profundizar


- [Scikit-learn: Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-learn: Regularización](https://scikit-learn.org/stable/modules/linear_model.html)
- [Bias-Variance Tradeoff — StatQuest](https://www.youtube.com/watch?v=EuBBz3bI-aA)



Píldora formativa creada para el Bootcamp de Data Analyst.
