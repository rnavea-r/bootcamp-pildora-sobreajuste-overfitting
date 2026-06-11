# ============================================================
# OVERFITTING: Ejemplo práctico completo
# Píldora formativa - Bootcamp Data Analyst
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# ============================================================
# 1. GENERAMOS UN DATASET DE EJEMPLO
# ============================================================
# Simulamos un dataset de predicción (ej: pisos en Málaga)
# 200 muestras, 10 features, algo de ruido

X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Dividimos en training (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 50)
print("DATASET")
print("=" * 50)
print(f"Training:  {X_train.shape[0]} muestras")
print(f"Test:      {X_test.shape[0]} muestras")


# ============================================================
# 2. LOS 3 MODELOS: underfitting, óptimo, overfitting
# ============================================================

modelos = {
    "Underfitting (max_depth=1)": DecisionTreeClassifier(max_depth=1, random_state=42),
    "Óptimo (max_depth=3)":       DecisionTreeClassifier(max_depth=3, random_state=42),
    "Overfitting (sin límite)":   DecisionTreeClassifier(random_state=42),
}

print("\n" + "=" * 50)
print("COMPARATIVA DE MODELOS")
print("=" * 50)
print(f"{'Modelo':<30} {'Train':>8} {'Test':>8} {'Veredicto'}")
print("-" * 60)

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_test  = accuracy_score(y_test,  modelo.predict(X_test))
    diferencia = acc_train - acc_test

    if diferencia > 0.15:
        veredicto = "⚠️  Overfitting"
    elif acc_train < 0.75:
        veredicto = "😴  Underfitting"
    else:
        veredicto = "✅  Bien"

    print(f"{nombre:<30} {acc_train:>7.1%} {acc_test:>7.1%}   {veredicto}")


# ============================================================
# 3. K-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 50)
print("K-FOLD CROSS VALIDATION (k=5)")
print("=" * 50)

for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
    print(f"{nombre:<30}  Media: {scores.mean():.1%}  ± {scores.std():.1%}")

print("\n💡 Alta desviación estándar = el modelo es inestable = señal de overfitting")


# ============================================================
# 4. CURVA DE APRENDIZAJE
#    Muestra cómo evoluciona el error a medida que
#    añadimos más datos de entrenamiento
# ============================================================

from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Curvas de Aprendizaje: Train vs Validación", fontsize=14, fontweight="bold")

colores = {
    "Underfitting (max_depth=1)": "#EF4444",
    "Óptimo (max_depth=3)":       "#10B981",
    "Overfitting (sin límite)":   "#3B82F6",
}

for ax, (nombre, modelo) in zip(axes, modelos.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy"
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    color = colores[nombre]

    ax.plot(train_sizes, train_mean, "o-", color=color,
            label="Training", linewidth=2)
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color=color)

    ax.plot(train_sizes, val_mean, "s--", color="#F59E0B",
            label="Validación", linewidth=2)
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color="#F59E0B")

    ax.set_title(nombre, fontsize=11, fontweight="bold")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Anotación de la brecha
    brecha = train_mean[-1] - val_mean[-1]
    ax.annotate(
        f"Brecha: {brecha:.1%}",
        xy=(train_sizes[-1], (train_mean[-1] + val_mean[-1]) / 2),
        fontsize=9,
        color="gray",
        ha="right"
    )

plt.tight_layout()
plt.savefig("curvas_aprendizaje.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊 Gráfico guardado como 'curvas_aprendizaje.png'")


# ============================================================
# 5. CURVA DE COMPLEJIDAD (la U del Tradeoff Sesgo-Varianza)
#    Muestra cómo varía el error según la profundidad del árbol
# ============================================================

profundidades  = range(1, 20)
train_accs = []
test_accs  = []

for d in profundidades:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs.append(accuracy_score(y_test,  m.predict(X_test)))

errores_train = [1 - a for a in train_accs]
errores_test  = [1 - a for a in test_accs]

plt.figure(figsize=(10, 5))
plt.plot(profundidades, errores_train, "o-", color="#10B981",
         label="Error Training", linewidth=2)
plt.plot(profundidades, errores_test,  "s-", color="#EF4444",
         label="Error Test/Validación", linewidth=2)

# Marcamos el punto óptimo
idx_optimo = errores_test.index(min(errores_test))
plt.axvline(x=idx_optimo + 1, color="#F59E0B", linestyle="--", linewidth=1.5)
plt.annotate(
    f"✅ Óptimo\n(max_depth={idx_optimo + 1})",
    xy=(idx_optimo + 1, min(errores_test)),
    xytext=(idx_optimo + 3, min(errores_test) + 0.05),
    arrowprops=dict(arrowstyle="->", color="#F59E0B"),
    fontsize=10,
    color="#F59E0B"
)

plt.title("Curva en U: Error vs Complejidad del modelo", fontsize=13, fontweight="bold")
plt.xlabel("Profundidad del árbol (complejidad)")
plt.ylabel("Error (1 - Accuracy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("curva_u_complejidad.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Gráfico guardado como 'curva_u_complejidad.png'")

print("\n" + "=" * 50)
print("CONCLUSIÓN")
print("=" * 50)
print("Nuestro objetivo como Data Analysts no es crear")
print("el modelo que mejor explique el pasado,")
print("sino el que mejor prediga el futuro. 🎯")
