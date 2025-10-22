# Detector de Noticias Falsas con Naive Bayes

Sistema automático de clasificación de noticias utilizando procesamiento de lenguaje natural (NLP) y el algoritmo Naive Bayes.

## 🎯 Objetivo

Entrenar un modelo de Machine Learning capaz de distinguir entre noticias reales y noticias falsas analizando su contenido textual.

## 🛠️ Tecnologías

- **Python 3.x**
- **pandas**: Manejo de datos
- **scikit-learn**: Algoritmo Naive Bayes y métricas
- **NLTK**: Procesamiento de lenguaje natural

## 📦 Instalación

```bash
pip install pandas scikit-learn nltk
```

## 🚀 Uso

```bash
python clasificador_noticias.py
```

## 📊 Dataset

**10 noticias** etiquetadas manualmente:
- 5 noticias reales (reformas, anuncios oficiales, estudios científicos)
- 5 noticias falsas (afirmaciones extraordinarias, pseudociencia)

## 🔧 Preprocesamiento de Texto

1. **Conversión a minúsculas**
2. **Eliminación de números y puntuación**
3. **Eliminación de stopwords** (palabras comunes en español)
4. **Stemming** (reducción a raíz de palabras)
5. **Vectorización TF-IDF** con unigramas y bigramas

## 🧠 Modelo

**Multinomial Naive Bayes** con suavizado de Laplace (alpha=0.5)

### Características:
- Vectorización TF-IDF
- N-gramas (1,2) para capturar contexto
- Train/test split (70/30)

## 📈 Métricas

El modelo genera:
- **Accuracy** (precisión total)
- **Matriz de confusión**
- **Precisión, Recall y F1-Score** por clase
- **Características más discriminantes**

## 💡 Ejemplos de Clasificación

```python
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]
```

**Resultados esperados:**
- Noticia 1 → REAL (estudio científico creíble)
- Noticia 2 → FAKE (afirmación extraordinaria)

## 📝 Características Discriminantes

El modelo identifica automáticamente palabras y frases que son más comunes en:
- **Noticias reales**: gobierno, nasa, ministerio, salud
- **Noticias falsas**: milagrosa, cura, convierte, espían

## 🎓 Conceptos Clave

### Naive Bayes
Algoritmo probabilístico basado en el teorema de Bayes que asume independencia entre características.

**Fórmula:**
```
P(clase|texto) ∝ P(clase) × ∏ P(palabra|clase)
```

### TF-IDF
Técnica de vectorización que pondera palabras según:
- **TF**: Frecuencia en el documento
- **IDF**: Rareza en el corpus completo

### Stemming
Reducción de palabras a su raíz:
- "científicos" → "cientif"
- "confirmado" → "confirm"

## 📂 Estructura del Proyecto

```
.
├── clasificador_noticias.py    # Código principal
├── README.md                   # Este archivo
└── requirements.txt            # Dependencias
```

## 🔍 Mejoras Futuras

- [ ] Aumentar dataset de entrenamiento
- [ ] Usar embeddings pre-entrenados (Word2Vec, BERT)
- [ ] Implementar validación cruzada
- [ ] Crear API REST para clasificación en tiempo real
- [ ] Agregar análisis de sentimientos

## 📄 Licencia

MIT License

## ✨ Autor

Proyecto educativo de Machine Learning y NLP

---

⭐ **Nota**: Este es un modelo de demostración educativa. Para uso en producción se requiere un dataset mucho más grande y diverso.
