import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

# Descargar recursos NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Dataset
data = {
    'texto': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    'etiqueta': ['real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake']
}

df = pd.DataFrame(data)

# Configuración preprocesamiento
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

def preprocesar_texto(texto):
    """Limpieza, eliminación de stopwords y stemming"""
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.split()
    palabras_procesadas = [
        stemmer.stem(palabra) 
        for palabra in palabras 
        if palabra not in stop_words and len(palabra) > 2
    ]
    return ' '.join(palabras_procesadas)

df['texto_procesado'] = df['texto'].apply(preprocesar_texto)

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=1)
X = vectorizer.fit_transform(df['texto_procesado'])
y = df['etiqueta']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Entrenar modelo
modelo = MultinomialNB(alpha=0.5)
modelo.fit(X_train, y_train)

# Evaluación
y_pred_test = modelo.predict(X_test)
y_pred_all = modelo.predict(X)

accuracy_test = accuracy_score(y_test, y_pred_test)
accuracy_all = accuracy_score(y, y_pred_all)
conf_matrix = confusion_matrix(y, y_pred_all, labels=['real', 'fake'])
report = classification_report(y, y_pred_all, target_names=['real', 'fake'], output_dict=True)

# Resultados
print("="*80)
print("RESULTADOS DEL MODELO")
print("="*80)
print(f"\nPrecisión total (accuracy): {accuracy_all:.2%}")
print(f"Precisión en prueba: {accuracy_test:.2%}")

print("\nMATRIZ DE CONFUSIÓN")
print("-" * 80)
print(f"                    Predicho: REAL    Predicho: FAKE")
print(f"Real: REAL          {conf_matrix[0][0]:^15} {conf_matrix[0][1]:^15}")
print(f"Real: FAKE          {conf_matrix[1][0]:^15} {conf_matrix[1][1]:^15}")

print("\nMÉTRICAS POR CLASE")
print("-" * 80)
print(f"Clase REAL:")
print(f"  - Precisión: {report['real']['precision']:.2%}")
print(f"  - Recall: {report['real']['recall']:.2%}")
print(f"  - F1-Score: {report['real']['f1-score']:.2%}")

print(f"\nClase FAKE:")
print(f"  - Precisión: {report['fake']['precision']:.2%}")
print(f"  - Recall: {report['fake']['recall']:.2%}")
print(f"  - F1-Score: {report['fake']['f1-score']:.2%}")

# Clasificación de nuevas noticias
print("\n" + "="*80)
print("CLASIFICACIÓN DE NUEVAS NOTICIAS")
print("="*80)

nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

nuevas_procesadas = [preprocesar_texto(noticia) for noticia in nuevas_noticias]
X_nuevas = vectorizer.transform(nuevas_procesadas)
predicciones = modelo.predict(X_nuevas)
probabilidades = modelo.predict_proba(X_nuevas)

for i, noticia in enumerate(nuevas_noticias):
    clase_idx = modelo.classes_.tolist().index(predicciones[i])
    confianza = probabilidades[i][clase_idx]
    
    print(f"\n📰 Noticia {i+1}: \"{noticia}\"")
    print(f"   Clasificación: {predicciones[i].upper()}")
    print(f"   Confianza: {confianza:.1%}")
    print(f"   Probabilidades: fake={probabilidades[i][0]:.2%}, real={probabilidades[i][1]:.2%}")

# Características más discriminantes
print("\n" + "="*80)
print("CARACTERÍSTICAS MÁS DISCRIMINANTES")
print("="*80)

feature_names = vectorizer.get_feature_names_out()
log_prob_fake = modelo.feature_log_prob_[0]
log_prob_real = modelo.feature_log_prob_[1]
diff = log_prob_fake - log_prob_real

top_fake_idx = np.argsort(diff)[-5:][::-1]
print("\n🔴 Top 5 características FAKE:")
for idx in top_fake_idx:
    print(f"   - '{feature_names[idx]}'")

top_real_idx = np.argsort(diff)[:5]
print("\n🟢 Top 5 características REAL:")
for idx in top_real_idx:
    print(f"   - '{feature_names[idx]}'")