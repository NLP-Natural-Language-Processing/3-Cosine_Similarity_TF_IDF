# 3-Cosine_Similarity_TF_IDF
0. [About Dataset](#schema0)
1. [NLP Cosine Similarity | TF-IDF](#schema1)
2. [Tweets - Sentiment Analysis](#schema2)


<hr>

<a name="schema0"></a>

## 0. About Dataset

**Context**

This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the Twitter API. The tweets have been annotated (0 = negative, 4 = positive) and can be used to detect sentiment.

**Content**

It contains the following 6 fields:

- target: the polarity of the tweet (0 = negative and 4 = positive)
- ids: The id of the tweet ( 2087)
- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- flag: The query (lyx). If there is no query, then this value is NO_QUERY.
- user: the user that tweeted.
- text: the text of the tweet.

<hr>

<a name="schema1"></a>

## 1. NLP Cosine Similarity | TF-IDF

### **TfidfVectorizer** 

Es una herramienta en Python, específicamente en la librería `scikit-learn`, que se utiliza para convertir una colección de documentos de texto en una matriz de términos y frecuencias de documentos (TF-IDF). **TF-IDF** significa `"Term Frequency-Inverse Document Frequency"` (Frecuencia de Término-Frecuencia Inversa de Documento) y es una medida numérica que indica la importancia de una palabra en relación con un conjunto de documentos.

Aquí hay una breve explicación de lo que hace TfidfVectorizer:

- **Tokenización:** Divide cada documento de texto en palabras individuales o **"tokens"**.
- **Construcción del vocabulario:** Crea un vocabulario de todas las palabras únicas que aparecen en los documentos.
- **Calcula la frecuencia de términos (TF)**: Para cada documento, calcula la frecuencia de cada palabra en ese documento.
- **Calcula la frecuencia inversa de documentos (IDF)**: Calcula la importancia de cada palabra en todo el corpus (conjunto de documentos) considerando cuántos documentos contienen esa palabra. Las palabras comunes que aparecen en muchos documentos tendrán un IDF más bajo, mientras que las palabras menos comunes tendrán un IDF más alto.
- **Calcula TF-IDF**: Multiplica la frecuencia del término (TF) por la frecuencia inversa del documento (IDF) para obtener el peso de cada término en cada documento.



### **CountVectorizer**
Al igual que TfidfVectorizer, es una herramienta proporcionada por la biblioteca scikit-learn de Python que se utiliza para convertir una colección de documentos de texto en una matriz de recuentos de términos. A diferencia de TfidfVectorizer, que utiliza la ponderación TF-IDF para medir la importancia de los términos, **CountVectorizer** simplemente cuenta la frecuencia de ocurrencia de cada término en cada documento.

Aquí está lo que hace CountVectorizer:

- **Tokenización:** Divide cada documento de texto en palabras individuales o "tokens".
- **Construcción del vocabulario:** Crea un vocabulario de todas las palabras únicas que aparecen en los documentos.
- **Calcula la frecuencia de términos (TF):** Para cada documento, cuenta cuántas veces aparece cada palabra en ese documento.
- **Representación de la matriz:** Convierte los documentos de texto en una matriz donde cada fila representa un documento y cada columna representa una palabra del vocabulario. Los valores en la matriz son los recuentos de términos para cada palabra en cada documento.

**CountVectorizer** proporciona una representación simple y directa de la frecuencia de términos en los documentos de texto y es útil para tareas como la clasificación de texto, agrupación de documentos y recuperación de información. Esencialmente, es una forma de convertir datos de texto en un formato numérico que los algoritmos de aprendizaje automático pueden entender y procesar.

### **TfidfVectorizer vs CountVectorizer**

Si bien tanto **TfidfVectorizer** como **CountVectorizer** son herramientas utilizadas para preprocesar datos de texto en Python, tienen algunas diferencias clave en términos de cómo representan la información textual:

- **Ponderación de términos:**
    - **CountVectorizer:** Solo cuenta la frecuencia de ocurrencia de cada término en cada documento. No considera la importancia relativa de los términos en el corpus.
    - **TfidfVectorizer:** Utiliza la ponderación TF-IDF (Frecuencia de Término-Frecuencia Inversa de Documento) para medir la importancia de un término en un documento en relación con el corpus completo. TF-IDF tiene en cuenta tanto la frecuencia de ocurrencia del término en el documento como su rareza en el corpus.
- **Impacto de términos comunes:**
  - **CountVectorizer:** No penaliza los términos comunes que aparecen en muchos documentos. Por lo tanto, los términos muy frecuentes pueden dominar la representación.
  - **TfidfVectorizer:** Penaliza los términos comunes que aparecen en muchos documentos mediante la ponderación IDF. Esto ayuda a reducir el impacto de los términos muy frecuentes y destacar términos más raros y específicos.
- **Uso en diferentes contextos:**
  - **CountVectorizer:** Es útil cuando se quiere simplemente contar la ocurrencia de palabras en un conjunto de documentos, sin tener en cuenta su importancia relativa.
  - **TfidfVectorizer:** Es más adecuado cuando se busca medir la importancia relativa de las palabras en los documentos, especialmente en tareas como la recuperación de información, donde es importante identificar términos clave.

En resumen, **CountVectorizer** es una opción más simple y directa que cuenta la ocurrencia de palabras en documentos, mientras que **TfidfVectorizer** ofrece una representación más sofisticada que considera la importancia relativa de las palabras mediante la ponderación TF-IDF. La elección entre ambas depende del contexto específico de la tarea y de si se desea una representación más básica o más informativa de los datos de texto.


### **cosine_similarity**

La función **cosine_similarity** es una herramienta comúnmente utilizada en Python, específicamente en bibliotecas como scikit-learn, para calcular la similitud coseno entre dos vectores. La similitud coseno es una medida numérica que cuantifica la similitud direccional entre dos vectores en un espacio euclidiano. Es ampliamente utilizada en diversas aplicaciones, incluyendo recuperación de información, clustering, y recomendación, entre otros.
- **Aplicaciones:**
- En procesamiento de lenguaje natural, se utiliza para calcular la similitud entre vectores de representaciones de texto, como el TF-IDF o los embeddings de palabras.
- En sistemas de recomendación, se utiliza para encontrar elementos similares en función de sus características o atributos.
- En clustering, se utiliza para medir la similitud entre diferentes puntos de datos en un espacio de características.

En resumen, **cosine_similarity** es una función que calcula la similitud coseno entre vectores, lo que proporciona una medida numérica de la similitud direccional entre ellos en un espacio euclidiano. Es una herramienta valiosa en una variedad de aplicaciones de aprendizaje automático y análisis de datos.


### **Bag of Words (BOWs)**

La Bolsa de Palabras es una representación simple pero poderosa que captura la información sobre la presencia y la frecuencia de las palabras en un texto, lo que la hace útil para una variedad de tareas de NLP. Sin embargo, no conserva información sobre la estructura y el contexto del texto.

Por ejemplo, considera los siguientes tres documentos:

"El gato está en la alfombra."

"El perro corre en el parque."

"El pájaro canta en el árbol."

El vocabulario resultante podría ser: ["el", "gato", "está", "en", "la", "alfombra", "perro", "corre", "parque", "pájaro", "canta", "árbol"].

Entonces, la representación BoW de estos documentos sería:

[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

[1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0]

[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]


<hr>

<a name="schema2"></a>

## 2. Tweets - Sentiment Analysis

### **Stemmer**

**SnowballStemmer** ofrece soporte para una amplia gama de idiomas, incluidos el inglés, francés, español, alemán, italiano, portugués, holandés, sueco, ruso, entre otros.