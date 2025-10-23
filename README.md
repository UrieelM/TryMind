# Optimizador de Parámetros CLAHE con IA

Este proyecto implementa un sistema de agentes de IA auto-mejorable para encontrar los parámetros óptimos (`clip_limit`, `tile_size`) del algoritmo **CLAHE** (Contrast Limited Adaptive Histogram Equalization) para el realce de imágenes.

## 1\. Descripción del Problema

Aplicar CLAHE requiere ajustar dos parámetros clave: `clip_limit` y `tile_size`. Encontrar la combinación óptima es un desafío, ya que:

  * Parámetros incorrectos pueden no tener efecto o, peor aún, introducir artefactos visuales y ruido excesivo.
  * La combinación ideal depende de las características de cada imagen.
  * Existe una **restricción matemática** (`formula_result = (clip_limit * tile_size²) / 256`) que debe cumplirse para que los parámetros sean efectivos, haciendo la búsqueda manual aún más compleja.

Este sistema **resuelve el problema automatizando la búsqueda** de estos parámetros. Utiliza un ciclo de agentes de IA que proponen, validan, ejecutan y evalúan iterativamente los parámetros hasta converger en una solución óptima, basada en métricas de calidad de imagen.

## 2\. Arquitectura del Sistema

El sistema está compuesto por un orquestador (`SistemaAutoMejorable`) y tres agentes especializados:

### Agente 1: `AgenteEjecutor`

  * **Rol:** Ejecutar la operación CLAHE y actuar como "guardián" de los parámetros.
  * **Función Clave:** Antes de aplicar CLAHE, valida que los parámetros propuestos cumplan con la fórmula crítica de validación:
      * $formula = (clip\_limit \times tile\_size^2) / 256$
  * **Reglas de Falla:** Rechaza los parámetros si:
    1.  $formula < 1$ (el umbral de clipping es inefectivo).
    2.  $formula > clip\_limit$ (no se produce un clipping real).

### Agente 2: `AgenteEvaluador`

  * **Rol:** Cuantificar la "calidad" de la imagen procesada.
  * **Métricas:** Calcula un vector de características de 2 dimensiones:
    1.  **Entropía de Shannon:** Mide la riqueza de información y detalle en la imagen.
    2.  **Varianza del Laplaciano:** Mide la nitidez y la presencia de bordes.
  * **Métrica de Éxito:** El objetivo no es maximizar una sola métrica, sino minimizar la **distancia euclidiana** entre el vector de la imagen actual y el **vector promedio** de todas las ejecuciones válidas anteriores. Esto ayuda al sistema a converger hacia un resultado estable y balanceado.

### Agente 3: `AgenteOptimizador` (El Cerebro)

  * **Rol:** Analizar los resultados históricos y proponer nuevos parámetros.
  * **Tecnología:** Utiliza un modelo LLM (`gpt-4o`) a través de la API de OpenAI.
  * **Proceso de Decisión:** Se le proporciona el historial completo de la optimización:
      * **Intentos Válidos:** Qué parámetros produjeron qué métricas y qué `distancia_promedio`.
      * **Intentos Inválidos:** Qué parámetros fallaron y *por qué* (la razón del `AgenteEjecutor`).
  * **Salida:** Genera un análisis JSON con su razonamiento y los nuevos parámetros a probar, balanceando la **exploración** (probar ideas nuevas) y la **explotación** (refinar los mejores resultados).

## 3\. Instrucciones de Ejecución

### Prerrequisitos

  * Python 3.7+
  * Dependencias de Python

### Instalación

1.  Clona el repositorio.
2.  Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Crea un archivo `.env` en la raíz del proyecto para almacenar tu clave de API de OpenAI:
    ```
    API_KEY="sk-..."
    ```

### Ejecución

1.  Asegúrate de tener una imagen de prueba (por ejemplo, en `src/test1.jpg`).
2.  Ejecuta el script principal:
    ```bash
    python tu_script.py
    ```
3.  El sistema correrá las 10 iteraciones de optimización. Al finalizar, encontrarás:
      * Las imágenes (`01_original.jpg`, `02_optimizado.jpg`, `03_comparacion.jpg`) en la carpeta `outputs/`.
      * Un reporte detallado de cada ciclo en `reporte_optimizacion.json`.

## 4\. Explicación del Ciclo de Auto-Mejora

El sistema aprende y se adapta en cada iteración. Este es el flujo exacto basado en el reporte:

1.  **Run 1 (Aprendizaje del Fracaso):**

      * **Propuesta Inicial:** `clip_limit=2.0`, `tile_size=8`.
      * **Ejecutor:** Falla. La validación determina `Formula result = 0`, lo cual es `< 1`.
      * **Optimizador (IA):** Recibe el error: "threshold inefectivo". Analiza la fórmula y razona que `tile_size` era muy pequeño. Propone una corrección: `clip=3.0, tile=16`.

2.  **Run 2 (Primera Línea Base):**

      * **Propuesta (de IA):** `clip=3.0, tile=16`.
      * **Ejecutor:** ¡Éxito\! `Formula result = 3`, lo cual es válido.
      * **Evaluador:** Calcula el vector. La `distancia_promedio` (comparada con el original) es **587.25**.
      * **Optimizador (IA):** Recibe el éxito y la distancia. Decide explorar aumentando el `clip_limit` a `4.0`.

3.  **Run 3-7 (Exploración):**

      * El sistema prueba diferentes valores de `clip_limit` (`4.0`, `4.5`, `5.0`, `3.5`, `3.25`).
      * El `vector_promedio` se actualiza en cada paso, haciendo que el objetivo sea más estable.
      * La IA observa que `clip_limit` demasiado altos (como `5.0`) empeoran la distancia (Score: -278.6), y `clip_limit` más bajos la mejoran (Run 6, `clip=3.5`, Score: -152.3).

4.  **Run 8 (Convergencia / Explotación):**

      * **Propuesta (de IA):** El agente razona que el óptimo está entre `3.5` y `4.0`. Propone `clip=3.75`.
      * **Evaluador:** ¡Gran mejora\! La distancia cae drásticamente a **17.03**.
      * **Optimizador (IA):** Confirma que está en la zona correcta y decide "explotar" (refinar) este valor, proponiendo `3.6`.

5.  **Run 9-10 (Ajuste Fino):**

      * Run 9 (`clip=3.6`) resulta peor (Distancia: `70.59`).
      * La IA analiza esto y determina que el óptimo debe estar *por encima* de `3.75`. Propone `clip=3.8`.
      * **Run 10 (Óptimo):** `clip=3.8, tile=16`. Se alcanza la mejor distancia del ciclo: **5.77**.

## 5\. Métricas de Mejora

La evidencia cuantificable de la mejora se encuentra en la métrica objetivo: `distancia_promedio` (distancia al vector promedio `[Entropía, Var. Laplaciano]`). **Un valor más bajo es mejor.**

Basado en el `reporte_optimizacion.json`, el sistema demostró un aprendizaje y una mejora significativos:

| Run | Parámetros (Clip, Tile) | `distancia_promedio` (Error) | Análisis de IA (Estrategia) |
| :--- | :--- | :--- | :--- |
| 1 | (2.0, 8) | **INVÁLIDO** (`Formula < 1`) | Exploración (Corrección de error) |
| 2 | (3.0, 16) | 587.25 | Exploración |
| 8 | (3.75, 16) | 17.03 | Explotación (Refinamiento) |
| 10 | **(3.8, 16)** | **5.77** (Óptimo) | Explotación (Ajuste fino) |

### Conclusión Cuantificable

El sistema de auto-mejora logró:

  * **Identificar y corregir** automáticamente parámetros matemáticamente inválidos (Run 1).
  * **Reducir la métrica de error (distancia) de 587.25 a 5.77** en 9 iteraciones válidas.
  * **Converger exitosamente** en los parámetros óptimos (`clip_limit=3.8`, `tile_size=16`) para esta imagen.