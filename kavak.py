import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import json
from openai import OpenAI


def cargar_api_key(env_var: str = "API_KEY", env_file: str = ".env") -> str:
    """Obtiene la API key desde variables de entorno o desde un archivo .env."""
    valor = os.getenv(env_var)
    if valor:
        return valor

    ruta_env = Path(env_file)
    if ruta_env.exists():
        for linea in ruta_env.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, contenido = linea.split("=", 1)
            if clave.strip() != env_var:
                continue
            contenido = contenido.strip().strip('"').strip("'")
            os.environ[env_var] = contenido
            return contenido

    raise RuntimeError(
        f"No se encontró la variable {env_var}. "
        f"Configúrala en el entorno o agrégala al archivo {env_file}."
    )


# ============================================================================
# AGENTE 1: EJECUTOR - Aplica CLAHE con validación de fórmula
# ============================================================================

class AgenteEjecutor:
    """Ejecuta CLAHE validando que los parámetros sean efectivos según la fórmula"""
    
    def __init__(self):
        self.history = []
    
    def reset(self):
        """Limpia el historial de ejecuciones"""
        self.history = []
        print("  🔄 Ejecutor reiniciado")
    
    @staticmethod
    def validar_parametros_clahe(clip_limit: float, tile_size: Tuple[int, int]) -> Dict:
        """
        Valida parámetros CLAHE usando la fórmula crítica:
        formula_result = (clip_limit * tile_size²) / 256
        
        Returns:
            dict con 'valido', 'formula_result', 'razon'
        """
        ws = tile_size[0]  # Asumimos tile_size cuadrado
        formula_result = int((clip_limit * (ws * ws)) / 256)
        
        if formula_result < 1:
            return {
                'valido': False,
                'formula_result': formula_result,
                'razon': f'Formula result ({formula_result}) < 1: threshold inefectivo'
            }
        
        if formula_result > clip_limit:
            return {
                'valido': False,
                'formula_result': formula_result,
                'razon': f'Formula result ({formula_result}) > clip_limit ({clip_limit}): sin clipping real'
            }
        
        return {
            'valido': True,
            'formula_result': formula_result,
            'razon': 'Parámetros válidos'
        }
    
    def aplicar_clahe(self, imagen: np.ndarray, clip_limit: float, tile_size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Aplica CLAHE validando primero los parámetros
        
        Returns:
            (imagen_procesada, info_validacion)
        """
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gray = imagen.copy()
        
        # VALIDACIÓN CRÍTICA
        validacion = self.validar_parametros_clahe(clip_limit, tile_size)
        
        if not validacion['valido']:
            print(f"  ⚠️  Parámetros inválidos: {validacion['razon']}")
            return None, validacion
        
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        resultado = clahe.apply(imagen_gray)
        
        # Guardar en historial
        self.history.append({
            'clip_limit': clip_limit,
            'tile_size': tile_size,
            'formula_result': validacion['formula_result'],
            'resultado': resultado
        })
        
        return resultado, validacion


# ============================================================================
# AGENTE 2: EVALUADOR - Calcula métricas y vector promedio
# ============================================================================

class AgenteEvaluador:
    """Evalúa calidad usando entropía, varianza Laplaciano y distancia al vector promedio"""
    
    def __init__(self):
        self.resultados_evaluaciones = []
        self.imagen_original = None
        self.vector_original = None
        self.todos_los_vectores = []  # Para calcular promedio verdadero
    
    def reset(self):
        """Limpia todo el estado del evaluador para procesar nueva imagen"""
        self.resultados_evaluaciones = []
        self.imagen_original = None
        self.vector_original = None
        self.todos_los_vectores = []
        print("  🔄 Evaluador reiniciado")
    
    def set_imagen_original(self, imagen: np.ndarray):
        """Establece la imagen original y calcula su vector base"""
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        self.imagen_original = imagen
        self.vector_original = self._calcular_vector(imagen)
    
    def _calcular_entropia(self, imagen: np.ndarray) -> float:
        """Calcula entropía de Shannon"""
        histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        histograma = histograma / histograma.sum()
        histograma = histograma[histograma > 0]
        entropia = -np.sum(histograma * np.log2(histograma))
        return float(entropia)
    
    def _calcular_varianza_laplaciano(self, imagen: np.ndarray) -> float:
        """Calcula varianza del Laplaciano (nitidez/bordes)"""
        laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
        varianza = laplaciano.var()
        return float(varianza)
    
    def _calcular_vector(self, imagen: np.ndarray) -> np.ndarray:
        """Vector de características [entropía, varianza_laplaciano]"""
        entropia = self._calcular_entropia(imagen)
        varianza = self._calcular_varianza_laplaciano(imagen)
        return np.array([entropia, varianza])
    
    def _calcular_vector_promedio(self, excluir_ultimo: bool = True) -> np.ndarray:
        """
        Calcula el vector promedio de evaluaciones históricas
        
        Args:
            excluir_ultimo: Si True, excluye el vector actual del cálculo del promedio
                          (evita que la distancia sea artificialmente cero en Run 1)
        
        Returns:
            Vector promedio
        """
        if not self.todos_los_vectores:
            return self.vector_original
        
        # Si solo hay 1 vector y queremos excluirlo, usar el original como referencia
        if len(self.todos_los_vectores) == 1 and excluir_ultimo:
            return self.vector_original
        
        # Si hay múltiples vectores, calcular promedio excluyendo el último
        if excluir_ultimo and len(self.todos_los_vectores) > 1:
            vectores = np.array(self.todos_los_vectores[:-1])  # Excluir último
        else:
            vectores = np.array(self.todos_los_vectores)
        
        vector_promedio = np.mean(vectores, axis=0)
        return vector_promedio
    
    def evaluar(self, imagen_procesada: Optional[np.ndarray], params: Dict, 
                run_id: int, validacion: Dict) -> Optional[Dict]:
        """
        Evalúa la imagen procesada calculando distancia al vector promedio
        
        Returns:
            None si la imagen es inválida, dict con evaluación si es válida
        """
        # Si los parámetros fueron inválidos
        if imagen_procesada is None:
            return None
        
        # Calcular vector de la imagen procesada
        vector_procesado = self._calcular_vector(imagen_procesada)
        self.todos_los_vectores.append(vector_procesado)
        
        # Calcular vector promedio EXCLUYENDO el vector actual
        # Esto evita que Run 1 tenga distancia = 0 artificialmente
        vector_promedio = self._calcular_vector_promedio(excluir_ultimo=True)
        
        # Calcular distancias euclidianas
        distancia_original = float(np.linalg.norm(vector_procesado - self.vector_original))
        distancia_promedio = float(np.linalg.norm(vector_procesado - vector_promedio))
        
        evaluacion = {
            'run_id': run_id,
            'params': params,
            'validacion': validacion,
            'vector_procesado': vector_procesado.tolist(),
            'vector_original': self.vector_original.tolist(),
            'vector_promedio': vector_promedio.tolist(),
            'distancia_original': distancia_original,
            'distancia_promedio': distancia_promedio,
            'entropia': float(vector_procesado[0]),
            'varianza_laplaciano': float(vector_procesado[1]),
            'score': -distancia_promedio,  # Negativo porque minimizamos distancia
            'formula_result': validacion['formula_result']
        }
        
        self.resultados_evaluaciones.append(evaluacion)
        return evaluacion


# ============================================================================
# AGENTE 3: OPTIMIZADOR - Aprende y propone nuevos parámetros
# ============================================================================

class AgenteOptimizador:
    """Usa LLM para analizar resultados y proponer parámetros que cumplan la fórmula"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.historial_decisiones = []
        self.mejor_resultado = None
    
    def reset(self):
        """Limpia el historial de decisiones para nueva imagen"""
        self.historial_decisiones = []
        self.mejor_resultado = None
        print("  🔄 Optimizador reiniciado")
    
    def analizar_y_optimizar(self, evaluaciones: List[Dict], 
                            evaluaciones_invalidas: List[Dict],
                            run_actual: int) -> Dict:
        """
        Analiza resultados válidos e inválidos para proponer mejores parámetros
        
        Args:
            evaluaciones: Lista de evaluaciones válidas
            evaluaciones_invalidas: Lista de intentos con parámetros inválidos
            run_actual: Número de iteración actual
        """
        contexto = self._preparar_contexto(evaluaciones, evaluaciones_invalidas)
        
        prompt = f"""Eres un optimizador experto en CLAHE (Contrast Limited Adaptive Histogram Equalization).

OBJETIVO: Encontrar parámetros óptimos (clip_limit, tile_size) que minimicen la distancia euclidiana al vector promedio.

FÓRMULA CRÍTICA DE VALIDACIÓN CLAHE:
formula_result = (clip_limit × tile_size²) / 256

REGLAS OBLIGATORIAS:
1. formula_result DEBE ser >= 1 (si no, el threshold es inefectivo)
2. formula_result DEBE ser <= clip_limit (si no, no hay clipping real)
3. tile_size debe ser potencia de 2: [1, 2, 4, 8, 16, 32, 64]
4. clip_limit típicamente entre 1.0 y 15.0

CONTEXTO ACTUAL:
{contexto}

ANÁLISIS REQUERIDO:
1. ¿Qué parámetros han sido inválidos y por qué?
2. ¿Qué patrón observas en las distancias válidas?
3. ¿Cómo afecta clip_limit a la entropía?
4. ¿Cómo afecta tile_size a la varianza Laplaciano?
5. ¿Qué ajuste específico minimizaría la distancia al promedio?

IMPORTANTE: Asegúrate que tus parámetros propuestos cumplan:
- (clip_limit × tile_size²) / 256 >= 1
- (clip_limit × tile_size²) / 256 <= clip_limit

RESPONDE EN JSON:
{{
  "analisis": "Tu análisis detallado",
  "razonamiento": "Por qué estos parámetros son válidos y mejoran",
  "nuevos_params": {{
    "clip_limit": <float entre 1.0 y 15.0>,
    "tile_size": <int potencia de 2: 1,2,4,8,16,32,64>
  }},
  "formula_esperada": <cálculo de (clip_limit × tile_size²) / 256>,
  "confianza": <float entre 0 y 1>,
  "estrategia": "exploración o explotación"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un optimizador que entiende las restricciones matemáticas de CLAHE."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        resultado = json.loads(response.choices[0].message.content)
        resultado = self._validar_y_ajustar_params(resultado)
        
        self.historial_decisiones.append({
            'run': run_actual,
            'decision': resultado
        })
        
        return resultado
    
    def _preparar_contexto(self, evaluaciones: List[Dict], 
                          evaluaciones_invalidas: List[Dict]) -> str:
        """Prepara resumen de intentos válidos e inválidos"""
        lineas = []
        
        # Mostrar últimos 5 válidos
        if evaluaciones:
            lineas.append("RESULTADOS VÁLIDOS:")
            for ev in evaluaciones[-5:]:
                lineas.append(
                    f"  Run {ev['run_id']}: cl={ev['params']['clip_limit']:.2f}, "
                    f"ws={ev['params']['tile_size']}, "
                    f"formula={ev['formula_result']}, "
                    f"dist_promedio={ev['distancia_promedio']:.4f}"
                )
        
        # Mostrar últimos 3 inválidos
        if evaluaciones_invalidas:
            lineas.append("\nINTENTOS INVÁLIDOS (APRENDER DE ESTOS):")
            for ev in evaluaciones_invalidas[-3:]:
                lineas.append(
                    f"  Run {ev['run_id']}: cl={ev['params']['clip_limit']:.2f}, "
                    f"ws={ev['params']['tile_size']} → {ev['validacion']['razon']}"
                )
        
        # Vector promedio actual
        if evaluaciones:
            ultimo = evaluaciones[-1]
            lineas.append(f"\nVector promedio actual: {ultimo['vector_promedio']}")
            lineas.append(f"Mejor distancia hasta ahora: {min(e['distancia_promedio'] for e in evaluaciones):.4f}")
        
        return "\n".join(lineas)
    
    def _validar_y_ajustar_params(self, resultado: Dict) -> Dict:
        """Valida que los parámetros propuestos cumplan la fórmula"""
        params = resultado['nuevos_params']
        
        # Ajustar clip_limit
        clip_limit = np.clip(float(params['clip_limit']), 1.0, 15.0)
        
        # Ajustar tile_size a potencia de 2 más cercana
        tile_size_raw = int(params['tile_size'])
        potencias = [1, 2, 4, 8, 16, 32, 64]
        tile_size = min(potencias, key=lambda x: abs(x - tile_size_raw))
        
        # Verificar fórmula
        formula_result = int((clip_limit * (tile_size * tile_size)) / 256)
        
        # Si no cumple, ajustar
        if formula_result < 1:
            # Incrementar clip_limit para que fórmula >= 1
            clip_limit = max(clip_limit, 256.0 / (tile_size * tile_size))
            clip_limit = min(clip_limit, 15.0)
        
        if formula_result > clip_limit:
            # Reducir tile_size o aumentar clip_limit
            while formula_result > clip_limit and tile_size > 1:
                idx = potencias.index(tile_size)
                if idx > 0:
                    tile_size = potencias[idx - 1]
                    formula_result = int((clip_limit * (tile_size * tile_size)) / 256)
                else:
                    break
        
        resultado['nuevos_params'] = {
            'clip_limit': float(clip_limit),
            'tile_size': tile_size
        }
        resultado['formula_esperada'] = int((clip_limit * (tile_size * tile_size)) / 256)
        
        return resultado


# ============================================================================
# SISTEMA ORQUESTADOR
# ============================================================================

class SistemaAutoMejorable:
    """Coordina los 3 agentes con validación de fórmula CLAHE"""
    
    def __init__(self, api_key: str):
        self.ejecutor = AgenteEjecutor()
        self.evaluador = AgenteEvaluador()
        self.optimizador = AgenteOptimizador(api_key)
        self.run_actual = 0
        self.mejor_score = float('-inf')
        self.mejor_params = None
        self.evaluaciones_invalidas = []
    
    def reset(self):
        """Reinicia el sistema completo para procesar una nueva imagen"""
        print("\n🔄 REINICIANDO SISTEMA COMPLETO...")
        self.ejecutor.reset()
        self.evaluador.reset()
        self.optimizador.reset()
        self.run_actual = 0
        self.mejor_score = float('-inf')
        self.mejor_params = None
        self.evaluaciones_invalidas = []
        print("✓ Sistema reiniciado completamente\n")
    
    def inicializar(self, imagen: np.ndarray):
        """Inicializa el sistema con la imagen original"""
        self.evaluador.set_imagen_original(imagen)
        print(f"✓ Sistema inicializado")
        print(f"  Vector original: {self.evaluador.vector_original}")
    
    def ejecutar_ciclo(self, imagen: np.ndarray, params: Dict) -> Dict:
        """Ejecuta un ciclo: Ejecutor → Evaluador → Optimizador"""
        self.run_actual += 1
        
        print(f"\n{'='*70}")
        print(f"RUN {self.run_actual}")
        print(f"{'='*70}")
        print(f"Parámetros propuestos: clip_limit={params['clip_limit']:.2f}, tile_size={params['tile_size']}")
        
        # PASO 1: Ejecutor valida y aplica CLAHE
        imagen_procesada, validacion = self.ejecutor.aplicar_clahe(
            imagen, 
            params['clip_limit'], 
            (params['tile_size'], params['tile_size'])
        )
        
        print(f"Validación: {validacion['razon']}")
        print(f"Formula result: {validacion['formula_result']}")
        
        # PASO 2: Evaluador mide calidad (o registra fallo)
        if imagen_procesada is None:
            # Parámetros inválidos
            self.evaluaciones_invalidas.append({
                'run_id': self.run_actual,
                'params': params,
                'validacion': validacion
            })
            
            # Optimizador aprende del error
            decision = self.optimizador.analizar_y_optimizar(
                self.evaluador.resultados_evaluaciones,
                self.evaluaciones_invalidas,
                self.run_actual
            )
            
            return {
                'evaluacion': None,
                'decision': decision,
                'imagen_procesada': None,
                'valido': False
            }
        
        # Si es válido, evaluar
        evaluacion = self.evaluador.evaluar(imagen_procesada, params, self.run_actual, validacion)
        
        print(f"\nRESULTADOS:")
        print(f"  Vector procesado: [{evaluacion['entropia']:.4f}, {evaluacion['varianza_laplaciano']:.2f}]")
        print(f"  Vector promedio:  {evaluacion['vector_promedio']}")
        print(f"  Distancia al promedio: {evaluacion['distancia_promedio']:.4f}")
        print(f"  Score: {evaluacion['score']:.4f}")
        
        # Actualizar mejor resultado
        if evaluacion['score'] > self.mejor_score:
            self.mejor_score = evaluacion['score']
            self.mejor_params = params.copy()
            print(f"  ✓ ¡NUEVO MEJOR RESULTADO!")
        
        # PASO 3: Optimizador propone nuevos parámetros
        decision = self.optimizador.analizar_y_optimizar(
            self.evaluador.resultados_evaluaciones,
            self.evaluaciones_invalidas,
            self.run_actual
        )
        
        print(f"\nDECISIÓN DEL OPTIMIZADOR:")
        print(f"  Estrategia: {decision.get('estrategia', 'N/A')}")
        print(f"  Confianza: {decision.get('confianza', 0):.2f}")
        print(f"  Nuevos params: clip_limit={decision['nuevos_params']['clip_limit']:.2f}, "
              f"tile_size={decision['nuevos_params']['tile_size']}")
        print(f"  Formula esperada: {decision.get('formula_esperada', 'N/A')}")
        
        return {
            'evaluacion': evaluacion,
            'decision': decision,
            'imagen_procesada': imagen_procesada,
            'valido': True
        }
    
    def auto_mejorar(self, imagen: np.ndarray, num_iteraciones: int = 10, 
                    params_iniciales: Dict = None, reset_sistema: bool = True):
        """
        Ejecuta múltiples ciclos de auto-mejora
        
        Args:
            imagen: Imagen a procesar
            num_iteraciones: Número de ciclos de optimización
            params_iniciales: Parámetros de inicio (usa defaults si None)
            reset_sistema: Si True, limpia estado antes de empezar (recomendado para nueva imagen)
        """
        # Limpiar estado si se solicita (importante para nueva imagen)
        if reset_sistema:
            self.reset()
        
        self.inicializar(imagen)
        
        if params_iniciales is None:
            params = {'clip_limit': 2.0, 'tile_size': 8}
        else:
            params = params_iniciales
        
        resultados = []
        
        for i in range(num_iteraciones):
            resultado = self.ejecutar_ciclo(imagen, params)
            resultados.append(resultado)
            
            # Usar nuevos parámetros propuestos
            params = resultado['decision']['nuevos_params']
        
        # Resumen final
        print(f"\n{'='*70}")
        print(f"RESUMEN FINAL")
        print(f"{'='*70}")
        
        resultados_validos = [r for r in resultados if r['valido']]
        resultados_invalidos = [r for r in resultados if not r['valido']]
        
        print(f"Iteraciones válidas: {len(resultados_validos)}/{num_iteraciones}")
        print(f"Iteraciones inválidas: {len(resultados_invalidos)}/{num_iteraciones}")
        
        if resultados_validos:
            print(f"\nMejor score: {self.mejor_score:.4f}")
            print(f"Mejores parámetros: {self.mejor_params}")
            print(f"\nProgresión de scores válidos:")
            for r in resultados_validos:
                ev = r['evaluacion']
                print(f"  Run {ev['run_id']}: {ev['score']:.4f} "
                      f"(cl={ev['params']['clip_limit']:.2f}, ws={ev['params']['tile_size']})")
        
        return resultados
    
    def guardar_reporte_json(self, resultados: List[Dict], output_path: str = "reporte_optimizacion.json"):
        """
        Guarda un reporte JSON completo con el análisis de la IA en cada run
        
        Args:
            resultados: Lista de resultados de auto_mejorar()
            output_path: Ruta del archivo JSON a guardar
        """
        reporte = {
            'metadata': {
                'total_iteraciones': len(resultados),
                'iteraciones_validas': len([r for r in resultados if r['valido']]),
                'iteraciones_invalidas': len([r for r in resultados if not r['valido']]),
                'mejor_score': float(self.mejor_score) if self.mejor_params else None,
                'mejores_parametros': self.mejor_params
            },
            'vector_original': self.evaluador.vector_original.tolist() if self.evaluador.vector_original is not None else None,
            'iteraciones': []
        }
        
        for idx, resultado in enumerate(resultados, 1):
            iteracion_data = {
                'run': idx,
                'valido': resultado['valido']
            }
            
            # Si fue válido, incluir evaluación completa
            if resultado['valido'] and resultado['evaluacion']:
                ev = resultado['evaluacion']
                iteracion_data.update({
                    'parametros': ev['params'],
                    'formula_result': ev['formula_result'],
                    'vector_procesado': ev['vector_procesado'],
                    'vector_promedio': ev['vector_promedio'],
                    'metricas': {
                        'entropia': ev['entropia'],
                        'varianza_laplaciano': ev['varianza_laplaciano'],
                        'distancia_original': ev['distancia_original'],
                        'distancia_promedio': ev['distancia_promedio'],
                        'score': ev['score']
                    }
                })
            else:
                # Si fue inválido, incluir razón del fallo
                for eval_invalida in self.evaluaciones_invalidas:
                    if eval_invalida['run_id'] == idx:
                        iteracion_data.update({
                            'parametros': eval_invalida['params'],
                            'razon_fallo': eval_invalida['validacion']['razon'],
                            'formula_result': eval_invalida['validacion']['formula_result']
                        })
                        break
            
            # Incluir análisis y decisión de la IA (siempre presente)
            if resultado['decision']:
                decision = resultado['decision']
                iteracion_data['analisis_ia'] = {
                    'analisis': decision.get('analisis', ''),
                    'razonamiento': decision.get('razonamiento', ''),
                    'estrategia': decision.get('estrategia', ''),
                    'confianza': decision.get('confianza', 0),
                    'parametros_propuestos': decision.get('nuevos_params', {}),
                    'formula_esperada': decision.get('formula_esperada', None)
                }
            
            reporte['iteraciones'].append(iteracion_data)
        
        # Guardar JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Reporte JSON guardado: {output_path}")
        return reporte
    
    def generar_presentacion_resultados(self, resultados: List[Dict], imagen_original: np.ndarray, 
                                       output_dir: str = "outputs"):
        """
        Genera una presentación visual completa:
        1. Resultado óptimo primero (impacto visual)
        2. Explicación del proceso de auto-mejora run por run
        
        Args:
            resultados: Lista de resultados de auto_mejorar()
            imagen_original: Imagen original procesada
            output_dir: Directorio donde guardar los archivos
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convertir a escala de grises si es necesario
        if len(imagen_original.shape) == 3:
            imagen_gray = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gray = imagen_original.copy()
        
        resultados_validos = [r for r in resultados if r['valido']]
        
        if not resultados_validos:
            print("⚠️ No hay resultados válidos para presentar")
            return
        
        # ====================================================================
        # PASO 1: RESULTADO ÓPTIMO (LO PRIMERO QUE VE LA AUDIENCIA)
        # ====================================================================
        print(f"\n{'='*80}")
        print("🏆 RESULTADO ÓPTIMO ALCANZADO")
        print(f"{'='*80}\n")
        
        mejor_imagen, _ = self.ejecutor.aplicar_clahe(
            imagen_gray,
            self.mejor_params['clip_limit'],
            (self.mejor_params['tile_size'], self.mejor_params['tile_size'])
        )
        
        # Encontrar la evaluación del mejor resultado
        mejor_evaluacion = None
        for r in resultados_validos:
            if (r['evaluacion']['params']['clip_limit'] == self.mejor_params['clip_limit'] and
                r['evaluacion']['params']['tile_size'] == self.mejor_params['tile_size']):
                mejor_evaluacion = r['evaluacion']
                break
        
        print(f"📊 PARÁMETROS ÓPTIMOS:")
        print(f"   ├─ Clip Limit: {self.mejor_params['clip_limit']:.2f}")
        print(f"   ├─ Tile Size: {self.mejor_params['tile_size']}×{self.mejor_params['tile_size']}")
        print(f"   └─ Formula Result: {mejor_evaluacion['formula_result']}")
        
        print(f"\n📈 MEJORA ALCANZADA:")
        mejor_run = mejor_evaluacion['run_id']
        primer_resultado = resultados_validos[0]['evaluacion']
        distancia_inicial = primer_resultado['distancia_promedio']
        distancia_final = mejor_evaluacion['distancia_promedio']
        mejora_pct = ((distancia_inicial - distancia_final) / distancia_inicial) * 100
        
        print(f"   ├─ Distancia Inicial (Run 1): {distancia_inicial:.2f}")
        print(f"   ├─ Distancia Óptima (Run {mejor_run}): {distancia_final:.2f}")
        print(f"   ├─ Mejora: {mejora_pct:.1f}%")
        print(f"   └─ Score Final: {self.mejor_score:.4f}")
        
        print(f"\n🎯 MÉTRICAS DE CALIDAD:")
        print(f"   ├─ Entropía: {primer_resultado['entropia']:.4f} → {mejor_evaluacion['entropia']:.4f}")
        print(f"   └─ Nitidez (Var. Laplaciano): {primer_resultado['varianza_laplaciano']:.1f} → {mejor_evaluacion['varianza_laplaciano']:.1f}")
        
        # Guardar comparación visual del resultado óptimo
        comparacion_optima = np.hstack([imagen_gray, mejor_imagen])
        cv2.imwrite(f"{output_dir}/00_RESULTADO_OPTIMO.jpg", comparacion_optima)
        print(f"\n✅ Imagen guardada: {output_dir}/00_RESULTADO_OPTIMO.jpg")
        
        # ====================================================================
        # PASO 2: EXPLICACIÓN DEL PROCESO DE AUTO-MEJORA (RUN POR RUN)
        # ====================================================================
        print(f"\n{'='*80}")
        print("🔄 PROCESO DE AUTO-MEJORA: CÓMO SE ALCANZÓ EL ÓPTIMO")
        print(f"{'='*80}\n")
        
        print("El sistema utilizó 3 agentes colaborativos:\n")
        print("  🤖 Agente 1 - EJECUTOR: Aplica CLAHE con validación de parámetros")
        print("  📊 Agente 2 - EVALUADOR: Mide calidad (entropía + nitidez)")
        print("  🧠 Agente 3 - OPTIMIZADOR (GPT-4): Analiza y propone mejoras\n")
        
        for idx, resultado in enumerate(resultados, 1):
            print(f"\n{'─'*80}")
            print(f"RUN {idx}/{len(resultados)}")
            print(f"{'─'*80}")
            
            if not resultado['valido']:
                # Run inválido
                print(f"❌ PARÁMETROS INVÁLIDOS")
                for eval_inv in self.evaluaciones_invalidas:
                    if eval_inv['run_id'] == idx:
                        print(f"   ├─ Clip Limit: {eval_inv['params']['clip_limit']:.2f}")
                        print(f"   ├─ Tile Size: {eval_inv['params']['tile_size']}")
                        print(f"   └─ Razón: {eval_inv['validacion']['razon']}")
                        break
                
                # Análisis de la IA sobre el error
                if resultado['decision'] and 'analisis' in resultado['decision']:
                    print(f"\n🧠 ANÁLISIS DEL OPTIMIZADOR:")
                    analisis = resultado['decision']['analisis']
                    # Truncar si es muy largo
                    if len(analisis) > 200:
                        analisis = analisis[:200] + "..."
                    print(f"   {analisis}")
                    
                    print(f"\n💡 ACCIÓN CORRECTIVA:")
                    print(f"   ├─ Nuevo Clip Limit: {resultado['decision']['nuevos_params']['clip_limit']:.2f}")
                    print(f"   ├─ Nuevo Tile Size: {resultado['decision']['nuevos_params']['tile_size']}")
                    print(f"   └─ Estrategia: {resultado['decision'].get('estrategia', 'N/A')}")
                
                continue
            
            # Run válido
            ev = resultado['evaluacion']
            decision = resultado['decision']
            
            print(f"✅ EJECUCIÓN EXITOSA")
            print(f"\n📋 Parámetros Aplicados:")
            print(f"   ├─ Clip Limit: {ev['params']['clip_limit']:.2f}")
            print(f"   ├─ Tile Size: {ev['params']['tile_size']}")
            print(f"   └─ Formula Result: {ev['formula_result']}")
            
            print(f"\n📊 Resultados de Evaluación:")
            print(f"   ├─ Entropía: {ev['entropia']:.4f}")
            print(f"   ├─ Varianza Laplaciano: {ev['varianza_laplaciano']:.1f}")
            print(f"   ├─ Distancia al Promedio: {ev['distancia_promedio']:.2f}")
            print(f"   └─ Score: {ev['score']:.4f}")
            
            # Calcular mejora respecto al run anterior válido
            if idx > 1:
                runs_previos_validos = [r for r in resultados[:idx-1] if r['valido']]
                if runs_previos_validos:
                    ultimo_valido = runs_previos_validos[-1]['evaluacion']
                    mejora_run = ((ultimo_valido['distancia_promedio'] - ev['distancia_promedio']) / 
                                 ultimo_valido['distancia_promedio']) * 100
                    
                    if mejora_run > 0:
                        print(f"\n📈 Mejora vs Run Anterior: +{mejora_run:.1f}% ✓")
                    elif mejora_run < 0:
                        print(f"\n📉 Cambio vs Run Anterior: {mejora_run:.1f}% (explorando)")
            
            # Marcar si es el mejor resultado
            if ev['params'] == self.mejor_params:
                print(f"\n🏆 ¡MEJOR RESULTADO HASTA AHORA!")
            
            # Análisis de la IA
            if decision and 'analisis' in decision:
                print(f"\n🧠 ANÁLISIS DEL OPTIMIZADOR:")
                analisis = decision['analisis']
                if len(analisis) > 250:
                    analisis = analisis[:250] + "..."
                # Dividir en líneas para mejor legibilidad
                for linea in analisis.split('. '):
                    if linea.strip():
                        print(f"   • {linea.strip()}")
                
                print(f"\n💡 DECISIÓN PARA PRÓXIMO RUN:")
                print(f"   ├─ Razonamiento: {decision.get('razonamiento', 'N/A')[:150]}...")
                print(f"   ├─ Nuevo Clip Limit: {decision['nuevos_params']['clip_limit']:.2f}")
                print(f"   ├─ Nuevo Tile Size: {decision['nuevos_params']['tile_size']}")
                print(f"   ├─ Estrategia: {decision.get('estrategia', 'N/A').upper()}")
                print(f"   └─ Confianza: {decision.get('confianza', 0)*100:.0f}%")
        
        # ====================================================================
        # PASO 3: RESUMEN EJECUTIVO
        # ====================================================================
        print(f"\n{'='*80}")
        print("📋 RESUMEN EJECUTIVO DEL PROCESO")
        print(f"{'='*80}\n")
        
        print(f"🎯 OBJETIVO: Minimizar distancia euclidiana al vector promedio")
        print(f"   └─ Vector = [Entropía, Varianza Laplaciano]\n")
        
        print(f"📊 ESTADÍSTICAS:")
        print(f"   ├─ Total de Iteraciones: {len(resultados)}")
        print(f"   ├─ Runs Válidos: {len(resultados_validos)}")
        print(f"   ├─ Runs Inválidos: {len(resultados) - len(resultados_validos)}")
        print(f"   └─ Run Óptimo: #{mejor_run}\n")
        
        print(f"📈 CURVA DE APRENDIZAJE:")
        distancias = [r['evaluacion']['distancia_promedio'] for r in resultados_validos]
        for i, dist in enumerate(distancias, 1):
            run_valido = resultados_validos[i-1]['evaluacion']['run_id']
            barra = "█" * int((1 - dist/distancias[0]) * 40)
            print(f"   Run {run_valido:2d}: {dist:6.2f} {barra}")
        
        print(f"\n🔑 FACTORES CLAVE DE ÉXITO:")
        print(f"   ✓ Validación automática de fórmula CLAHE")
        print(f"   ✓ Aprendizaje de errores (parámetros inválidos)")
        print(f"   ✓ Análisis inteligente con GPT-4")
        print(f"   ✓ Balance exploración-explotación")
        print(f"   ✓ Convergencia en {len(resultados_validos)} iteraciones válidas")
        
        print(f"\n{'='*80}\n")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    API_KEY = cargar_api_key()
    
    # Cargar imagen
    imagen = cv2.imread("src/test1.jpg")
    
    if imagen is None:
        print("⚠️  Usando imagen sintética para demostración")
        imagen = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        imagen = cv2.GaussianBlur(imagen, (5, 5), 0)
    else:
        print(f"✓ Imagen cargada: {imagen.shape}")
    
    # Crear sistema
    sistema = SistemaAutoMejorable(api_key=API_KEY)
    
    # Ejecutar auto-mejora con más iteraciones para explorar
    resultados = sistema.auto_mejorar(
        imagen=imagen,
        num_iteraciones=10,
        params_iniciales={'clip_limit': 2.0, 'tile_size': 8}
    )

     # 💾 Guarda el reporte JSON
    sistema.guardar_reporte_json(resultados, "reporte_optimizacion.json")
    
    # Guardar outputs si hay resultados válidos
    if sistema.mejor_params:
        import os
        os.makedirs("outputs", exist_ok=True)
        
        # Aplicar mejores parámetros
        mejor_imagen, _ = sistema.ejecutor.aplicar_clahe(
            imagen,
            sistema.mejor_params['clip_limit'],
            (sistema.mejor_params['tile_size'], sistema.mejor_params['tile_size'])
        )
        
        if mejor_imagen is not None:
            # Guardar comparación
            if len(imagen.shape) == 3:
                imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gray = imagen
            
            cv2.imwrite("outputs/01_original.jpg", imagen_gray)
            cv2.imwrite("outputs/02_optimizado.jpg", mejor_imagen)
            comparacion = np.hstack([imagen_gray, mejor_imagen])
            cv2.imwrite("outputs/03_comparacion.jpg", comparacion)
            
            print(f"\n✅ Outputs guardados en 'outputs/'")
