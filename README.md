# Práctica 3 - Visión por Computador

Este repositorio contiene la tercera práctica de la asignatura Visión por Computador. Se trabaja con Python, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn y SciPy para detectar objetos, extraer sus características geométricas y visuales y clasificarlos mediante técnicas clásicas de visión por computador.

La práctica se divide en dos tareas:

- **Tarea 1:** detección y clasificación de monedas de euro en una imagen fija y en tiempo real mediante una webcam, con estimación del valor total.
- **Tarea 2:** detección y clasificación de partículas de tres categorías, fragmentos plásticos (FRA), pellets (PEL) y alquitrán (TAR), mediante segmentación, extracción de características geométricas y visuales y un clasificador basado en distancias y reglas heurísticas.

## Autores

- [Pablo Llopis Parrilla](https://github.com/Putrici0)
- [David González Espino](https://github.com/002avid)

## Contenido del repositorio

- `VC_P3.ipynb`: cuaderno principal donde se desarrolla la práctica.
- `imagenes/monedas/`: imágenes utilizadas para la detección y clasificación de monedas.
- `imagenes/microplasticos/`: imágenes completas de entrenamiento, subimágenes de referencia y la imagen de prueba.
- `datos/MPs_test_bbs.csv`: anotaciones de referencia de la imagen de prueba, con las cajas delimitadoras y la clase de cada partícula.
- `resultados/`: carpeta donde se guardan las figuras generadas durante la práctica.

## Instalación y ejecución

No se necesita instalar nada adicional respecto al entorno indicado en el [README de la práctica 3 proporcionada por el profesor](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P3/README.md). Con ese entorno es suficiente para ejecutar el cuaderno `VC_P3.ipynb`.

El cuaderno debe ejecutarse desde el directorio raíz del repositorio para que funcionen las rutas relativas a `imagenes/`, `datos/` y `resultados/`.

La primera tarea dispone de un modo de ejecución con webcam, por lo que se necesita una cámara disponible y permisos para acceder a ella. La segunda tarea utiliza las imágenes y anotaciones almacenadas en el repositorio.

## Tarea 1. Detección y clasificación de monedas

La tarea tiene dos implementaciones: una procesa `imagenes/monedas/Monedas.jpg` y calcula el total de dinero; la otra procesa fotogramas de la webcam. Ambas detectan contornos y estiman diámetros, pero **no clasifican igual**: la imagen fija utiliza también el color para elegir la familia de monedas, mientras que la webcam compara el diámetro con las ocho denominaciones.

### 1.1. Carga, bordes y contornos

Se convierte la imagen a escala de grises y se aplica un suavizado gaussiano antes de detectar bordes:

```python
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gris, (5, 5), 0)
bordes = cv2.Canny(suave, 50, 150)

kernel = np.ones((5, 5), np.uint8)
bordes = cv2.morphologyEx(
    bordes, cv2.MORPH_CLOSE, kernel, iterations=2
)
contornos, _ = cv2.findContours(
    bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

`GaussianBlur((5, 5), 0)` reduce pequeñas variaciones que podrían generar bordes falsos. El tamaño `5 × 5` es la ventana del filtro. En `Canny(suave, 50, 150)`, `50` es el umbral inferior y `150` el superior: los dos controlan qué cambios de intensidad se consideran bordes. El cierre morfológico (`MORPH_CLOSE`) usa un kernel de `5 × 5` y dos iteraciones para unir interrupciones pequeñas. `RETR_EXTERNAL` recupera solo los contornos exteriores, evitando procesar contornos internos de una misma forma.

### 1.2. Filtrado de candidatos

No todos los contornos son monedas. Se descartan formas con área demasiado pequeña o demasiado grande respecto a la imagen y se comprueba su circularidad:

```python
if area < area_img * 0.0002 or area > area_img * 0.20:
    continue

circularidad = (4 * np.pi * area) / (perimetro ** 2)
if circularidad < 0.65:
    continue

centro, radio = cv2.minEnclosingCircle(c)
diametro_px = radio * 2
```

Los límites `0.0002` y `0.20` equivalen al `0,02 %` y al `20 %` del área total: eliminan ruido diminuto y regiones que ocupan gran parte de la imagen. La circularidad se calcula como `4π · área / perímetro²`; un círculo ideal se aproxima a `1` y una forma irregular obtiene un valor menor. El umbral `0.65` tolera imperfecciones sin aceptar demasiadas formas no circulares. Para cada candidato, `minEnclosingCircle` estima el círculo mínimo que contiene el contorno; su diámetro se guarda en píxeles.

### 1.3. Calibración: pasar de píxeles a milímetros

El tamaño en píxeles no equivale directamente a un diámetro físico. Por eso se elige uno de los candidatos, se indica su valor real y se usa su diámetro oficial para calcular la escala:

```python
diametros_euro = {
    0.01: 16.25, 0.02: 18.75, 0.05: 21.25, 0.10: 19.75,
    0.20: 22.25, 0.50: 24.25, 1.00: 23.25, 2.00: 25.75
}

mm_por_pixel = diametros_euro[valor_ref] / diametro_ref
diametro_mm = diametro_px * mm_por_pixel
```

Los valores del diccionario son los diámetros oficiales en milímetros. Si, por ejemplo, una moneda de referencia mide 200 píxeles y su diámetro oficial es 20 mm, la escala resultante sería `0,1 mm/píxel`. La calibración debe hacerse con una moneda bien detectada y situada en el mismo plano que las demás; la perspectiva o una referencia mal seleccionada introducen error en todos los diámetros.

### 1.4. Clasificación de la imagen fija: primero color, después diámetro

La imagen se convierte a HSV para estimar la familia metálica de cada contorno. Se crea una máscara del interior de la moneda y se ignoran los píxeles demasiado oscuros o demasiado brillantes, porque suelen corresponder a sombras o reflejos:

```python
validos = (brillos > 30) & (brillos < 245)

cobrizo = (((tonos <= 12) | (tonos >= 170)) & (saturaciones > 45))
dorado = ((tonos > 12) & (tonos <= 35) & (saturaciones > 45))
metal_neutro = saturaciones < 65
```

En OpenCV, el tono HSV ocupa el rango `0–179`, mientras que saturación y brillo ocupan `0–255`. Los límites de tono `12` y `35` separan aproximadamente los tonos cobrizos y dorados. La saturación mínima `45` evita tratar píxeles casi grises como colores metálicos; `S < 65` sirve para detectar zonas plateadas. No se decide por un único píxel: se calcula la proporción de píxeles de cada tipo dentro de la moneda.

La función `detectar_color()` asigna las familias con estas reglas:

```python
if prop_cobrizo > 0.30 and prop_cobrizo > prop_dorado:
    return "cobre"
if prop_dorado > 0.40 and prop_neutro < 0.40:
    return "dorado"
if prop_dorado > 0.15 and prop_neutro > 0.30:
    return "bimetalica"
return "desconocido"
```

Así, se consideran cobrizas las monedas de 1, 2 y 5 céntimos; doradas, las de 10, 20 y 50 céntimos; y bimetálicas, las de 1 y 2 euros. Los porcentajes `0.30`, `0.40` y `0.15` son proporciones mínimas de píxeles que deben cumplir los criterios; `0.40` en `prop_neutro < 0.40` limita cuánto metal neutro puede tener una moneda clasificada como dorada. Son umbrales heurísticos dependientes de la iluminación y de la imagen.

Una vez determinada la familia, solo se comparan los diámetros oficiales de sus denominaciones. Se elige la denominación con menor diferencia y se acepta si el error no supera `1.5 mm`:

```python
error = abs(diametros_euro[valor] - diametro_mm)

if valor_probable is not None and error_minimo <= tolerancia_mm:
    total += valor_probable
    clasificadas += 1
else:
    dudosas += 1
```

La tolerancia `1.5 mm` permite pequeñas imprecisiones de borde y calibración. Si la familia es desconocida o ningún diámetro se acerca lo suficiente, la moneda se marca como dudosa y no se suma. Esto evita forzar una denominación cuando los datos no son consistentes.

### Resultado de la imagen fija

En la ejecución incluida en el notebook se detectaron **8 candidatos**, se clasificaron **8 monedas**, no hubo monedas dudosas y el total estimado fue **3,88 €**.

![Resultado de la detección y clasificación de monedas](resultados/deteccion_monedas.jpg)

### 1.5. Webcam: funcionamiento en tiempo real

La versión de webcam usa `cv2.VideoCapture(0)` para abrir la cámara predeterminada y ejecuta el procesamiento fotograma a fotograma. A diferencia de la imagen fija, no aplica suavizado gaussiano antes de Canny y usa umbrales más bajos:

```python
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(gris, 25, 120)
bordes = cv2.morphologyEx(
    bordes, cv2.MORPH_CLOSE, kernel, iterations=2
)
contornos, _ = cv2.findContours(
    bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

`Canny(25, 120)` intenta conservar bordes con menor contraste que los de la imagen fija, útil cuando el vídeo tiene iluminación variable; también puede aumentar el ruido. El filtro de área se mantiene (`0.0002`–`0.20` del área del fotograma), pero el umbral de circularidad sube de `0.65` a `0.70`, descartando formas menos redondas.

El flujo de uso está pensado para calibrar antes de clasificar:

1. Antes de calibrar, se dibujan los contornos candidatos y se numeran en pantalla.
2. Se pulsa una tecla del `1` al `9` para elegir el candidato que servirá como referencia.
3. En la consola se introduce el valor de esa moneda (`0.01`, `0.02`, `0.05`, `0.10`, `0.20`, `0.50`, `1` o `2`). El programa calcula `mm_por_pixel` a partir del diámetro oficial y del diámetro detectado.
4. En cada fotograma posterior, convierte el diámetro de cada candidato a milímetros, compara con las ocho denominaciones y suma las que estén dentro de la tolerancia de `1.5 mm`.
5. Se pulsa `C` para volver a calibrar y `ESC` para liberar la cámara y cerrar la ventana.

La comparación de la webcam es distinta de la imagen fija: **no usa HSV para separar las familias cobre, dorado y bimetálica**. Compara directamente con las ocho denominaciones. Por eso, monedas de diámetros próximos pueden confundirse y la calibración influye mucho en el resultado. Además, la escala supone que las monedas están aproximadamente a la misma distancia y plano que la moneda de referencia; al acercar una moneda a la cámara, su tamaño en píxeles cambia y la estimación deja de ser fiable.

### Resultado en tiempo real

![Clasificación de monedas en tiempo real](resultados/monedas_webcam.gif)

## Tarea 2. Detección y clasificación de partículas

La imagen de prueba es `imagenes/microplasticos/MPs_test.jpg`; las cajas y etiquetas reales se encuentran en `datos/MPs_test_bbs.csv`. Se distinguen `FRA` (fragmentos), `PEL` (pellets) y `TAR` (alquitrán, clase que aparece en los datos aunque no sea un microplástico propiamente dicho).

El clasificador es una implementación propia basada en distancias ponderadas a muestras etiquetadas, que promedia las tres muestras más cercanas de cada clase y después aplica reglas heurísticas de color y forma para revisar algunos casos.

### 2.1. Segmentación: crear una máscara de partículas

La función `detectar()` crea varias máscaras para encontrar partículas claras, oscuras o de color distinto al fondo. Primero compara una versión suavizada con una estimación más amplia del fondo:

```python
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gris, (5, 5), 0)
fondo = cv2.GaussianBlur(gris, (51, 51), 0)
claras = cv2.subtract(suave, fondo)
oscuras = cv2.subtract(fondo, suave)
```

El filtro `5 × 5` conserva los detalles locales; el de `51 × 51` genera un fondo mucho más suave. `suave - fondo` resalta zonas más claras que el entorno, mientras que `fondo - suave` resalta las más oscuras. En ambas diferencias se calcula un umbral con Otsu y se impone un mínimo de `8.5`, para no aceptar cambios de intensidad muy pequeños.

Se añaden otras dos máscaras: una basada en diferencias de saturación HSV y otra basada en la distancia entre el color LAB de cada píxel y una estimación del fondo local. En la máscara LAB, una distancia superior a `10` se considera diferencia de color. Las cuatro máscaras se combinan mediante OR y se aplica un cierre pequeño:

```python
mascara = cv2.bitwise_or(
    cv2.bitwise_or(m1, m2),
    cv2.bitwise_or(m3, m4)
)
mascara = cv2.morphologyEx(
    mascara, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
)
```

La unión permite que una partícula se detecte por contraste claro/oscuro o por color. El kernel elíptico de `3 × 3` cierra pequeños huecos, intentando no unir partículas vecinas. Después se extraen contornos externos. Los valores `51`, `20` y `10` controlan la escala de comparación con el fondo y la sensibilidad de las máscaras; afectan directamente a qué regiones se detectan.

### 2.2. Separar partículas que se tocan

Un contorno puede contener varias partículas pegadas. `separar_grupos()` solo intenta dividir contornos con área de al menos `350 px²` y ancho y alto de al menos `22 px`. Calcula la transformada de distancia, donde los valores altos se encuentran en el interior de cada objeto, y selecciona núcleos por encima de `max(3, 0.52 * maximo)`. Se descartan núcleos de menos de `7` píxeles y solo se intenta dividir si aparecen entre 2 y 6 centros. Los píxeles se asignan al centro más cercano y la división se acepta si las nuevas regiones conservan más del `65 %` del área original.

Hay una segunda función, `separar_alargados()`, para grupos grandes y alargados que no se separaron en la primera pasada. Solo considera contornos con área de al menos `400 px²`, caja de al menos `1500 px²` y relación entre lado largo y corto de `1.8` o más. Busca máximos locales usando una ventana de `13 × 13`, exige una distancia mínima de `12 px` entre centros y acepta la división si conserva al menos el `70 %` del área. Estos límites son deliberadamente conservadores: se busca separar grupos evidentes sin fragmentar objetos normales.

### 2.3. Características: representar cada contorno con números

La función `caracteristicas()` convierte cada partícula en un vector de **20 valores**. Antes, descarta contornos con área menor que `20 px²`, mayor que `5000 px²` o perímetro cero. En entrenamiento también elimina los que tocan el borde; en test se permite conservarlos salvo los demasiado pequeños, porque una partícula puede quedar cortada por el encuadre.

Las primeras características describen la forma:

```python
return [p*p/(4*np.pi*a), a/(w*h), min(w, h)/max(w, h),
        relacion_ejes, regularidad, a/hull if hull else 0,
        # ... características de brillo y color
]
```

`p*p/(4πa)` es la compacidad: vale cerca de `1` para un círculo ideal y aumenta con formas más irregulares. `a/(w*h)` mide qué proporción de su rectángulo ocupa el contorno. `min(w,h)/max(w,h)` se acerca a `1` en formas cuadradas o compactas y disminuye en formas alargadas. `relacion_ejes` procede de la elipse ajustada; `regularidad` compara las distancias de los puntos del contorno al centro; `a/hull` compara el área con la envolvente convexa.

El resto del vector mide apariencia: brillo relativo respecto al anillo de fondo cercano, saturación y variación de intensidad; tono y fracción de píxeles con color; diferencias de los canales `a` y `b` del espacio LAB entre partícula y fondo; medianas de color; y proporciones de tonos crema/beige y amarillos. Se usan tanto forma como color porque un pellet puede ser compacto y claro, mientras que un fragmento suele tener una forma más irregular. Las diferencias respecto al fondo ayudan a reducir la dependencia de la iluminación general.

### 2.4. Entrenamiento y algoritmo de clasificación

Las muestras de referencia se extraen de las imágenes de entrenamiento de cada clase y de sus subimágenes. Las imágenes completas se reducen a `25 %` del ancho y alto (`fx=.25`, `fy=.25`) con `INTER_AREA`, lo que reduce el coste del procesamiento. Cada contorno válido aporta un vector de características y una etiqueta `FRA`, `PEL` o `TAR`.

El clasificador principal se implementa en `predecir_distancia()`. **No es Random Forest, SVM ni un `KNeighborsClassifier` de scikit-learn**: calcula manualmente distancias ponderadas entre el vector nuevo y todos los vectores de entrenamiento:

```python
pesos = np.array([1,1,1,1,1,1,2,2,2,0,0,0,0,0,0,0,0,0,0,0])
media = Xref.mean(axis=0)
desv = Xref.std(axis=0) + 1e-6
A = (Xref - media) / desv
B = (Xnuevo - media) / desv

for f in B:
    dist = np.sqrt(np.sum(pesos * (A - f)**2, axis=1))
    d = {c: np.mean(np.sort(dist[yref == c])[:3]) for c in clases}
    pred.append(min(d, key=d.get))
```

La estandarización resta la media y divide por la desviación típica de cada característica, para que magnitudes distintas sean comparables. Se suma `1e-6` a la desviación para evitar divisiones por cero. Los pesos `1` dan importancia normal a las seis características geométricas iniciales; los pesos `2` duplican la contribución de brillo relativo, saturación y variación de intensidad; los pesos `0` dejan fuera del cálculo de distancia las características restantes, que se aprovechan en las reglas heurísticas posteriores. Para cada clase se calcula la media de las **tres distancias más pequeñas** y se asigna la clase con menor media. Esto reduce la dependencia de una sola muestra de referencia.

Después, `predecir_material()` corrige algunos resultados usando reglas de color y forma basadas en percentiles de las muestras `PEL` y `TAR`. Por ejemplo, puede asignar `TAR` si el brillo relativo es bajo y la señal amarilla es inferior al percentil 20 de pellets. Puede asignar `PEL` cuando las señales amarilla y beige, el relleno del rectángulo, la compacidad y el brillo cumplen los límites aprendidos de las muestras. Si la forma parece un fragmento irregular, evita convertirlo en pellet salvo que el color de pellet sea suficientemente fuerte.

Los parámetros finales son `seleccion = (50, 65)`, que se pasan a `fuerza_fra` y `proteccion_pel`. `50` selecciona percentiles intermedios para los límites de irregularidad y relleno usados al detectar fragmentos; `65` exige una evidencia de color más alta para proteger pellets claros de ser reclasificados como fragmentos. Son reglas heurísticas, no hiperparámetros de Random Forest. El código indica que se eligieron previamente con las imágenes de entrenamiento; las etiquetas del CSV de test solo se consultan después de obtener las predicciones.

### 2.5. Evaluación: emparejar detecciones con las anotaciones

La evaluación compara las cajas detectadas con las cajas del CSV. Para cada pareja calcula IoU (*Intersection over Union*): área de intersección dividida por área de unión. Una pareja se considera coincidente si `IoU >= 0.30`.

```python
umbral_iou = 0.30
recompensa = np.where(solapes >= umbral_iou, 2.0 + solapes, 0.0)
filas, cols = linear_sum_assignment(-recompensa)
```

La asignación húngara (`linear_sum_assignment`, de SciPy) empareja cajas sin reutilizar una misma detección o anotación. Las parejas por debajo de `0.30` reciben recompensa cero; las válidas reciben `2 + IoU`, de modo que se prioriza obtener el máximo número de coincidencias válidas y, después, un mayor solapamiento. Solo después de emparejar se comparan las etiquetas reales y predichas para calcular las métricas de clasificación. Así se distinguen errores de detección de errores de clase.

### Resultados de test

En la ejecución registrada, el sistema encontró **119 detecciones** frente a **120 partículas anotadas**. Se emparejaron **102 detecciones** con anotaciones y **77** de esas parejas tenían la clase correcta. La exactitud entre las parejas es `77/102 = 75,5 %`; si se cuentan también como fallos las partículas reales no emparejadas, el acierto respecto al total anotado es `77/120 = 64,2 %`.

![Comparación entre las anotaciones reales y las predicciones](resultados/microplasticos_test.png)

La matriz de confusión permite ver qué clases se confunden. En esta ejecución, `FRA` obtuvo precisión aproximada de `0,77` y recall de `0,88`; `PEL`, precisión de `0,65` y recall de `0,52`; y `TAR`, precisión de `0,90` y recall de `0,82`. La confusión más destacada es entre pellets y fragmentos. `TAR` tiene pocos ejemplos, así que sus métricas son menos estables.

![Matriz de confusión de la clasificación de partículas](resultados/matriz_confusion_microplasticos.png)

El resultado final depende de dos etapas: la segmentación debe encontrar una región y separarla correctamente, y el clasificador debe asignarle la clase adecuada. Si una partícula no se detecta o no se empareja con una anotación, no puede contarse como clasificación correcta.

## Fuentes y herramientas utilizadas

- Repositorio completo de la práctica proporcionado por el profesor, usado como enunciado y referencia principal: https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3
- ChatGPT-6, usado como apoyo para resumir la documentación sobre OpenCV, NumPy, Matplotlib y scikit-learn durante el desarrollo de la práctica: https://chatgpt.com/