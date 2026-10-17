
# Práctica 3 - Visión por Computador

Este repositorio contiene la tercera práctica de la asignatura Visión por Computador. En esta práctica se trabaja con Python, OpenCV, NumPy y Matplotlib para detectar objetos, analizar sus características geométricas y clasificarlos a partir de imágenes.

Las tareas planteadas consisten en identificar monedas y calcular su valor total, así como desarrollar un clasificador de partículas de microplásticos utilizando características geométricas y visuales, sin emplear modelos de aprendizaje profundo.

## Autores

- [Pablo Llopis Parrilla](https://github.com/Putrici0)
- [David González Espino](https://github.com/002avid)

## Contenido del repositorio

- `VC_P3.ipynb`: cuaderno principal donde se desarrolla la práctica y se encuentran las distintas tareas realizadas.
- `imagenes/monedas/`: imágenes utilizadas en la detección y clasificación de monedas.
- `imagenes/microplasticos/`: imágenes de prueba y referencia para microplásticos.
- `datos/`: CSV con las anotaciones de las imágenes de microplásticos.

## Instalación y ejecución

No se necesita instalar nada adicional respecto al entorno indicado en el [README de la práctica original proporcionada por el profesor](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P3/README.md). Con ese entorno es suficiente para ejecutar el cuaderno `VC_P3.ipynb`.

Para la tarea de detección de monedas en tiempo real, el equipo debe disponer de una webcam y tener habilitados los permisos necesarios para acceder a ella.

## TAREA: detección y clasificación de monedas

El objetivo de esta tarea es identificar monedas de euro, determinar su valor y calcular el importe total presente en una imagen. Primero se trabaja con una fotografía fija (`imagenes/monedas/Monedas.jpg`) y después se adapta la solución para procesar imágenes capturadas en tiempo real mediante una webcam.

Para distinguir las denominaciones utilizamos **características geométricas y de color**, principalmente la circularidad, el diámetro y los tonos de la superficie. No se emplean modelos de aprendizaje profundo. Una parte importante del ejercicio es la **calibración**: el tamaño de una moneda en píxeles varía según su distancia a la cámara, por lo que necesitamos una referencia de tamaño conocido para estimar sus dimensiones reales.

### 1. Detección de monedas en una imagen fija

#### Carga y preparación de la imagen

Se carga la fotografía con `cv2.imread()`, se convierte a escala de grises y se suaviza con un filtro gaussiano:

```python
img = cv2.imread("imagenes/monedas/money.jpg")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gris, (5, 5), 0)
```

La conversión a gris simplifica la detección inicial: para encontrar el contorno de las monedas no necesitamos todavía distinguir entre cobres, dorados y plateados. El filtro `GaussianBlur()` reduce variaciones pequeñas y ruido de la imagen, haciendo que la detección de bordes sea más estable.

A continuación, se utiliza Canny para encontrar cambios de intensidad que puedan corresponder a los bordes de las monedas. Después aplicamos un **cierre morfológico**, que ayuda a unir interrupciones pequeñas del borde:

```python
bordes = cv2.Canny(suave, 50, 150)

kernel = np.ones((5, 5), np.uint8)
bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel, iterations=2)

contornos, _ = cv2.findContours(
    bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

En Canny, `50` y `150` son los umbrales inferior y superior utilizados para decidir qué cambios de intensidad se consideran bordes. El cierre morfológico (`MORPH_CLOSE`) consiste en una dilatación seguida de una erosión. Usamos un kernel de `5 × 5` y dos iteraciones. Finalmente, `findContours()` obtiene únicamente los **contornos externos** (`RETR_EXTERNAL`), mientras que `CHAIN_APPROX_SIMPLE` reduce los puntos necesarios para representar cada contorno.

#### Filtrado de los contornos

No todos los bordes pertenecen a monedas: también pueden aparecer sombras, reflejos u objetos del fondo. Por eso, para cada contorno calculamos su área y perímetro:

```python
area = cv2.contourArea(c)
perimetro = cv2.arcLength(c, True)

if perimetro == 0:
    continue

if area < area_img * 0.0002 or area > area_img * 0.20:
    continue

circularidad = (4 * np.pi * area) / (perimetro ** 2)

if circularidad < 0.65:
    continue
```

El filtro de área descarta regiones que ocupan menos del **0,02 %** o más del **20 %** de la imagen. De este modo evitamos procesar contornos excesivamente pequeños o grandes. Después utilizamos la **circularidad**, una medida que vale aproximadamente `1` para un círculo perfecto y disminuye cuanto más se aleja la forma de un círculo. El límite elegido, `0.65`, permite admitir monedas cuyos bordes no hayan quedado completamente regulares.

Para los contornos aceptados, calculamos el círculo mínimo que los contiene:

```python
centro, radio = cv2.minEnclosingCircle(c)
diametro_px = radio * 2
```

Este cálculo proporciona el centro y un diámetro estimado en píxeles. Guardamos estos datos junto con el contorno para utilizarlos en la calibración y clasificación. Es importante recordar que `minEnclosingCircle()` devuelve el círculo que contiene el contorno, por lo que su diámetro puede resultar mayor que el real si hay ruido o partes del fondo unidas a la moneda.

#### Selección de una moneda de referencia y calibración

Las monedas candidatas se dibujan en verde y se numeran sobre la fotografía. El programa muestra sus diámetros detectados y solicita seleccionar una moneda cuyo valor conozcamos:

```python
indice_ref = int(input("Número de la moneda de referencia: ")) - 1
valor_ref = float(input("Valor de la moneda (ej. 0.50, 1 o 2): ").replace(",", "."))
```

Utilizamos un diccionario (`diametros_euro`) que relaciona cada denominación con su diámetro expresado en milímetros. Por ejemplo, en el código la moneda de 1 euro mide `23.25 mm` y la de 2 euros `25.75 mm`. Con el diámetro real de la referencia y el diámetro medido en la imagen obtenemos la relación de escala:

```python
mm_por_pixel = diametros_euro[valor_ref] / diametro_ref
```

Esta operación nos indica cuántos milímetros representa un píxel en la fotografía. A partir de ella estimamos el diámetro de cualquier otra moneda:

```python
diametro_mm = c["diametro_px"] * mm_por_pixel
```

La calibración funciona como una aproximación siempre que las monedas estén situadas **aproximadamente en el mismo plano y a la misma distancia de la cámara**. Si una moneda está mucho más cerca o más lejos, su tamaño en píxeles cambia y la comparación deja de ser fiable.

#### Identificación de la familia por color

Solo con el diámetro no siempre resulta fácil distinguir denominaciones de tamaños similares. Por ello también analizamos el color. Convertimos la imagen de BGR a **HSV**, un espacio que separa el tono (`H`), la saturación (`S`) y el brillo (`V`):

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

La función `detectar_color()` crea una máscara con la forma del contorno. De esta manera calcula las propiedades de los píxeles que están dentro de la moneda y no de toda la imagen:

```python
mascara = np.zeros(forma, dtype=np.uint8)
cv2.drawContours(mascara, [moneda], -1, 255, -1)
pixeles = hsv[mascara == 255]
```

Se separan los canales HSV y se descartan los píxeles extremadamente oscuros o brillantes (`30 < V < 245`), porque las sombras y los reflejos pueden alterar la identificación del metal. Con los píxeles restantes calculamos tres proporciones:

```python
cobrizo = (((tonos <= 12) | (tonos >= 170)) & (saturaciones > 45))
dorado = ((tonos > 12) & (tonos <= 35) & (saturaciones > 45))
metal_neutro = saturaciones < 65

prop_cobrizo = np.mean(cobrizo)
prop_dorado = np.mean(dorado)
prop_neutro = np.mean(metal_neutro)
```

Como cada expresión es una máscara de valores verdaderos o falsos, `np.mean()` obtiene la proporción de píxeles que cumplen cada condición. Los porcentajes no son probabilidades aprendidas, sino **umbrales definidos en el programa** para distinguir los acabados de las monedas.

Se consideran tres familias:

- **Cobre:** monedas de **1, 2 y 5 céntimos**. Se exige una proporción cobriza superior al `30 %` y mayor que la dorada.
- **Dorado:** monedas de **10, 20 y 50 céntimos**. La proporción dorada debe superar el `40 %` y la neutra quedar por debajo del `40 %`.
- **Bimetálica:** monedas de **1 y 2 euros**. Deben aparecer tanto tonos dorados (`> 15 %`) como zonas metálicas poco saturadas (`> 30 %`).

Si ninguna condición se cumple, la función devuelve `"desconocido"`. Esta decisión evita asignar un valor a una moneda cuyo color no se corresponde claramente con alguna de las familias.

#### Clasificación por diámetro y cálculo del total

Después de identificar la familia, el programa compara el diámetro estimado **solo con las denominaciones compatibles con ese color**:

```python
familias = {
    "cobre": [0.01, 0.02, 0.05],
    "dorado": [0.10, 0.20, 0.50],
    "bimetalica": [1.00, 2.00]
}

error = abs(diametros_euro[valor] - diametro_mm)
```

Se elige la denominación con la diferencia absoluta más pequeña respecto al diámetro oficial. Sin embargo, solo se acepta si el error no supera la **tolerancia de `1.5 mm`**. Cuando no se identifica la familia o el diámetro queda fuera de esa tolerancia, la moneda se marca como **«Dudosa»** y no se añade al total.

```python
if valor_probable is not None and error_minimo <= tolerancia_mm:
    total += valor_probable
    clasificadas += 1
else:
    dudosas += 1
```

Finalmente, se dibujan en **verde** las monedas clasificadas y en **naranja** las dudosas. Se muestra la fotografía resultante y se imprime el número de candidatos, el número de monedas clasificadas, las dudosas y el importe total estimado.

### 2. Detección y clasificación de monedas en tiempo real

Tras resolver la imagen fija, adaptamos el mismo procedimiento a una cámara para analizar los fotogramas de forma continua. La mayor parte del proceso se mantiene: escala de grises, suavizado gaussiano, Canny, cierre morfológico, búsqueda de contornos, filtros de área y circularidad, análisis de color y comparación de diámetros.

#### Captura y procesamiento de los fotogramas

El programa abre la webcam y entra en un bucle que obtiene un fotograma en cada iteración:

```python
vid = cv2.VideoCapture(1)

while True:
    disponible, img = vid.read()

    if not disponible:
        print("No se ha podido leer el fotograma.")
        break
```

El índice `1` es el dispositivo de captura utilizado en el código; en otros equipos podría ser necesario cambiarlo por `0`. En cada fotograma se vuelven a detectar los contornos, se filtran y se guardan las monedas candidatas. Esta repetición es necesaria porque las monedas pueden desplazarse, entrar o salir del campo de visión.

#### Calibración interactiva

Al comenzar, el programa se encuentra **sin calibrar**. En la ventana se dibujan los contornos detectados y se numeran para que el usuario seleccione una moneda conocida mediante las teclas del `1` al `9`. Después se solicita su denominación en la consola y se calcula la escala `mm_por_pixel`.

La variable `calibrado` controla las dos fases:

```python
calibrado = False
mm_por_pixel = None

if not calibrado:
    # Se muestran los candidatos numerados.
else:
    # Se clasifican las monedas y se calcula el total.
```

Una vez calibrado, el programa utiliza esa escala para procesar los siguientes fotogramas. Si cambia la distancia entre la cámara y las monedas, podemos pulsar **`C`** para volver a seleccionar una referencia. La tecla **`ESC`** finaliza la ejecución.

#### Clasificación y visualización del dinero presente

Con la calibración realizada, el programa convierte cada fotograma a HSV, llama a `detectar_color()` para cada contorno y compara su diámetro con las denominaciones de la familia correspondiente. Las monedas reconocidas aparecen con su valor en verde y las que no cumplen las condiciones aparecen en naranja con la etiqueta `Dudosa`.

En la parte superior de la ventana se muestran el **importe total visible**, el número de monedas candidatas, las clasificadas y las dudosas. El importe se calcula desde cero en cada fotograma:

```python
total = 0.0
clasificadas = 0
dudosas = 0

for c in candidatos:
    # Clasificación de las monedas del fotograma actual.
    ...
```

Por tanto, no se acumula dinero entre fotogramas: la cifra representa una estimación del valor de las monedas **que aparecen en ese instante**.

Al terminar se liberan los recursos utilizados:

```python
vid.release()
cv2.destroyAllWindows()
```

### 3. Resultados y limitaciones

La solución permite detectar monedas aproximadamente circulares, estimar su diámetro real a partir de una referencia y clasificarlas combinando tamaño y color. También distingue entre monedas a las que asigna una denominación y detecciones dudosas, evitando incluir estas últimas en el importe total.

No obstante, su funcionamiento depende de las condiciones de captura:

- **Iluminación y reflejos:** pueden modificar los tonos HSV y provocar confusiones entre acabados metálicos.
- **Calibración:** la escala depende de la distancia y del tamaño aparente de la moneda seleccionada. Si la cámara se mueve o las monedas están a distintas alturas, el diámetro estimado puede variar.
- **Fondo y bordes:** un fondo con textura, sombras intensas o bordes poco definidos puede generar falsos contornos o impedir que se detecte alguna moneda.
- **Monedas juntas o solapadas:** sus contornos pueden unirse y ser interpretados como un solo objeto, que quizá sea descartado por los filtros de forma.
- **Tolerancias fijas:** los valores elegidos para circularidad, color y diámetro funcionan como criterios heurísticos y pueden necesitar ajustes en otras condiciones.

Por ello, el importe calculado debe interpretarse como **una estimación**, no como un conteo infalible. La propuesta cumple el objetivo de trabajar con operaciones de procesamiento de imagen y características geométricas y visuales, sin recurrir a aprendizaje profundo.

> **Demostración visual:** incorporar aquí una captura de la imagen fija clasificada y otra del programa funcionando con la webcam. Son útiles para mostrar el resultado real, junto con algún ejemplo de detección dudosa o fallo observado durante las pruebas.


## TAREA: clasificación de microplásticos

Esta tarea consiste en desarrollar un clasificador capaz de distinguir entre tres tipos de partículas: fragmentos (FRA), pellets (PEL) y alquitrán (TAR).

La clasificación debe realizarse mediante características geométricas y/o visuales, sin utilizar modelos de aprendizaje profundo. Para evaluar el clasificador se proporciona una imagen de prueba con sus correspondientes anotaciones, que permitirá calcular distintas métricas y generar una matriz de confusión.

*Pendiente de documentar la implementación.*

### Resultado

*Pendiente de incorporar los resultados, las métricas de evaluación y la matriz de confusión.*

## Fuentes y herramientas utilizadas

- Repositorio completo de la práctica proporcionado por el profesor, usado como enunciado y referencia principal: https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3
- ChatGPT, usado como apoyo para resumir la documentación sobre OpenCV, NumPy y Matplotlib durante el desarrollo de la práctica: https://chatgpt.com/
