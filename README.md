
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

## TAREA 1: detección y clasificación de monedas

El objetivo de esta tarea es **detectar monedas de euro, reconocer su valor y calcular el importe total** utilizando técnicas de visión por computador. Hemos desarrollado dos versiones: una sobre una imagen fija y otra que procesa los fotogramas de una webcam en tiempo real. Ambas utilizan OpenCV, características geométricas y análisis del color, sin recurrir a aprendizaje profundo.

### 1. Detección en una imagen fija

Trabajamos con `imagenes/monedas/Monedas.jpg`. Primero convertimos la imagen a escala de grises y aplicamos un filtro gaussiano para suavizar pequeñas variaciones. Después, **Canny** detecta los bordes y un **cierre morfológico** conecta los tramos que hayan quedado separados. Finalmente, `findContours()` obtiene los contornos externos.

```python
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gris, (5, 5), 0)
bordes = cv2.Canny(suave, 50, 150)
kernel = np.ones((5, 5), np.uint8)
bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel, iterations=2)
contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

No todos los contornos detectados son monedas. Descartamos los que tengan un área inferior al **0,02 %** o superior al **20 %** de la imagen y comprobamos su **circularidad**, calculada como `4 * π * área / perímetro²`. Una circunferencia perfecta tiene circularidad 1; aceptamos valores a partir de `0.65` para tolerar pequeñas irregularidades.

Para los contornos válidos, `cv2.minEnclosingCircle()` calcula el círculo mínimo que los contiene. Su radio permite estimar el diámetro de la moneda en píxeles.

La clasificación combina **color y tamaño**. El color se analiza en HSV mediante una máscara del interior del contorno, distinguiendo tres familias: monedas cobrizas (1, 2 y 5 céntimos), doradas (10, 20 y 50 céntimos) y bimetálicas (1 y 2 euros). Después se compara el diámetro estimado con los diámetros oficiales de las monedas de esa familia. Si el error supera `1.5 mm`, la moneda se marca como **dudosa** y no se suma al total.

**Resultado sobre la imagen fija:**

![Clasificación de monedas en imagen fija](resultados/deteccion_monedas.jpg)

### 2. Detección de monedas en tiempo real

Esta es la parte principal de la tarea. Adaptamos el procesamiento anterior para que se repita continuamente con los fotogramas de la webcam, de forma que el importe mostrado cambie al introducir o retirar monedas.

#### Captura y detección de candidatos

La cámara se abre con `cv2.VideoCapture(1)`. Dentro de un bucle, `vid.read()` devuelve un nuevo fotograma y una variable que indica si se ha podido leer correctamente. Si falla la captura, el programa termina el bucle.

```python
vid = cv2.VideoCapture(1)

while True:
    disponible, img = vid.read()
    if not disponible:
        break

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suave, 50, 150)
    bordes = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel, iterations=2)
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Para cada fotograma se repite el filtrado por área y circularidad utilizado en la imagen fija. Además, se comprueba que el contorno tenga al menos cinco puntos antes de calcular su círculo mínimo. Los candidatos válidos se guardan con su contorno, posición y diámetro en píxeles.

```python
circularidad = (4 * np.pi * area) / (perimetro ** 2)
if circularidad < 0.65 or len(c) < 5:
    continue

centro, radio = cv2.minEnclosingCircle(c)
diametro_px = radio * 2
```

#### Calibración interactiva

El diámetro medido en píxeles depende de la distancia entre la moneda y la cámara. Por eso, **antes de clasificar es necesario calibrar**. Mientras el programa no está calibrado, dibuja los contornos detectados y los numera. El usuario pulsa una tecla del **1 al 9** para seleccionar una moneda conocida y escribe su valor en la consola.

Con el diámetro oficial de esa denominación y el diámetro detectado se obtiene una escala en milímetros por píxel:

```python
mm_por_pixel = diametros_euro[valor_ref] / diametro_ref
calibrado = True
```

Por ejemplo, si la moneda elegida mide 100 píxeles de diámetro y su diámetro real es 25 mm, la escala será `0.25 mm/píxel`. Esta escala se aplica después a las demás monedas del fotograma. Para que funcione correctamente, las monedas deben encontrarse aproximadamente a la misma distancia de la cámara que la referencia.

#### Clasificación por color y diámetro

Una vez calibrado, cada fotograma se convierte a **HSV**, que permite separar tono, saturación y brillo. La función `detectar_color()` crea una máscara del interior de cada moneda, descarta los píxeles demasiado oscuros o brillantes y calcula las proporciones de tonos cobrizos, dorados y metálicos neutros.

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
familia = detectar_color(c["contorno"], hsv, gris.shape)
diametro_mm = c["diametro_px"] * mm_por_pixel
```

El programa agrupa las denominaciones según la familia identificada y busca **el diámetro oficial más próximo** al calculado. De este modo, no compara una moneda cobriza con las denominaciones bimetálicas, por ejemplo.

```python
if familia in familias:
    for valor in familias[familia]:
        error = abs(diametros_euro[valor] - diametro_mm)
        if error < error_minimo:
            error_minimo = error
            valor_probable = valor
```

Solo se acepta la denominación cuando se ha identificado una familia y la diferencia de diámetro es de **1,5 mm o menos**. Las monedas reconocidas aparecen con un contorno **verde** y su valor; las que no cumplen las condiciones se muestran en **naranja**, con la etiqueta `Dudosa`, y no se incluyen en el total.

#### Actualización del resultado y controles

En cada vuelta del bucle se reinician el importe y los contadores. Por tanto, el total representa **las monedas visibles en ese momento**, no una suma acumulada a lo largo del vídeo.

```python
total = 0.0
clasificadas = 0
dudosas = 0

# Tras clasificar las monedas del fotograma:
cv2.putText(vista, f"Total visible: {total:.2f} EUR", (10, 30), font, 0.8, (0, 255, 0), 2)
cv2.imshow("Detector de monedas en tiempo real", vista)
tecla = cv2.waitKey(1) & 0xFF
```

La ventana muestra el **importe total**, los **candidatos detectados** y cuántos han sido **clasificados o marcados como dudosos**. Durante la ejecución se pueden utilizar estos controles:

- **1–9:** seleccionar la moneda de referencia cuando no hay calibración.
- **C:** eliminar la calibración anterior y realizarla de nuevo.
- **ESC:** salir del programa, liberar la cámara y cerrar las ventanas.

**Demostración del funcionamiento en tiempo real:**

![Detección y clasificación de monedas mediante webcam](resultados/monedas_webcam.gif)

### 3. Limitaciones

El resultado depende de la iluminación, los reflejos y la calidad de los contornos. Las monedas solapadas pueden detectarse como un único objeto y los colores pueden variar con la luz. Además, si cambia la distancia a la cámara después de calibrar, la conversión de píxeles a milímetros deja de ser fiable. Por estos motivos, el programa puede marcar algunas monedas como dudosas o confundir denominaciones de tamaños parecidos.



## TAREA: clasificación de microplásticos

Esta tarea consiste en desarrollar un clasificador capaz de distinguir entre tres tipos de partículas: fragmentos (FRA), pellets (PEL) y alquitrán (TAR).

La clasificación debe realizarse mediante características geométricas y/o visuales, sin utilizar modelos de aprendizaje profundo. Para evaluar el clasificador se proporciona una imagen de prueba con sus correspondientes anotaciones, que permitirá calcular distintas métricas y generar una matriz de confusión.

*Pendiente de documentar la implementación.*

### Resultado

*Pendiente de incorporar los resultados, las métricas de evaluación y la matriz de confusión.*

## Fuentes y herramientas utilizadas

- Repositorio completo de la práctica proporcionado por el profesor, usado como enunciado y referencia principal: https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3
- ChatGPT, usado como apoyo para resumir la documentación sobre OpenCV, NumPy y Matplotlib durante el desarrollo de la práctica: https://chatgpt.com/
