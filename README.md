
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

En esta tarea se desarrolla un detector de monedas de euro en tiempo real utilizando la webcam. El objetivo es identificar las monedas presentes en la escena y calcular su valor total mediante características geométricas y de color, sin utilizar modelos de aprendizaje profundo.

El programa utiliza `cv2.VideoCapture(1)` para capturar los fotogramas de la cámara y procesarlos individualmente. Antes de clasificar las monedas, es necesario realizar una calibración que permita relacionar sus dimensiones en píxeles con los diámetros reales en milímetros.

### Detección de monedas

Para localizar las monedas, cada fotograma se convierte a escala de grises mediante `cv2.cvtColor()`, ya que inicialmente interesa detectar sus formas y no sus colores. Después se aplica `cv2.GaussianBlur()` con un kernel de `5x5`, que suaviza la imagen y reduce pequeñas variaciones de intensidad que podrían generar bordes innecesarios.

Sobre la imagen suavizada se utiliza `cv2.Canny(suave, 50, 150)` para detectar los bordes. El valor `50` representa el umbral inferior, que permite conservar bordes débiles conectados a otros más fuertes, mientras que `150` es el umbral superior, utilizado para identificar los bordes más definidos.

Posteriormente se aplica `cv2.morphologyEx()` con `cv2.MORPH_CLOSE`, utilizando un kernel de `5x5` y dos iteraciones. Esta operación permite cerrar pequeñas discontinuidades entre los bordes, facilitando que los contornos de las monedas aparezcan más completos.

A continuación, `cv2.findContours()` obtiene los contornos externos mediante `cv2.RETR_EXTERNAL`, evitando recuperar contornos interiores que no son necesarios para localizar las monedas.

No todos los contornos detectados corresponden a monedas, por lo que se aplican varios filtros:

- **Área:** se descartan los contornos cuya superficie sea inferior al `0.02 %` o superior al `20 %` del área de la imagen. Esto permite eliminar objetos demasiado pequeños, que podrían ser ruido, y regiones excesivamente grandes.
- **Circularidad:** se calcula mediante `4 * π * área / perímetro²`. Una circunferencia ideal tiene un valor de 1, mientras que las formas menos circulares presentan valores inferiores. Se utiliza un mínimo de `0.65` para admitir contornos aproximadamente circulares, sin exigir una forma perfecta.
- **Número de puntos:** se descartan los contornos con menos de cinco puntos, como comprobación adicional antes de calcular sus dimensiones.

Los valores anteriores actúan como filtros para reducir las detecciones incorrectas, aunque también pueden descartar monedas si sus contornos aparecen incompletos o deformados.

Finalmente, para cada contorno válido se utiliza `cv2.minEnclosingCircle()`, que calcula el círculo mínimo que lo contiene. Su radio permite obtener el diámetro en píxeles mediante `diametro_px = radio * 2`.

### Calibración

El diámetro de una moneda en píxeles depende de la distancia a la cámara, por lo que no puede compararse directamente con sus dimensiones reales.

Para resolverlo, el programa permite seleccionar una moneda de referencia. Antes de calibrar, los contornos detectados se muestran en verde y se numeran. El usuario puede pulsar una tecla del `1` al `9` para seleccionar una de las monedas e introducir su valor real en euros mediante la consola.

El programa dispone de un diccionario llamado `diametros_euro`, que relaciona cada denominación con su diámetro oficial en milímetros. Por ejemplo, una moneda de 1 euro tiene un diámetro de `23.25 mm` y una de 2 euros de `25.75 mm`.

Conociendo el diámetro real de la moneda seleccionada y su diámetro detectado en píxeles, se calcula la escala:

`mm_por_pixel = diametro_real_mm / diametro_detectado_px`

Esta relación permite convertir posteriormente los diámetros de las demás monedas a milímetros mediante `diametro_mm = diametro_px * mm_por_pixel`.

Una vez establecida la referencia, el programa comienza a clasificar las monedas. La calibración presupone que todas se encuentran aproximadamente a la misma distancia de la cámara. Si esta distancia cambia, los diámetros estimados pueden dejar de ser correctos.

La tecla `C` permite repetir la calibración y `ESC` finaliza la ejecución.

### Clasificación de las monedas

Para identificar el valor de cada moneda se combinan dos características: su diámetro estimado y su color.

En primer lugar, el fotograma se convierte al espacio HSV mediante `cv2.cvtColor()`. Este espacio separa el tono, la saturación y el brillo, facilitando la identificación de los distintos acabados metálicos.

La función `detectar_color()` utiliza `cv2.drawContours()` para crear una máscara que selecciona únicamente los píxeles del interior de cada moneda. Después se analizan sus valores HSV, descartando aquellos cuyo brillo sea inferior a `30` o superior a `245`, con el objetivo de reducir la influencia de sombras intensas y reflejos.

A partir de los píxeles restantes se calculan las proporciones de tres grupos de color:

- **Cobrizo:** tonos entre `0-12` o `170-179`, con saturación superior a `45`.
- **Dorado:** tonos entre `13-35`, también con saturación superior a `45`.
- **Metálico neutro:** píxeles con saturación inferior a `65`, asociados a zonas de apariencia plateada o grisácea.

Estas condiciones permiten agrupar las monedas en tres familias. Para considerar una moneda cobriza se exige una proporción superior al `30 %` y mayor que la proporción dorada. Para las doradas se requiere más de un `40 %` de píxeles dorados y menos de un `40 %` de metálicos neutros. Finalmente, las bimetálicas deben presentar más de un `15 %` de píxeles dorados y un `30 %` de metálicos neutros.

Estos porcentajes son umbrales utilizados para distinguir los acabados de las monedas. No representan probabilidades de clasificación y pueden verse afectados por las condiciones de iluminación.

Una vez identificada la familia, se limita la comparación a las denominaciones compatibles:

- **Cobre:** 1, 2 y 5 céntimos.
- **Dorado:** 10, 20 y 50 céntimos.
- **Bimetálica:** 1 y 2 euros.

Para determinar el valor concreto, se calcula la diferencia absoluta entre el diámetro estimado y el diámetro oficial de cada moneda de esa familia:

`error = abs(diametro_oficial - diametro_estimado)`

Se selecciona la denominación que presenta el menor error, siempre que este no supere `tolerancia_mm = 1.5`.

Esta tolerancia permite admitir pequeñas diferencias en la detección del contorno o en la calibración. Sin embargo, una tolerancia demasiado grande podría provocar confusiones entre monedas de tamaños similares, mientras que una demasiado pequeña podría rechazar monedas correctamente detectadas.

Si no se identifica una familia de color válida o ninguna denominación cumple la tolerancia establecida, la moneda se considera **dudosa** y no se suma al importe total.

### Resultado

El programa muestra en tiempo real los contornos de las monedas detectadas junto con su valor estimado. Las monedas clasificadas aparecen resaltadas en verde, mientras que aquellas que no cumplen los criterios de clasificación se muestran en naranja con la etiqueta `Dudosa`.

En la parte superior de la ventana se presenta el importe total visible, el número de monedas detectadas y la cantidad de monedas clasificadas y dudosas.

El importe se calcula de nuevo en cada fotograma, por lo que representa el dinero presente en la escena en ese momento y no una suma acumulada durante toda la ejecución.

*Pendiente de incorporar una demostración visual y las observaciones obtenidas durante las pruebas.*

## TAREA: clasificación de microplásticos

Esta tarea consiste en desarrollar un clasificador capaz de distinguir entre tres tipos de partículas: fragmentos (FRA), pellets (PEL) y alquitrán (TAR).

La clasificación debe realizarse mediante características geométricas y/o visuales, sin utilizar modelos de aprendizaje profundo. Para evaluar el clasificador se proporciona una imagen de prueba con sus correspondientes anotaciones, que permitirá calcular distintas métricas y generar una matriz de confusión.

*Pendiente de documentar la implementación.*

### Resultado

*Pendiente de incorporar los resultados, las métricas de evaluación y la matriz de confusión.*

## Fuentes y herramientas utilizadas

- Repositorio completo de la práctica proporcionado por el profesor, usado como enunciado y referencia principal: https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3
- ChatGPT, usado como apoyo para resumir la documentación sobre OpenCV, NumPy y Matplotlib durante el desarrollo de la práctica: https://chatgpt.com/
