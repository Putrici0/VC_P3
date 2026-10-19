
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

# TAREA 1: detección y clasificación de monedas

El objetivo es detectar monedas de euro en una imagen, estimar su diámetro y calcular el importe total. Se han implementado dos versiones: una procesa una imagen guardada y otra trabaja en tiempo real con la webcam.

## 1. Imagen fija

### 1.1. Carga y preprocesamiento

La imagen se carga con OpenCV y se convierte a escala de grises, ya que para detectar los contornos no necesitamos conservar el color. Después se aplica un desenfoque gaussiano para reducir el ruido.

```python
img = cv2.imread("imagenes/monedas/Monedas.jpg")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
suave = cv2.GaussianBlur(gris, (5, 5), 0)

bordes = cv2.Canny(suave, 50, 150)
kernel = np.ones((5, 5), np.uint8)
bordes = cv2.morphologyEx(
    bordes, cv2.MORPH_CLOSE, kernel, iterations=2
)
```

`Canny` detecta los bordes de la imagen usando los umbrales 50 y 150. Después, la operación morfológica `MORPH_CLOSE` cierra pequeños huecos en esos bordes, lo que ayuda a obtener contornos completos.

A continuación, se extraen los contornos externos:

```python
contornos, _ = cv2.findContours(
    bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

`RETR_EXTERNAL` evita trabajar con contornos internos y `CHAIN_APPROX_SIMPLE` reduce los puntos almacenados de cada contorno.

### 1.2. Filtrado de contornos

La detección de bordes puede encontrar objetos que no son monedas. Por eso, se filtran los contornos usando su área, perímetro y circularidad.

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

- **Área:** se descartan los contornos demasiado pequeños o grandes respecto al área total de la imagen.
- **Perímetro:** se descarta el contorno si su perímetro es cero, para evitar una división inválida.
- **Circularidad:** se calcula mediante \(C = 4\pi A/P^2\). Cuanto más se acerca a 1, más circular es el objeto. El umbral `0.65` elimina formas demasiado irregulares.

Para los contornos que superan los filtros, se estima el círculo que los contiene y se obtiene su diámetro en píxeles:

```python
centro, radio = cv2.minEnclosingCircle(c)
diametro_px = radio * 2
```

El programa guarda los datos de cada candidato —como su contorno, centro, diámetro y circularidad— para utilizarlos durante la clasificación.

### 1.3. Selección de la moneda de referencia y calibración

Los candidatos se muestran numerados para que el usuario pueda elegir una moneda cuya denominación conozca. El programa dispone de los diámetros oficiales de las ocho monedas de euro, expresados en milímetros.

Una vez elegida la referencia, calcula la relación entre píxeles y milímetros:

```python
mm_por_pixel = diametros_euro[valor_ref] / diametro_ref
diametro_mm = diametro_px * mm_por_pixel
```

`diametros_euro[valor_ref]` es el diámetro real de la moneda seleccionada y `diametro_ref` es el diámetro detectado en píxeles. La escala obtenida se aplica al resto de candidatos para estimar sus diámetros reales.

Esta calibración es importante: sin ella, el programa solo conocería tamaños en píxeles, que dependen de la resolución y de la distancia a la cámara.

### 1.4. Identificación del color

En la imagen fija, además del diámetro, se utiliza el color para limitar las denominaciones posibles. La función `detectar_color()` crea una máscara con el contorno de la moneda y analiza únicamente los píxeles de su interior.

La imagen se convierte al espacio HSV, que representa el tono, la saturación y el brillo. Se descartan píxeles demasiado oscuros o demasiado claros para reducir la influencia de sombras y reflejos.

La función calcula la proporción de píxeles cobrizos, dorados y neutros. Con esas proporciones decide si la moneda es cobriza, dorada, bimetálica o desconocida. Por ejemplo, las condiciones principales son:

```python
if prop_cobrizo > 0.30 and prop_cobrizo > prop_dorado:
    return "cobriza"
elif prop_dorado > 0.40 and prop_neutro < 0.40:
    return "dorada"
elif prop_dorado > 0.15 and prop_neutro > 0.30:
    return "bimetalica"
else:
    return "desconocida"
```

Los umbrales son heurísticos: ayudan a distinguir familias de monedas, pero la iluminación y los reflejos pueden alterar el color detectado.

Las familias que se consideran son:
- **Cobrizas:** 1, 2 y 5 céntimos.
- **Doradas:** 10, 20 y 50 céntimos.
- **Bimetálicas:** 1 € y 2 €.

### 1.5. Clasificación por diámetro y cálculo del total

Una vez conocida la familia de color, el programa compara el diámetro medido con los diámetros oficiales de las monedas de esa familia. Se queda con la denominación que tenga el menor error absoluto.

```python
error = abs(diametro_mm - diametro_oficial)
```

La clasificación solo se acepta si la diferencia entre el diámetro medido y el oficial no supera la tolerancia de `1.5 mm`. Si no hay una coincidencia suficientemente cercana, el candidato se marca como dudoso y no se suma al total.

Cuando una moneda se clasifica correctamente, el programa añade su valor al importe acumulado y actualiza el contador de monedas clasificadas. Finalmente, dibuja los contornos y etiquetas sobre la imagen para distinguir las monedas aceptadas de las dudosas, y muestra el resultado.

### 1.6. Resultado sobre la imagen fija

![Clasificación de monedas en imagen fija](resultados/deteccion_monedas.jpg)

**RESULTADO FINAL**
- Candidatos detectados: 8
- Monedas clasificadas: 8
- Monedas dudosas: 0
- Valor total estimado: 3.88 €

## 2. Detección en tiempo real

### 2.1. Captura y procesamiento de cada fotograma

La webcam se abre con `cv2.VideoCapture(0)`. El programa ejecuta un bucle que lee fotogramas continuamente y aplica un procesamiento similar al de la imagen fija.

```python
vid = cv2.VideoCapture(0)

while True:
    disponible, img = vid.read()
    if not disponible:
        break

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 25, 120)
    bordes = cv2.morphologyEx(
        bordes, cv2.MORPH_CLOSE, kernel, iterations=2
    )
```

En este caso, Canny utiliza los umbrales 25 y 120. Se vuelven a extraer los contornos y se filtran por área y circularidad; el umbral de circularidad es `0.70`, algo más exigente que en la imagen fija. También se descartan contornos con menos de cinco puntos.

Estos filtros se repiten en cada fotograma, por lo que la detección se actualiza a medida que se mueven la cámara o las monedas.

### 2.2. Calibración interactiva

Al comenzar, el programa todavía no conoce la escala. Muestra los candidatos numerados y espera a que el usuario pulse una tecla entre `1` y `9` para seleccionar uno. Después solicita la denominación de esa moneda y calcula `mm_por_pixel` usando su diámetro oficial.

Una vez calibrado, conserva la escala para procesar los siguientes fotogramas. La tecla `C` permite reiniciar la calibración si se cambia la posición de la cámara o se necesita seleccionar otra referencia.

### 2.3. Clasificación en cada fotograma

En tiempo real no se analiza el color. Para cada candidato, el diámetro en píxeles se convierte a milímetros y se compara con los ocho diámetros oficiales.

```python
diametro_mm = diametro_px * mm_por_pixel

error_minimo = float("inf")
for valor, diametro_oficial in diametros_euro.items():
    error = abs(diametro_mm - diametro_oficial)
    if error < error_minimo:
        error_minimo = error
        mejor_valor = valor

if error_minimo <= tolerancia:
    total += mejor_valor
    clasificadas += 1
```

El programa busca la denominación con menor diferencia y solo la acepta si el error es como máximo `1.5 mm`. A diferencia de la versión fija, aquí no se restringe la búsqueda por familia de color, por lo que monedas con diámetros parecidos pueden confundirse.

El total se calcula de nuevo en cada fotograma a partir de las monedas clasificadas que se ven en ese momento; por eso, puede variar si una moneda entra o sale del encuadre.

### 2.4. Controles y cierre

- **Teclas `1`–`9`:** elegir la moneda de referencia durante la calibración.
- **`C`:** reiniciar la calibración.
- **`ESC`:** terminar la ejecución.

Al salir del bucle, el programa libera la cámara y cierra las ventanas:

```python
vid.release()
cv2.destroyAllWindows()
```

Esto permite que la webcam quede disponible para otras aplicaciones.

### 2.5. Resultado en tiempo real

![Clasificación de monedas en tiempo real](resultados/monedas_webcam.gif)

## 3. Limitaciones

- La detección depende de la iluminación, las sombras, los reflejos y el contraste entre las monedas y el fondo.
- La calibración funciona mejor si todas las monedas están aproximadamente a la misma distancia y orientación que la moneda de referencia.
- En tiempo real, la clasificación se basa solo en el diámetro, por lo que puede confundir denominaciones con medidas cercanas.
- En el código de la webcam, `dudosas` se muestra sin inicializarse ni actualizarse. Además, `color` y `etiqueta` pueden quedar sin definir si un candidato no supera la tolerancia; conviene inicializarlas antes de dibujar cada candidato.

## TAREA: clasificación de microplásticos

Esta tarea consiste en desarrollar un clasificador capaz de distinguir entre tres tipos de partículas: fragmentos (FRA), pellets (PEL) y alquitrán (TAR).

La clasificación debe realizarse mediante características geométricas y/o visuales, sin utilizar modelos de aprendizaje profundo. Para evaluar el clasificador se proporciona una imagen de prueba con sus correspondientes anotaciones, que permitirá calcular distintas métricas y generar una matriz de confusión.

*Pendiente de documentar la implementación.*

### Resultado

*Pendiente de incorporar los resultados, las métricas de evaluación y la matriz de confusión.*

## Fuentes y herramientas utilizadas

- Repositorio completo de la práctica proporcionado por el profesor, usado como enunciado y referencia principal: https://github.com/otsedom/otsedom.github.io/tree/main/VC/P3
- ChatGPT, usado como apoyo para resumir la documentación sobre OpenCV, NumPy y Matplotlib durante el desarrollo de la práctica: https://chatgpt.com/
