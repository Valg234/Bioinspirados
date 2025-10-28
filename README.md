# Bioinspirados — Rutas y Resultados del TSP

Proyecto académico sobre el Problema del Viajante (TSP) utilizando algoritmos bioinspirados colonia de hormigas y algoritmos genéticos), con instancias, notebooks, gráficas y resultados reproducibles.

## Contenido del Repositorio

- `reto_final.ipynb`: Notebook principal con el flujo de trabajo del reto (carga de instancias, ejecución de heurísticas/metaheurísticas y comparación de resultados).
- `graficas_resultados.ipynb`: Notebook para generar y/o consolidar gráficas a partir de los resultados.
- `redes/`: Instancias base del TSP en formato CSV (nodos/ponderaciones). Los nombres siguen el patrón `X_N.csv` donde `N` es el número de nodos.
- `Instancias_Equipo4/`: Conjunto adicional de instancias del TSP (formato CSV) usadas en los experimentos.
- `Gams/`: Modelo del TSP en GAMS (p. ej., `Gams.gms`). Puede usarse junto con los archivos `.inc` generados en `resultados/` para resolver/validar instancias con el solver de GAMS.
- `resultados/`: Carpeta con resultados por instancia:
  - Subcarpetas `i_n/` con archivos `.json` (resultados numéricos), `.html` (reportes) y `.inc` (archivos GAMS auxiliares).
  - `geocoding_cache.json` (si se usó geocodificación para mapas/visualizaciones).
- `graficas/`: Imágenes PNG con comparativas de desempeño (p. ej., `tiempo_n100.png`, `distancia_n100.png`).
- `Tiempos para las instancias.csv`: Tabla consolidada de tiempos de ejecución por instancia/tamaño.
- `Reporte.pdf`: Reporte del proyecto con metodología, resultados y conclusiones.

> Nota: No hay scripts `.py` en la raíz; el trabajo se organiza principalmente en notebooks y datos/resultados precomputados.

## Cómo Usarlo

1. Clona el repositorio:
   
2. Crea un entorno virtual de Python 3.9+.
3. Abre los notebooks en Jupyter (o VS Code):
   - `pip install jupyterlab` y luego `jupyter lab`, o
   - Abre `reto_final.ipynb` y `graficas_resultados.ipynb` directamente en tu IDE.
4. Ejecuta las celdas en orden. Si el entorno solicita librerías faltantes, instálalas según el mensaje. Típicamente se usan: `pandas`, `numpy`, `matplotlib`/`seaborn` y, según el caso, utilidades para grafos o mapas.

Si deseas usar GAMS:
- Requiere tener instalado GAMS.
- Abre `Gams/Gams.gms` y, si corresponde, referencia un archivo `.inc` desde `resultados/X_N/gams_X_N.inc` para cargar parámetros/datos de una instancia concreta.

Los datos y resultados incluidos permiten revisar sin necesidad de recomputar todo. Para reproducir desde cero, asegúrate de tener las dependencias y ejecuta el notebook principal.

## Estructura de Nombres en Resultados

- Subcarpetas de `resultados/` siguen el patrón `X_N/`, donde:
  - `X` identifica una configuración/instancia específica.
  - `N` es el tamaño de la instancia (número de nodos).
- Dentro de cada subcarpeta encontrarás:
  - `X_N.json`: métricas clave (distancias, tiempos, parámetros, etc.).
  - `X_N_aco.html` / `X_N_genetico.html`: reportes por método.
  - `gams_X_N.inc`: archivo auxiliar para GAMS, si corresponde.

## Gráficas Incluidas

En `graficas/` hay comparativas como:
- `tiempo_n{40,100,150,200}.png`: tiempos por tamaño de instancia.
- `distancia_n{40,100,150,200}.png`: calidad de solución (distancia) por tamaño.

Estas imágenes se generan/actualizan desde `graficas_resultados.ipynb`.

## Reproducibilidad y Notas

- Este repositorio contiene tanto instancias como resultados ya generados para facilitar la revisión.
- Si cambias parámetros de los algoritmos en el notebook, puedes volver a generar y guardar resultados en `resultados/` siguiendo la misma convención de nombres.
- Los archivos `.html` permiten inspeccionar rápidamente salidas detalladas sin ejecutar código.


