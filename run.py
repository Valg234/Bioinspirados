#!/usr/bin/env python3
"""
Parallel TSP solver using Genetic Algorithm and Ant Colony Optimization.
Processes multiple network files in parallel for faster execution.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import folium
import random
import os
import json
import warnings
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
PATH_DE_REDES = 'redes'
PATH_DE_RESULTADOS = 'resultados'

# Ensure output directory exists
os.makedirs(PATH_DE_RESULTADOS, exist_ok=True)


# ============================================================================
# Distance calculation functions
# ============================================================================

def haversine_distance(coord1, coord2):
    """Calculate the great circle distance between two points on earth (in km)"""
    R = 6371
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat/2)**2 + cos(lat1_rad)*cos(lat2_rad)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def crear_matriz_distancias(df_red):
    """Create distance matrix from coordinates"""
    coordenadas = df_red[['lat', 'lon']].values
    n = len(coordenadas)
    matriz_distancias = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            distancia = haversine_distance(coordenadas[i], coordenadas[j])
            matriz_distancias[i, j] = distancia
            matriz_distancias[j, i] = distancia

    return matriz_distancias


# ============================================================================
# Genetic Algorithm implementation
# ============================================================================

def inicializar_poblacion(tamano_poblacion, indices_ciudades):
    """Initialize population with random permutations"""
    poblacion = []
    for _ in range(tamano_poblacion):
        individuo = indices_ciudades.copy()
        random.shuffle(individuo)
        poblacion.append(individuo)
    return poblacion


def calcular_aptitud(individuo, matriz_distancias):
    """Calculate total distance for a route"""
    distancia_total = 0
    for i in range(len(individuo)):
        ciudad_origen = individuo[i]
        ciudad_destino = individuo[(i + 1) % len(individuo)]
        distancia_total += matriz_distancias[ciudad_origen][ciudad_destino]
    return distancia_total


def seleccionar_padres(poblacion, aptitudes, num_padres):
    """Select the best parents for reproduction"""
    padres_indices = np.argsort(aptitudes)[:num_padres]
    padres = [poblacion[i] for i in padres_indices]
    return padres


def cruzar_padres(padre1, padre2):
    """Crossover operation using order crossover (OX)"""
    tamano = len(padre1)
    inicio, fin = sorted(random.sample(range(tamano), 2))
    hijo = [None]*tamano
    hijo[inicio:fin+1] = padre1[inicio:fin+1]

    pointer = 0
    for i in range(tamano):
        if hijo[i] is None:
            while padre2[pointer] in hijo:
                pointer += 1
            hijo[i] = padre2[pointer]
            pointer += 1
    return hijo


def mutar_individuo(individuo, tasa_mutacion):
    """Mutation operation using swap mutation"""
    for i in range(len(individuo)):
        if random.random() < tasa_mutacion:
            j = random.randint(0, len(individuo)-1)
            individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo


def algoritmo_genetico_tsp(matriz_distancias, tamano_poblacion=100, num_generaciones=500, tasa_mutacion=0.01):
    """Genetic algorithm for TSP"""
    indices_ciudades = list(range(len(matriz_distancias)))
    poblacion = inicializar_poblacion(tamano_poblacion, indices_ciudades)
    mejor_distancia = float('inf')
    mejor_ruta = None

    for generacion in range(num_generaciones):
        aptitudes = [calcular_aptitud(individuo, matriz_distancias) for individuo in poblacion]
        distancia_actual = min(aptitudes)
        if distancia_actual < mejor_distancia:
            mejor_distancia = distancia_actual
            mejor_ruta = poblacion[np.argmin(aptitudes)]

        num_padres = tamano_poblacion // 2
        padres = seleccionar_padres(poblacion, aptitudes, num_padres)

        siguiente_generacion = []
        while len(siguiente_generacion) < tamano_poblacion:
            padre1, padre2 = random.sample(padres, 2)
            hijo = cruzar_padres(padre1, padre2)
            hijo = mutar_individuo(hijo, tasa_mutacion)
            siguiente_generacion.append(hijo)

        poblacion = siguiente_generacion

    return mejor_ruta, mejor_distancia


# ============================================================================
# Ant Colony Optimization implementation
# ============================================================================

def calcular_heuristica(costos):
    """Calculate heuristic information (inverse of distance)"""
    num_nodos = costos.shape[0]
    heuristica = np.zeros_like(costos)
    for i in range(num_nodos):
        for j in range(num_nodos):
            if costos[i, j] > 0 and not np.isinf(costos[i, j]):
                heuristica[i, j] = 1 / costos[i, j]
            else:
                heuristica[i, j] = 0
    return heuristica


def calcular_probabilidades(feromonas, heuristica, nodo_actual, visitados, alpha, beta):
    """Calculate transition probabilities for ACO"""
    num_nodos = len(feromonas)
    probabilidades = np.zeros(num_nodos)
    for j in range(num_nodos):
        if j not in visitados:
            probabilidades[j] = (feromonas[nodo_actual, j] ** alpha) * (heuristica[nodo_actual, j] ** beta)
    suma_probabilidades = np.sum(probabilidades)
    if suma_probabilidades == 0:
        return probabilidades
    return probabilidades / suma_probabilidades


def construir_camino(feromonas, heuristica, costos, alpha, beta):
    """Construct a path for one ant"""
    num_nodos = costos.shape[0]
    nodo_actual = random.randint(0, num_nodos - 1)
    camino = [nodo_actual]
    costo_total = 0
    visitados = set(camino)

    while len(camino) < num_nodos:
        probabilidades = calcular_probabilidades(feromonas, heuristica, nodo_actual, visitados, alpha, beta)
        if np.sum(probabilidades) == 0:
            nodos_no_visitados = list(set(range(num_nodos)) - visitados)
            siguiente_nodo = random.choice(nodos_no_visitados)
        else:
            siguiente_nodo = random.choices(range(num_nodos), weights=probabilidades, k=1)[0]
        costo_total += costos[nodo_actual, siguiente_nodo]
        camino.append(siguiente_nodo)
        visitados.add(siguiente_nodo)
        nodo_actual = siguiente_nodo

    costo_total += costos[camino[-1], camino[0]]
    camino.append(camino[0])

    return camino, costo_total


def actualizar_feromonas(feromonas, soluciones, rho):
    """Update pheromone levels"""
    feromonas *= (1 - rho)
    for camino, costo in soluciones:
        for i in range(len(camino) - 1):
            feromonas[camino[i], camino[i + 1]] += 1 / costo


def ACO(costos, num_hormigas=500, num_iteraciones=100, alpha=1.0, beta=2.0, rho=0.5):
    """Ant Colony Optimization algorithm for TSP"""
    num_nodos = costos.shape[0]
    mejor_camino = None
    mejor_costo = float('inf')

    feromonas = np.ones((num_nodos, num_nodos)) * 0.1
    heuristica = calcular_heuristica(costos)

    for iteracion in range(num_iteraciones):
        soluciones = []
        for _ in range(num_hormigas):
            camino, costo = construir_camino(feromonas, heuristica, costos, alpha, beta)
            soluciones.append((camino, costo))
            if costo < mejor_costo:
                mejor_camino = camino
                mejor_costo = costo

        actualizar_feromonas(feromonas, soluciones, rho)

    return mejor_camino, mejor_costo


# ============================================================================
# Visualization and output functions
# ============================================================================

def crear_visualizacion(df_red, mejor_ruta, nombre_archivo):
    """Create Folium map visualization of the route"""
    coordenadas = df_red[['lat', 'lon']].values
    ruta_coordenadas = coordenadas[mejor_ruta]
    ruta_coordenadas = np.vstack([ruta_coordenadas, ruta_coordenadas[0]])

    centro_lat = np.mean(coordenadas[:, 0])
    centro_lon = np.mean(coordenadas[:, 1])
    mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=12)

    for idx, coord in enumerate(ruta_coordenadas[:-1]):
        folium.Marker(location=coord, popup=f"Parada {idx + 1}").add_to(mapa)

    folium.PolyLine(ruta_coordenadas, color="blue", weight=2.5, opacity=1).add_to(mapa)

    return mapa.save(nombre_archivo)


def generar_archivos_inclusion(archivo, df_red, matriz_distancias):
    """Generate GAMS inclusion file with distance matrix"""
    n = len(df_red)

    nombre_base = os.path.basename(archivo).replace('.csv', '')
    path_resultado_dir = os.path.join(PATH_DE_RESULTADOS, nombre_base)
    nombre_archivo_inc = os.path.join(path_resultado_dir, f"gams_{nombre_base}.inc")

    with open(nombre_archivo_inc, 'w') as f:
        # Write header (column indices)
        f.write("    ")
        for j in range(n):
            f.write(f"{j:<8}")
        f.write("\n")

        # Write matrix rows
        for i in range(n):
            f.write(f"{i:<4}")
            for j in range(n):
                f.write(f"{matriz_distancias[i][j]:<8.1f}")
            f.write("\n")


# ============================================================================
# Main processing function
# ============================================================================

def procesar_archivo(archivo):
    """Process a single network file with both GA and ACO"""
    try:
        # Read network data
        df_red = pd.read_csv(archivo)
        nombre_base = os.path.basename(archivo).replace('.csv', '')

        # Create output directory
        path_resultado_dir = os.path.join(PATH_DE_RESULTADOS, nombre_base)
        os.makedirs(path_resultado_dir, exist_ok=True)

        # Define output paths
        path_json = os.path.join(path_resultado_dir, f"{nombre_base}.json")
        path_vis_genetico = os.path.join(path_resultado_dir, f"{nombre_base}_genetico.html")
        path_vis_aco = os.path.join(path_resultado_dir, f"{nombre_base}_aco.html")

        # Create distance matrix
        matriz_distancias = crear_matriz_distancias(df_red)

        # Run Genetic Algorithm
        start_genetico = time.time()
        mejor_ruta_genetico, mejor_distancia_genetico = algoritmo_genetico_tsp(
            matriz_distancias,
            tamano_poblacion=100,
            num_generaciones=500,
            tasa_mutacion=0.01
        )
        tiempo_genetico = time.time() - start_genetico

        # Run Ant Colony Optimization
        start_aco = time.time()
        mejor_ruta_aco, mejor_distancia_aco = ACO(
            matriz_distancias,
            num_hormigas=500,
            num_iteraciones=100,
            alpha=1.0,
            beta=2.0,
            rho=0.5
        )
        tiempo_aco = time.time() - start_aco

        # Save results to JSON
        resultados = {
            "genetico": {
                "ruta": mejor_ruta_genetico,
                "distancia": mejor_distancia_genetico,
                "tiempo": tiempo_genetico
            },
            "aco": {
                "ruta": mejor_ruta_aco,
                "distancia": mejor_distancia_aco,
                "tiempo": tiempo_aco
            }
        }

        with open(path_json, "w") as f:
            json.dump(resultados, f, indent=2)

        # Create visualizations
        crear_visualizacion(df_red, mejor_ruta_genetico, path_vis_genetico)
        crear_visualizacion(df_red, mejor_ruta_aco, path_vis_aco)

        # Generate GAMS inclusion file
        generar_archivos_inclusion(archivo, df_red, matriz_distancias)

        return {
            'archivo': nombre_base,
            'success': True,
            'genetico_distancia': mejor_distancia_genetico,
            'genetico_tiempo': tiempo_genetico,
            'aco_distancia': mejor_distancia_aco,
            'aco_tiempo': tiempo_aco
        }

    except Exception as e:
        return {
            'archivo': os.path.basename(archivo),
            'success': False,
            'error': str(e)
        }


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Main execution function with parallel processing"""
    print("=" * 80)
    print("TSP Solver - Parallel Execution")
    print("=" * 80)
    print(f"\nProcessing networks from: {PATH_DE_REDES}")
    print(f"Output directory: {PATH_DE_RESULTADOS}\n")

    # Get all CSV files from redes directory
    archivos = [
        os.path.join(PATH_DE_REDES, f)
        for f in os.listdir(PATH_DE_REDES)
        if f.endswith('.csv')
    ]

    # Sort files by size first (40, 100, 150, 200), then by instance number
    # This processes smaller models first for faster initial results
    def get_sort_key(filepath):
        basename = os.path.basename(filepath)
        parts = basename.replace('.csv', '').split('_')
        instance = int(parts[0])
        size = int(parts[1])
        return (size, instance)  # Sort by size first, then instance

    archivos_sorted = sorted(archivos, key=get_sort_key)

    if not archivos_sorted:
        print(f"No CSV files found in {PATH_DE_REDES}")
        return

    print(f"Found {len(archivos_sorted)} files to process")

    # Determine number of processes
    num_processes = max(1, cpu_count() - 1)  # Leave one core free
    print(f"Using {num_processes} parallel processes\n")

    # Process files in parallel
    start_time = time.time()

    # Use imap_unordered for better progress bar performance
    resultados = []
    with Pool(processes=num_processes) as pool:
        # Submit all tasks
        async_results = [pool.apply_async(procesar_archivo, (archivo,)) for archivo in archivos_sorted]

        # Monitor progress
        with tqdm(total=len(archivos_sorted), desc="Processing files", unit="file") as pbar:
            for async_result in async_results:
                resultado = async_result.get()  # Wait for result
                resultados.append(resultado)
                pbar.update(1)

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    exitosos = [r for r in resultados if r['success']]
    fallidos = [r for r in resultados if not r['success']]

    print(f"\nTotal files processed: {len(resultados)}")
    print(f"Successful: {len(exitosos)}")
    print(f"Failed: {len(fallidos)}")
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per file: {total_time/len(resultados):.2f} seconds")

    if exitosos:
        print("\n" + "-" * 80)
        print("Results Summary:")
        print("-" * 80)
        print(f"{'File':<15} {'GA Dist (km)':<15} {'GA Time (s)':<15} {'ACO Dist (km)':<15} {'ACO Time (s)':<15}")
        print("-" * 80)

        for r in exitosos:
            print(f"{r['archivo']:<15} {r['genetico_distancia']:<15.2f} {r['genetico_tiempo']:<15.2f} "
                  f"{r['aco_distancia']:<15.2f} {r['aco_tiempo']:<15.2f}")

    if fallidos:
        print("\n" + "-" * 80)
        print("Failed files:")
        print("-" * 80)
        for r in fallidos:
            print(f"  - {r['archivo']}: {r['error']}")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
