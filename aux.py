"""
Funciones auxiliares compartidas por los modelos de segmentación.

Incluye los canales y grupos de features, la lectura del dataset
preprocesado, la limpieza de la máscara predicha, la evaluación y la gestión
de los CSV.
"""

import json
import os

import cv2 as cv
import numpy as np
import pandas as pd

from metricas import calcular_metricas


# ===========================================================================
# 1. CANALES
# ===========================================================================
# El preprocesado guarda una pila (alto, ancho, 11) uint8 por imagen

CANALES = ('l', 'a', 'b', 'y', 'cr', 'cb', 'r', 'h', 's', 'exg', 'blob')

DESCRIPCION = {
    'l': 'Lab L (0-255)',
    'a': 'Lab a (0-255, 128 = neutro)',
    'b': 'Lab b (0-255, alto = amarillo)',
    'y': 'YCrCb Y (0-255)',
    'cr': 'YCrCb Cr (0-255)',
    'cb': 'YCrCb Cb (0-255)',
    'r': 'RGB R (0-255)',
    'h': 'HSV H reescalado de 0-179 a 0-255',
    's': 'HSV S (0-255)',
    'exg': 'ExG = 2G - R - B recortado a 0-255',
    'blob': 'DoG multiescala sobre b*, normalizado con el p99 de dev',
}

# Grupos candidatos para los modelos
GRUPOS = {
    'lab_ab': ('a', 'b'),
    'crcb': ('cr', 'cb'),
    'ab_s': ('a', 'b', 's'),
    'ab_cr_s': ('a', 'b', 'cr', 's'),
    'lab': ('l', 'a', 'b'),
    'lab_blob': ('l', 'a', 'b', 'blob'),
    'cr_blob': ('cr', 'blob'),
}


def columnas(grupo, canales=CANALES):
    """Índices de los canales del grupo dentro de la pila."""
    try:
        nombres = GRUPOS[grupo]
    except KeyError:
        raise KeyError(f'Grupo desconocido: {grupo}. Válidos: {list(GRUPOS)}') from None
    return [canales.index(c) for c in nombres]


def seleccionar(pila, grupo):
    """Selecciona un grupo de canales de una imagen o matriz de píxeles."""
    return pila[..., columnas(grupo)]


def canal(pila, nombre, canales=CANALES):
    """Selecciona un único canal de la pila por su nombre."""
    return pila[..., canales.index(nombre)]


def indices_canales(espacios, canales=CANALES):
    """Convierte ``nombre -> tupla de canales`` en índices."""
    return {e: [canales.index(c) for c in cs] for e, cs in espacios.items()}


# ===========================================================================
# 2. AJUSTES COMPARTIDOS
# ===========================================================================
RUTA_MASCARAS = './dataset_manzanas/mascaras'

# Test externo: solo se usa al final, nunca para ajustar ni calibrar.
RUTA_TEST = './manzanas_test_set/imagenes'
RUTA_TEST_MASCARAS = './manzanas_test_set/mascaras'
RUTA_TEST_GRID = './dataset_preprocesado_test'

SEMILLA = 42
PASO_MUESTREO = 50
MAX_MUESTRAS = 200_000
N_HILOS_CV = 4

FRAC_AJUSTE = 0.8       # del split dev: parte para ajustar, resto validación
K_FOLDS = 5             # particiones del k-fold sobre el split dev

ASPECTO_MIN = 0.3       # min(ancho, alto) / max(ancho, alto) de cada región
AREA_MIN_BASE = 100     # área mínima a escala 1.0 (se escala con escala**2)

CLAVES = ('accuracy', 'precision', 'recall',
          'fpr', 'specificity', 'iou', 'f1')


# ===========================================================================
# 3. LECTURA DEL DATASET
# ===========================================================================
def leer_nombres(ruta):
    """Lee ``nombres.txt`` de un dataset."""
    with open(os.path.join(ruta, 'nombres.txt')) as f:
        return [n.strip() for n in f if n.strip()]

def leer_mascara(ruta):
    """Máscara binaria 0/255, robusta a máscaras guardadas como 0/1."""
    m = cv.imread(ruta, cv.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(ruta)
    return np.where(m > 0, 255, 0).astype(np.uint8)

def leer_npy(ruta):
    """Lee un ``.npy`` y comprueba canales."""
    datos = np.load(ruta)
    if datos.ndim != 3 or datos.shape[2] != len(CANALES):
        raise ValueError(
            f'{ruta}: se esperaba (alto, ancho, {len(CANALES)}), '
            f'recibido {datos.shape}')
    return datos

def leer_config(ruta):
    """Lee ``config.json`` (vacío si no existe)."""
    ruta_cfg = os.path.join(ruta, 'config.json')
    if not os.path.exists(ruta_cfg):
        return {}
    with open(ruta_cfg) as f:
        return json.load(f)

def leer_datos(ruta, split):
    """Lee nombres e índices del split (``dev`` o ``test``)."""
    nombres = leer_nombres(ruta)
    indices = np.load(os.path.join(ruta, 'split.npz'))[split]
    return nombres, indices

def leer_dev_completo(ruta):
    """Todos los nombres del grid como conjunto de desarrollo (sin split)."""
    nombres = leer_nombres(ruta)
    return nombres, np.arange(len(nombres))

def ruta_test(ruta):
    """Carpeta del grid de test correspondiente a una configuración de dev."""
    return os.path.join(RUTA_TEST_GRID, os.path.basename(ruta))

def leer_test(ruta):
    """Nombres e índices (todos) del grid de test de una configuración."""
    nombres = leer_nombres(ruta)
    return nombres, np.arange(len(nombres))

def repartir_indices(indices, frac_ajuste, semilla=SEMILLA):
    """Divide los índices de desarrollo en ajuste y validación (siempre igual)."""
    if len(indices) < 2:
        raise ValueError('Se necesitan al menos 2 imágenes en el split.')

    rng = np.random.default_rng(semilla)
    permutacion = rng.permutation(indices)

    n_ajuste = int(len(permutacion) * frac_ajuste)
    n_ajuste = min(max(n_ajuste, 1), len(permutacion) - 1)

    return np.sort(permutacion[:n_ajuste]), np.sort(permutacion[n_ajuste:])


def folds_kfold(indices, k=K_FOLDS, frac_ajuste=FRAC_AJUSTE, semilla=SEMILLA):
    """Divide ``indices`` en ``k`` folds con validación ~ (1 - frac_ajuste)."""
    n = len(indices)
    if n < k:
        raise ValueError(f'Se necesitan al menos {k} imágenes para {k}-fold.')

    perm = np.random.default_rng(semilla).permutation(indices)
    n_val = int(round(n * (1.0 - frac_ajuste)))
    n_val = min(max(n_val, 1), n - k + 1)

    tamanos = [n_val] * k
    resto = n - n_val * k
    for i in range(abs(resto)):
        j = i % k if resto > 0 else k - 1 - (i % k)
        tamanos[j] += 1 if resto > 0 else -1

    if min(tamanos) < 1:
        return [np.sort(chunk) for chunk in np.array_split(perm, k)]

    folds, inicio = [], 0
    for tam in tamanos:
        folds.append(np.sort(perm[inicio:inicio + tam]))
        inicio += tam
    return folds

def cargar_datos(nombres, idx_ajuste, idx_validacion, ruta,
                 ruta_mascaras=RUTA_MASCARAS, paso_muestreo=PASO_MUESTREO,
                 max_muestras=MAX_MUESTRAS, semilla=SEMILLA):
    """Carga los ``.npy`` y devuelve lo necesario para entrenar y validar.

    - Ajuste: solo se guarda una muestra de píxeles y su etiqueta.
    - Validación: se guardan las pilas de canales completas y su referencia.

    Las máscaras se reescalan a la resolución del ``.npy`` si hace falta, así
    que el modelo no necesita saber si el preprocesado usó escala 1.0 o 0.5.
    """
    muestras = []
    etiquetas = []
    for i in idx_ajuste:
        datos = leer_npy(os.path.join(ruta, nombres[i] + '.npy'))
        ref = leer_mascara(os.path.join(ruta_mascaras, nombres[i] + '.png'))

        if datos.shape[:2] != ref.shape:
            ref = cv.resize(ref, (datos.shape[1], datos.shape[0]),
                            interpolation=cv.INTER_NEAREST)

        pix = datos.reshape(-1, len(CANALES))
        muestras.append(pix[::paso_muestreo])
        etiquetas.append(ref.ravel()[::paso_muestreo] == 255)

    muestra = np.concatenate(muestras)
    es_manzana = np.concatenate(etiquetas)

    if len(muestra) > max_muestras:
        rng = np.random.default_rng(semilla)
        sel = rng.choice(len(muestra), max_muestras, replace=False)
        muestra, es_manzana = muestra[sel], es_manzana[sel]

    validacion = []
    for i in idx_validacion:
        datos = leer_npy(os.path.join(ruta, nombres[i] + '.npy'))
        ref = leer_mascara(os.path.join(ruta_mascaras, nombres[i] + '.png'))
        if datos.shape[:2] != ref.shape:
            ref = cv.resize(ref, (datos.shape[1], datos.shape[0]),
                            interpolation=cv.INTER_NEAREST)
        validacion.append((datos, ref))

    return muestra, es_manzana, validacion


def cargar_pilas(nombres, indices, ruta, ruta_mascaras=RUTA_MASCARAS):
    """Lista de ``(pila, máscara)`` para las imágenes indicadas."""
    datos = []
    for i in indices:
        pila = leer_npy(os.path.join(ruta, nombres[i] + '.npy'))
        ref = leer_mascara(os.path.join(ruta_mascaras, nombres[i] + '.png'))
        if pila.shape[:2] != ref.shape:
            ref = cv.resize(ref, (pila.shape[1], pila.shape[0]),
                            interpolation=cv.INTER_NEAREST)
        datos.append((pila, ref))
    return datos

# ===========================================================================
# 4. LIMPIEZA DE LA MÁSCARA
# ===========================================================================
def limpiar(m, escala, aspecto_min=ASPECTO_MIN, area_min_base=AREA_MIN_BASE):
    """Apertura, cierre y filtrado de regiones por área y forma."""
    k_open, k_close = (3, 5) if escala < 1 else (5, 9)
    m = cv.morphologyEx(m, cv.MORPH_OPEN,
                        cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                 (k_open, k_open)))
    m = cv.morphologyEx(m, cv.MORPH_CLOSE,
                        cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                 (k_close, k_close)))

    n, etiquetas, stats, _ = cv.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)

    areas = stats[:, cv.CC_STAT_AREA]
    anchos = stats[:, cv.CC_STAT_WIDTH]
    altos = stats[:, cv.CC_STAT_HEIGHT]
    aspecto = np.minimum(anchos, altos) / np.maximum(np.maximum(anchos, altos), 1)

    area_min = max(1, int(area_min_base * escala ** 2))
    validos = (areas >= area_min) & (aspecto >= aspecto_min)
    validos[0] = False  # la etiqueta 0 es el fondo

    return np.where(validos[etiquetas], 255, 0).astype(np.uint8)


def binarizar_y_limpiar(puntuacion, umbral, escala,
                        aspecto_min=ASPECTO_MIN, area_min_base=AREA_MIN_BASE):
    """Umbraliza un mapa de puntuación (p. ej. p(manzana)) y limpia la máscara."""
    m = np.where(puntuacion >= umbral, 255, 0).astype(np.uint8)
    return limpiar(m, escala, aspecto_min, area_min_base)

# ===========================================================================
# 5. EVALUACIÓN
# ===========================================================================
def evaluar(validacion, predecir):
    """Métricas de cada imagen de validación.

    ``predecir(img)`` debe devolver la máscara predicha para esa imagen.
    """
    return [calcular_metricas(ref, predecir(img)) for img, ref in validacion]


def medias_metricas(metricas, claves=CLAVES):
    """Media de cada métrica de ``claves`` sobre todas las imágenes."""
    return {c: float(np.mean([m[c] for m in metricas])) for c in claves}

def resumen_metricas(metricas_por_fold, claves=CLAVES):
    """Media y desviación típica entre folds de cada métrica.

    ``metricas_por_fold`` contiene un diccionario de métricas por fold (cada
    uno ya promediado sobre sus imágenes de validación). Devuelve la media
    entre folds en ``<métrica>`` y su desviación en ``<métrica>_std``.
    """
    resumen = {}
    for c in claves:
        valores = [float(m[c]) for m in metricas_por_fold]
        resumen[c] = float(np.mean(valores))
        resumen[c + '_std'] = float(np.std(valores))
    return resumen

# ===========================================================================
# 6. CONFIGURACIONES Y CSV
# ===========================================================================
def info_config(cfg):
    """Columnas descriptivas de la configuración."""
    return {
        'config': cfg.get('id', 'unificado'),
        'escala': float(cfg.get('escala', 1.0)),
    }

def datos_config(ruta):
    """``id`` y ``escala`` leídos de ``config.json`` (formato del preprocesado)."""
    cfg = leer_config(ruta)
    prep = cfg.get('preprocesado', {})
    return {'id': prep.get('id', os.path.basename(ruta)),
            'escala': float(prep.get('escala', 1.0))}

def configs_hechas(csv_pruebas, reanudar=True):
    """Configs ya presentes en ``csv_pruebas``."""
    if not os.path.exists(csv_pruebas):
        return set()
    if not reanudar:
        os.remove(csv_pruebas)
        return set()
    return set(pd.read_csv(csv_pruebas, usecols=['config'])['config'])

def anadir_pruebas(df, csv_pruebas):
    """Añade los trials al CSV, escribiendo cabecera solo la primera vez."""
    df.to_csv(csv_pruebas, mode='a', index=False,
              header=not os.path.exists(csv_pruebas))

def mejores_por_config(df, metrica):
    """Mejor fila (según ``metrica``) de cada configuración, ordenadas."""
    idx = df.groupby('config')[metrica].idxmax()
    return (df.loc[idx]
              .sort_values(metrica, ascending=False)
              .reset_index(drop=True))

def rutas_configs(ruta):
    """Subcarpetas del grid que contienen ``config.json``."""
    return sorted(
        os.path.join(ruta, d) for d in os.listdir(ruta)
        if os.path.isfile(os.path.join(ruta, d, 'config.json')))
