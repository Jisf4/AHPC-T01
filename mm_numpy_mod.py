import numpy as np
from joblib import Parallel, delayed
import time
import pandas as pd

# Lista de dimesiones para las matrices
t = [256, 512, 1024, 2048, 4096]

# Diccionarios para almacenar los resultados
time_r = {"256": {}, "512": {}, "1024": {}, "2048": {}, "4096": {}}
performance_r = {"256": {},"512": {}, "1024": {}, "2048": {}, "4096": {}}
E_r = {"256": {},"512": {}, "1024": {}, "2048": {}, "4096": {}}

# Función para la realización del producto punto de dos matrices
def compute_row(i):
    return np.dot(A[i, :], B)

# Cálculo con matrices de diferentes dimensiones
for N in t:
  A = np.ones((N, N))
  B = np.ones((N, N))
  C = np.zeros((N, N))
  # Uso de diferente cantida de hilos del procesador
  for j in range(12):
    start = time.time()
    # Paralelizaje del proceso con joblib
    C_rows = Parallel(n_jobs=j+1)(delayed(compute_row)(i) for i in range(N))
    C = np.vstack(C_rows)
    end = time.time()
    # Cálculo de resultados y almacenamiento
    elapsed = end - start
    gflops = (2 * N**3) / 1e9 / elapsed
    E = gflops / (j+1)
    time_r[str(N)][str(j+1)] = elapsed
    performance_r[str(N)][str(j+1)] = gflops
    E_r[str(N)][str(j+1)] = E
# Conversión de diccionarios a dataframes para exportar resultados
df_time = pd.DataFrame(time_r)
df_performance = pd.DataFrame(performance_r)
df_E = pd.DataFrame(E_r)
# Exportación de resultados en formato xlsx
with pd.ExcelWriter('Resultados.xlsx') as writer:
    df_time.to_excel(writer, sheet_name='Sheet1', index=False)
    df_performance.to_excel(writer, sheet_name='Sheet2', index=False)
    df_E.to_excel(writer, sheet_name='Sheet3', index=False)