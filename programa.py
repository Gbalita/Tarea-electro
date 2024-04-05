# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:57:39 2024

@author: g.balaguera
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Datos que se le piden a los usuarios para el progra,a
N = int(input("Escribe el límite del eje x: "))  # Convertir a float para permitir valores decimales
M = int(input("Escribe el límite del eje y: "))  # Convertir a float para permitir valores decimales
dx = int(input("Escribe la rata de cambio de los ejes dx=dy: "))
V0_X_0 = int(input("Escribe el valor del límite de frontera cuando x=0: "))
V0_X_a = int(input("Escribe el valor del límite de frontera cuando x=a: "))
V0_Y_0 = int(input("Escribe el valor del límite de frontera cuando y=0: "))
V0_Y_b = int(input("Escribe el valor del límite de frontera cuando y=b: "))



# encontramos el valor de punto a partir de los datos dados
a= int(N*dx)
b= int(M*dx)

#igualamos dx a dy ya que en este caso son iguales
dy=dx


# creamos la matriz con las condiciones de frontera 
def Condiciones_frontera(N, M):
    V = np.zeros((N, M))
    V[:, 0] = V0_Y_0
    V[:, -1] = V0_Y_b
    V[0, :] = V0_X_0
    V[-1,:] =V0_X_a
    return V

#aplicamos el metodo de relajación de Laplace
def relajación_laplace(V, dx, dy, tolerance=1e-4, max_iterations=10000):
    iterations = 0
    while iterations < max_iterations:
        max_residual = 0.0
        for i in range(1, N-1):
            for j in range(1, M-1):
                # Calcula el nuevo valor del potencial usando el método de Gauss-Seidel
                new_V = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
                residual = np.abs(new_V - V[i, j])
                if residual > max_residual:
                    max_residual = residual
                V[i, j] = new_V
        iterations += 1
        if max_residual < tolerance:
            break
    return V

#gráficamos el resultado en 3D
def plot_solución_3d(V, dx, dy):
    x = np.linspace(0, a, N)
    y = np.linspace(0, b, M)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V.T)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

#ponemos la solución analítica encontrada en los puntos anteriores
def solución_analítica(x, y, n_max=1):
    V = 0
    V1 = 0
    for n in range(1, n_max + 1):
        #asumiendo que n es par podemos reducir la expresión encontrada
        A=(4*V0_Y_b/(np.pi*n)**2)*(1-(-1)**n)
        V += A*np.sin(n*np.pi*x/a)*((np.sinh(n*np.pi*y/a)*(1-np.cosh(n*np.pi*b/a))/np.sinh(n*np.pi*b/a))-np.cosh(n*np.pi*y/a))
    V1 = np.flip(V,axis=0)
    return (V1-V)

#graficamos la diferencia entre las soluciones encontradas
def plot_difference_scatter_3d(V_numerical, V_analytical, a, b):
    x = np.linspace(0, a, V_numerical.shape[0])
    y = np.linspace(0, b, V_numerical.shape[1])
    X, Y = np.meshgrid(x, y)
    diff = V_numerical - V_analytical
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y, X, diff, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Diferencia')
    ax.set_title('Diferencia entre Soluciones Numérica y Analítica en 3D')

# datos para la solución analítica
x = np.linspace(0, a, N)
y = np.linspace(0, b, M)
X, Y = np.meshgrid(x, y)

# calculamos la solución analítica
V_analitica = solución_analítica(X, Y)
# agregamos las condiciones de frontera
V = Condiciones_frontera(N, M)

# solución de la ecuación con el metodo de relajación
V = relajación_laplace(V, dx, dy)

# graficamos la solución
fig = plt.figure()
x = np.linspace(0, a, N)
y = np.linspace(0, b, M)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, V_analitica, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución analítica de V(x,y)')
plt.savefig("Solución analítica de V(x,y)")
plt.show()

# graficamos la solución de la ecuación
fig = plt.figure()
x = np.linspace(0, a, N)
y = np.linspace(0, b, M)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, V.T, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solución con el metodo de relajación de V(x,y)')
plt.savefig('Solución con el metodo de relajación de V(x,y)')
plt.show()

# graficamos la solución en 3D
plot_solución_3d(V, dx, dy)


# graficamos la solución analitica en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, V_analitica, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlim(-1,1)
ax.set_zlabel('intensidad V')
ax.set_title('Solución analítica en 3D')
plt.savefig('Solución analítica en 3D')
plt.show()

fig = plt.figure()
plot_difference_scatter_3d(V.T, V_analitica, a, b)
plt.show()
