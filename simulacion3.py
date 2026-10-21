import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. PARÁMETROS FÍSICOS DEL SISTEMA
# ==========================================
g = 9.81      # Aceleración de la gravedad (m/s^2)
a = 1.0       # Parámetro de la parábola z = a*x^2 (1/m)
m = 1.0       # Masa del vehículo (kg)

# ==========================================
# 2. ECUACIÓN DE MOVIMIENTO (Lagrangiana)
# ==========================================
def aceleracion_lagrangiana(x, v):
    """
    Calcula la aceleración x_ddot derivada de la lagrangiana:
    (1 + 4a^2*x^2)*x_ddot + 4a^2*x*v^2 = -2*g*a*x
    """
    numerador = -2 * g * a * x - 4 * (a**2) * x * (v**2)
    denominador = 1 + 4 * (a**2) * (x**2)
    return numerador / denominador

# Método numérico Runge-Kutta 4 (RK4) de alta precisión
def rk4_step(x, v, dt):
    k1_x = v
    k1_v = aceleracion_lagrangiana(x, v)
    
    x2 = x + 0.5 * dt * k1_x
    v2 = v + 0.5 * dt * k1_v
    k2_x = v2
    k2_v = aceleracion_lagrangiana(x2, v2)
    
    x3 = x + 0.5 * dt * k2_x
    v3 = v + 0.5 * dt * k2_v
    k3_x = v3
    k3_v = aceleracion_lagrangiana(x3, v3)
    
    x4 = x + dt * k3_x
    v4 = v + dt * k3_v
    k4_x = v4
    k4_v = aceleracion_lagrangiana(x4, v4)
    
    x_next = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v_next = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return x_next, v_next

# ==========================================
# 3. PRE-CÁLCULO DE LA TRAYECTORIA TEMPORAL
# ==========================================
dt = 0.015
tf = 8.0
n_steps = int(tf / dt)

t_list, x_list, v_list = [], [], []
t = 0.0
x = 1.2   # Posición inicial en x (soltado desde x = 1.2 m)
v = 0.0   # Velocidad inicial nula (partiendo del reposo)

for _ in range(n_steps):
    t_list.append(t)
    x_list.append(x)
    v_list.append(v)
    x, v = rk4_step(x, v, dt)
    t += dt

x_arr = np.array(x_list)
z_arr = a * (x_arr ** 2)

# ==========================================
# 4. CONFIGURACIÓN DE LA ANIMACIÓN GRÁFICA
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# Subplot 1: Animación física en el plano X-Z
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.2, 2.0)
ax1.set_xlabel('Posición $x$ (m)')
ax1.set_ylabel('Posición $z$ (m)')
ax1.set_title('Simulación Física Real: Vehículo en el Raíl Parabólico')
ax1.grid(True)

# Dibujar el raíl estático z = ax^2
x_par = np.linspace(-1.5, 1.5, 300)
z_par = a * (x_par ** 2)
ax1.plot(x_par, z_par, 'b--', label='Raíl $z = ax^2$', alpha=0.6, linewidth=2)

vehicle, = ax1.plot([], [], 'ro', markersize=12, label='Vehículo ($m$)')
trajectory, = ax1.plot([], [], 'r-', alpha=0.5, linewidth=1.5, label='Trayectoria')
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
ax1.legend(loc='upper right')

# Subplot 2: Evolución de la coordenada generalizada x(t)
ax2.set_xlim(0, tf)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('Tiempo $t$ (s)')
ax2.set_ylabel('Coordenada generalizada $x(t)$ (m)')
ax2.set_title('Evolución temporal de la posición')
ax2.grid(True)
line_x, = ax2.plot([], [], 'g-', linewidth=2, label='Posición $x(t)$')
ax2.legend(loc='upper right')

def init():
    vehicle.set_data([], [])
    trajectory.set_data([], [])
    time_text.set_text('')
    line_x.set_data([], [])
    return vehicle, trajectory, time_text, line_x

def update(frame):
    xi = x_arr[frame]
    zi = z_arr[frame]
    
    # Actualizar posición del vehículo en la parábola
    vehicle.set_data([xi], [zi])
    trajectory.set_data(x_arr[:frame+1], z_arr[:frame+1])
    time_text.set_text(f'Tiempo: {t_list[frame]:.2f} s\nx = {xi:.2f} m')
    
    # Actualizar gráfica temporal
    line_x.set_data(t_list[:frame+1], x_arr[:frame+1])
    
    return vehicle, trajectory, time_text, line_x

ani = animation.FuncAnimation(fig, update, frames=len(x_arr), init_func=init, interval=20, blit=True)

plt.tight_layout()
plt.show()