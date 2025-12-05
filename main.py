# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate(m, q, E, v0, t_max, dt):
    """Simulate motion of a charged particle in a uniform electric field.
    Parameters
    ----------
    m : float
        Mass of the particle (kg).
    q : float
        Charge of the particle (C).
    E : float
        Electric field magnitude along the x‑axis (V/m).
    v0 : array_like, shape (2,)
        Initial velocity vector (vx0, vy0) (m/s).
    t_max : float
        Total simulation time (s).
    dt : float
        Time step (s).
    Returns
    -------
    t : ndarray
        Time array.
    pos : ndarray, shape (N, 2)
        Positions (x, y) at each time step.
    vel : ndarray, shape (N, 2)
        Velocities (vx, vy) at each time step.
    """
    n_steps = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n_steps)
    pos = np.zeros((n_steps, 2))
    vel = np.zeros((n_steps, 2))
    vel[0] = v0
    a = np.array([q * E / m, 0.0])  # constant acceleration
    for i in range(1, n_steps):
        # explicit Euler integration
        vel[i] = vel[i-1] + a * dt
        pos[i] = pos[i-1] + vel[i-1] * dt
    return t, pos, vel

def exp1():
    # Parameters for experiment 1
    m = 1.0          # kg
    q = 1.0          # C
    E = 1.0          # V/m (uniform field along +x)
    v0 = np.array([0.0, 1.0])  # initial velocity (upward)
    t_max = 10.0    # seconds
    dt = 0.01       # time step

    t, pos, vel = simulate(m, q, E, v0, t_max, dt)
    speed = np.linalg.norm(vel, axis=1)

    # Plot trajectory (y vs x)
    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1])
    plt.title('Trajectory in Uniform Electric Field')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_uniform_field.png')
    plt.close()

    # Plot speed vs time
    plt.figure()
    plt.plot(t, speed)
    plt.title('Speed vs Time (Uniform Field)')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    plt.savefig('speed_vs_time_uniform.png')
    plt.close()

    return t, pos, vel

def exp2():
    # Parameters common to all runs
    m = 1.0
    q = 1.0
    v0 = np.array([0.0, 1.0])
    t_max = 10.0
    dt = 0.01
    field_strengths = [0.5, 1.0, 2.0]  # V/m

    trajectories = []
    final_displacements = []

    plt.figure()
    for E in field_strengths:
        t, pos, vel = simulate(m, q, E, v0, t_max, dt)
        trajectories.append((E, pos))
        final_displacements.append(pos[-1, 0])  # final x position
        plt.plot(pos[:, 0], pos[:, 1], label=f'E = {E} V/m')
    plt.title('Trajectories for Different Field Strengths')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_vs_field_strength.png')
    plt.close()

    # Scatter plot of final x displacement vs field strength
    plt.figure()
    plt.scatter(field_strengths, final_displacements)
    plt.title('Final x‑Displacement vs Electric Field Strength')
    plt.xlabel('Electric Field Strength (V/m)')
    plt.ylabel('Final x Displacement (m)')
    plt.grid(True)
    plt.savefig('final_displacement_vs_field.png')
    plt.close()

    return field_strengths, final_displacements

if __name__ == '__main__':
    # Run experiments
    exp1()
    fields, final_x = exp2()
    # Primary numeric answer: final x displacement for the strongest field (2.0 V/m)
    answer = final_x[-1]
    print('Answer:', answer)

