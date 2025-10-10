"""
Evaluate PID controller on spiral trajectory using CF2X_FastStates_HoverAviary and DSLPIDControl.
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add gym-pybullet-drones to path
sys.path.append('/home/osos/Mohamed_Masters_Thesis/PID_drone/gym-pybullet-drones')

from cf2x_fast_states_env import CF2X_FastStates_HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# --- Config ---
EPISODE_LEN_SEC = 25.0
CTRL_FREQ = 30
GUI = True
RECORD = True
TRAJECTORY_TYPE = "spiral"
TARGET_POS = np.array([0.0, 0.0, 1.0])

# Use true min/max RPMs based on DSLPIDControl's PWM2RPM conversion
PWM2RPM_SCALE = 0.2685
PWM2RPM_CONST = 4070.3
MIN_PWM = 20000
MAX_PWM = 65535
RPM_MIN = PWM2RPM_SCALE * MIN_PWM + PWM2RPM_CONST
RPM_MAX = PWM2RPM_SCALE * MAX_PWM + PWM2RPM_CONST

# --- Environment ---
env = CF2X_FastStates_HoverAviary(
    target_pos=TARGET_POS,
    gui=GUI,
    record=RECORD,
    use_trajectory=True,
    trajectory_type=TRAJECTORY_TYPE,
    full_state_obs=True
)

# --- PID Controller ---
pid = DSLPIDControl(drone_model=DroneModel.CF2X)

# --- Data Logging ---

# --- Logging for DHP-style best episode plots ---
log = {
    'time': [],
    'states': [],         # [z, roll, pitch, yaw, vz, wx, wy, wz]
    'references': [],     # [z_ref, roll_ref, pitch_ref, yaw_ref, vz_ref, wx_ref, wy_ref, wz_ref]
    'actions': [],        # Motor RPMs
    'x_positions': [],
    'y_positions': [],
    'x_references': [],
    'y_references': [],
    'position_error': [],
}

obs, info = env.reset()
state = env.get_full_state()


for step in range(int(EPISODE_LEN_SEC * CTRL_FREQ)):
    t = step / CTRL_FREQ
    target_pos = env.get_current_target_pos()

    # State extraction (from full state vector)
    pos = state[0:3]
    quat = state[3:7]
    rpy = state[7:10]
    vel = state[10:13]
    ang_vel = state[13:16]

    # Compose fast states (match DHP best episode logging)
    # fast states: [z, roll, pitch, yaw, vz, wx, wy, wz]
    z = pos[2]
    roll, pitch, yaw = rpy
    vz = vel[2]
    wx, wy, wz = ang_vel
    state_fast = np.array([z, roll, pitch, yaw, vz, wx, wy, wz])
    # Use mathematically correct reference from environment
    reference_fast = env.generate_reference(target_pos)

    # PID control
    target_rpy = np.zeros(3)
    target_vel = np.zeros(3)
    target_ang_vel = np.zeros(3)
    rpm_tuple = pid.computeControl(
        1.0/CTRL_FREQ,
        pos,
        quat,
        vel,
        ang_vel,
        target_pos,
        target_rpy
    )
    rpm = rpm_tuple[0]
    rpm_normalized = 2 * (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN) - 1
    obs, reward, terminated, truncated, info = env.step(rpm_normalized)
    state = env.get_full_state()

    # Log DHP-style arrays
    log['time'].append(t)
    log['states'].append(state_fast.copy())
    log['references'].append(reference_fast.copy())
    log['actions'].append(rpm.copy())
    log['x_positions'].append(pos[0])
    log['y_positions'].append(pos[1])
    log['x_references'].append(target_pos[0])
    log['y_references'].append(target_pos[1])
    log['position_error'].append(np.linalg.norm(pos - target_pos))

    if terminated or truncated:
        break



# --- DHP-style Best Episode Plotting for PID ---
def plot_pid_best_episode(log):

    # Convert logs to arrays
    for k in log:
        if not isinstance(log[k], np.ndarray):
            log[k] = np.array(log[k])
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        sns.set_context("notebook", font_scale=1.1)
        colors = sns.color_palette("husl", 10)
    except ImportError:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    states = log['states']
    references = log['references']
    actions = log['actions']
    t = log['time']
    x_pos = log['x_positions']
    y_pos = log['y_positions']
    x_ref = log['x_references']
    y_ref = log['y_references']
    z_pos = states[:, 0]
    z_ref = references[:, 0]
    roll = states[:, 1] * 180/np.pi
    roll_ref = references[:, 1] * 180/np.pi
    pitch = states[:, 2] * 180/np.pi
    pitch_ref = references[:, 2] * 180/np.pi
    yaw = states[:, 3] * 180/np.pi
    yaw_ref = references[:, 3] * 180/np.pi
    vz = states[:, 4]
    vz_ref = references[:, 4]
    wx = states[:, 5] * 180/np.pi
    wx_ref = references[:, 5] * 180/np.pi
    wy = states[:, 6] * 180/np.pi
    wy_ref = references[:, 6] * 180/np.pi
    wz = states[:, 7] * 180/np.pi
    wz_ref = references[:, 7] * 180/np.pi
    motor1 = actions[:, 0]
    motor2 = actions[:, 1]
    motor3 = actions[:, 2]
    motor4 = actions[:, 3]

    # Figure 1: Vertical control (z, vz, error, model error placeholder, cost placeholder)
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(4,1,1)
    ax1.plot(t, z_pos, 'b-', label='z (PID)')
    ax1.plot(t, z_ref, 'r--', label='z_ref')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('PID Best Episode Performance')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, vz, 'b-', label='vz (PID)')
    ax2.plot(t, vz_ref, 'r--', label='vz_ref')
    ax2.set_ylabel('Vertical Velocity [m/s]')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, log['position_error'], 'b-', label='Position Error (PID)')
    ax3.set_ylabel('Position Error [m]')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, motor1, 'b-', label='Motor 1 (PID)')
    ax4.plot(t, motor2, 'r-', label='Motor 2 (PID)')
    ax4.plot(t, motor3, 'g-', label='Motor 3 (PID)')
    ax4.plot(t, motor4, 'orange', label='Motor 4 (PID)')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Motor RPM')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    fig1.tight_layout()
    fig1.savefig('pid_vertical_control.png', dpi=150, bbox_inches='tight')

    # Figure 2: Attitude control - Roll
    fig2 = plt.figure(figsize=(10, 10))
    ax1 = fig2.add_subplot(4,1,1)
    ax1.plot(t, roll, 'b-', label='Roll (PID)')
    ax1.plot(t, roll_ref, 'r--', label='Roll_ref')
    ax1.set_ylabel('Roll [deg]')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, wx, 'b-', label='wx (PID)')
    ax2.plot(t, wx_ref, 'r--', label='wx_ref')
    ax2.set_ylabel('Roll Rate [deg/s]')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, motor2, 'g-', label='Motor 2 (PID)')
    ax3.plot(t, motor4, 'orange', label='Motor 4 (PID)')
    ax3.set_ylabel('Motor 2/4 RPM')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, pitch, 'b-', label='Pitch (PID)')
    ax4.plot(t, pitch_ref, 'r--', label='Pitch_ref')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Pitch [deg]')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    fig2.tight_layout()
    fig2.savefig('pid_roll_control.png', dpi=150, bbox_inches='tight')

    # Figure 3: Attitude control - Pitch & Yaw
    fig3 = plt.figure(figsize=(10, 10))
    ax1 = fig3.add_subplot(4,1,1)
    ax1.plot(t, wy, 'b-', label='wy (PID)')
    ax1.plot(t, wy_ref, 'r--', label='wy_ref')
    ax1.set_ylabel('Pitch Rate [deg/s]')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    ax2.plot(t, motor1, 'b-', label='Motor 1 (PID)')
    ax2.plot(t, motor3, 'r-', label='Motor 3 (PID)')
    ax2.set_ylabel('Motor 1/3 RPM')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    ax3.plot(t, yaw, 'b-', label='Yaw (PID)')
    ax3.plot(t, yaw_ref, 'r--', label='Yaw_ref')
    ax3.set_ylabel('Yaw [deg]')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    ax4.plot(t, wz, 'b-', label='wz (PID)')
    ax4.plot(t, wz_ref, 'r--', label='wz_ref')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Yaw Rate [deg/s]')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    fig3.tight_layout()
    fig3.savefig('pid_pitch_yaw_control.png', dpi=150, bbox_inches='tight')

    # Figure 4: XYZ Position Control
    fig4 = plt.figure(figsize=(14, 12))
    ax1 = fig4.add_subplot(5,1,1)
    ax1.plot(t, x_pos, color=colors[0], linewidth=2, label='x (PID)')
    ax1.plot(t, x_ref, '--', color=colors[1], linewidth=2, label='x_ref')
    ax1.set_ylabel('X Position [m]')
    ax1.set_title('XYZ Position Control - PID')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(5,1,2, sharex=ax1)
    ax2.plot(t, y_pos, color=colors[2], linewidth=2, label='y (PID)')
    ax2.plot(t, y_ref, '--', color=colors[3], linewidth=2, label='y_ref')
    ax2.set_ylabel('Y Position [m]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(5,1,3, sharex=ax1)
    ax3.plot(t, z_pos, color=colors[4], linewidth=2, label='z (PID)')
    ax3.plot(t, z_ref, '--', color=colors[5], linewidth=2, label='z_ref')
    ax3.set_ylabel('Z Position [m]')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(5,1,4, sharex=ax1)
    x_error = x_pos - x_ref
    y_error = y_pos - y_ref
    z_error = z_pos - z_ref
    xyz_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
    ax4.plot(t, x_error, color=colors[0], linewidth=2, label='e_x (PID)')
    ax4.plot(t, y_error, color=colors[2], linewidth=2, label='e_y (PID)')
    ax4.plot(t, z_error, color=colors[4], linewidth=2, label='e_z (PID)')
    ax4.plot(t, xyz_error, color='red', linewidth=3, label='|e_xyz| (PID)')
    ax4.set_ylabel('Position Errors [m]')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(5,1,5)
    ax5.plot(x_pos, y_pos, color=colors[0], linewidth=3, label='Actual XY Trajectory (PID)')
    ax5.plot(x_ref, y_ref, '--', color=colors[1], linewidth=3, label='Reference XY Trajectory')
    ax5.plot(x_pos[0], y_pos[0], 'o', color='green', markersize=10, label='Start')
    ax5.plot(x_pos[-1], y_pos[-1], 'o', color='red', markersize=10, label='End')
    ax5.set_xlabel('X Position [m]')
    ax5.set_ylabel('Y Position [m]')
    ax5.set_title('2D Trajectory View (XY Plane)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    fig4.tight_layout()
    fig4.savefig('pid_xyz_position_control.png', dpi=150, bbox_inches='tight')

    # 3D Trajectory Plot
    fig4_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig4_3d.add_subplot(111, projection='3d')
    ax_3d.plot(x_pos, y_pos, z_pos, color=colors[0], linewidth=3, label='Actual 3D Trajectory (PID)')
    ax_3d.plot(x_ref, y_ref, z_ref, '--', color=colors[1], linewidth=3, label='Reference 3D Trajectory')
    ax_3d.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=100, label='Start', marker='o')
    ax_3d.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=100, label='End', marker='s')
    ax_3d.set_xlabel('X Position [m]')
    ax_3d.set_ylabel('Y Position [m]')
    ax_3d.set_zlabel('Z Position [m]')
    ax_3d.set_title('3D Quadrotor Trajectory - PID')
    ax_3d.legend(loc='upper left')
    # Set equal aspect ratio for 3D plot
    max_range = max(
        np.max(x_pos) - np.min(x_pos),
        np.max(y_pos) - np.min(y_pos),
        np.max(z_pos) - np.min(z_pos)
    ) / 2.0
    mid_x = (np.max(x_pos) + np.min(x_pos)) * 0.5
    mid_y = (np.max(y_pos) + np.min(y_pos)) * 0.5
    mid_z = (np.max(z_pos) + np.min(z_pos)) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_3d.grid(True, alpha=0.3)
    fig4_3d.tight_layout()
    fig4_3d.savefig('pid_3d_trajectory.png', dpi=150, bbox_inches='tight')

    plt.show()
    print('PID best episode style plots generated and saved!')

plot_pid_best_episode(log)
