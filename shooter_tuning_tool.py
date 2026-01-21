"""
FRC 2026 Shooter Tuning Tool
Interactive tool for optimizing shooter parameters with real-time feedback.
"""
import json
import os
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# =============================================================================
# PRESETS AND CONSTANTS
# =============================================================================

MOTOR_PRESETS = {
    '1': {'name': 'Kraken X44', 'free_speed_rpm': 7530, 'stall_torque_nm': 4.05,
          'peak_torque_nm': 2.0, 'stall_current_a': 275, 'default_current_limit': 40},
    '2': {'name': 'Kraken X60', 'free_speed_rpm': 5800, 'stall_torque_nm': 9.37,
          'peak_torque_nm': 4.7, 'stall_current_a': 483, 'default_current_limit': 60},
    '3': {'name': 'Falcon 500', 'free_speed_rpm': 6380, 'stall_torque_nm': 4.69,
          'peak_torque_nm': 2.35, 'stall_current_a': 257, 'default_current_limit': 40},
    '4': {'name': 'NEO', 'free_speed_rpm': 5676, 'stall_torque_nm': 2.6,
          'peak_torque_nm': 1.3, 'stall_current_a': 105, 'default_current_limit': 40},
    '5': {'name': 'NEO Vortex', 'free_speed_rpm': 6784, 'stall_torque_nm': 3.6,
          'peak_torque_nm': 1.8, 'stall_current_a': 211, 'default_current_limit': 40},
}

FLYWHEEL_PRESETS = {
    '1': {'name': 'Base (Stealth + AM insert)', 'moi_lb_in2': 4.68},
    '2': {'name': '+ WCP Flywheels (2x)', 'moi_lb_in2': 10.08},
    '3': {'name': '+ Dual Stealth per side', 'moi_lb_in2': 9.36},
    '4': {'name': 'WCP + Dual Stealth (Recommended)', 'moi_lb_in2': 14.76},
    '5': {'name': 'Custom MOI', 'moi_lb_in2': None},
}

GEAR_RATIOS = {'1': 1.0, '2': 1.5, '3': 2.0, '4': 2.5, '5': 3.0}

# Physics constants
G = 9.81  # m/s^2
BALL_DIAMETER_M = 5.9 * 0.0254
BALL_RADIUS_M = BALL_DIAMETER_M / 2
BALL_MASS_KG = 0.5 * 0.453592
BALL_CROSS_SECTION = np.pi * BALL_RADIUS_M**2
AIR_DENSITY = 1.2  # kg/m^3

# Target geometry
RIM_HEIGHT_M = 72 * 0.0254
EXIT_HEIGHT_M = 20 * 0.0254
DELTA_Y = RIM_HEIGHT_M - EXIT_HEIGHT_M

# Shooting distances
DISTANCES_FT = [8, 10, 12, 14, 16, 18, 20]
DISTANCES_M = [d * 0.3048 for d in DISTANCES_FT]

# Unit conversions
def m_to_ft(m): return m / 0.3048
def ms_to_fps(ms): return ms / 0.3048

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

def get_default_config() -> Dict:
    """Return default configuration."""
    motor = MOTOR_PRESETS['1']
    return {
        'motor_name': motor['name'],
        'motor_free_speed_rpm': motor['free_speed_rpm'],
        'motor_stall_torque_nm': motor['stall_torque_nm'],
        'motor_peak_torque_nm': motor['peak_torque_nm'],
        'motor_stall_current_a': motor['stall_current_a'],
        'wheels_per_side': 1,
        'motor_num_motors': 2,
        'motor_current_limit_a': motor['default_current_limit'],
        'motor_efficiency': 0.85,
        'selected_angle': None,  # None = use optimal
        'idle_speed_rpm': 500,
        'gear_ratio': 2.0,
        'flywheel_name': 'WCP + Dual Stealth (Recommended)',
        'flywheel_moi_lb_in2': 14.76,
        'drag_coefficient': 0.50,
        'slip_factor': 1.15,
        'wheel_diameter_in': 4.0,
    }

# =============================================================================
# CORE PHYSICS FUNCTIONS
# =============================================================================

def simulate_trajectory(v0: float, angle_deg: float, drag_coeff: float, dt: float = 0.0001) -> Dict:
    """Simulate trajectory with quadratic air resistance."""
    drag_constant = 0.5 * AIR_DENSITY * drag_coeff * BALL_CROSS_SECTION
    theta = np.radians(angle_deg)
    x, y = 0.0, 0.0
    vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)
    xs, ys, vxs, vys = [x], [y], [vx], [vy]
    t = 0.0

    # Simulate until ball falls below ground or goes too far
    max_time = 5.0
    while t < max_time and y >= -EXIT_HEIGHT_M and x < 30:
        v = np.sqrt(vx**2 + vy**2)
        if v > 0:
            a_drag = drag_constant * v**2 / BALL_MASS_KG
            ax = -a_drag * vx / v
            ay = -G - a_drag * vy / v
        else:
            ax, ay = 0, -G
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        # Subsample for memory efficiency
        if len(xs) < 10000 or t - len(xs) * dt > 0.001:
            xs.append(x)
            ys.append(y)
            vxs.append(vx)
            vys.append(vy)

    return {'x': np.array(xs), 'y': np.array(ys), 'vx': np.array(vxs), 'vy': np.array(vys)}


def find_rim_crossing(traj: Dict) -> Tuple[float, float, float, bool]:
    """Find where trajectory crosses rim height (descending)."""
    xs, ys, vxs, vys = traj['x'], traj['y'], traj['vx'], traj['vy']
    for i in range(1, len(ys)):
        if ys[i-1] >= DELTA_Y and ys[i] < DELTA_Y and vys[i] < 0:
            frac = (DELTA_Y - ys[i-1]) / (ys[i] - ys[i-1])
            x_cross = xs[i-1] + frac * (xs[i] - xs[i-1])
            vy_cross = vys[i-1] + frac * (vys[i] - vys[i-1])
            vx_cross = vxs[i-1] + frac * (vxs[i] - vxs[i-1])
            entry_angle = np.degrees(np.arctan2(vy_cross, vx_cross))
            return x_cross, vy_cross, entry_angle, True
    return np.inf, 0, 0, False


def find_required_velocity(target_m: float, angle_deg: float, drag_coeff: float) -> Tuple[float, Dict]:
    """Binary search for exit velocity to hit target distance."""
    v_low, v_high = 1.0, 50.0
    for _ in range(50):
        v_mid = (v_low + v_high) / 2
        traj = simulate_trajectory(v_mid, angle_deg, drag_coeff)
        x_land, _, _, valid = find_rim_crossing(traj)
        if not valid:
            v_low = v_mid
        elif abs(x_land - target_m) < 0.01:
            return v_mid, traj
        elif x_land < target_m:
            v_low = v_mid
        else:
            v_high = v_mid
    return v_mid, traj


def calc_sensitivity(angle_deg: float, v0: float, drag_coeff: float) -> Tuple[float, float]:
    """Calculate landing sensitivity to velocity and angle errors."""
    traj_base = simulate_trajectory(v0, angle_deg, drag_coeff)
    x_base, _, _, _ = find_rim_crossing(traj_base)

    delta_v, delta_a = 0.1, 0.5
    x_plus_v, _, _, _ = find_rim_crossing(simulate_trajectory(v0 + delta_v, angle_deg, drag_coeff))
    x_minus_v, _, _, _ = find_rim_crossing(simulate_trajectory(v0 - delta_v, angle_deg, drag_coeff))
    x_plus_a, _, _, _ = find_rim_crossing(simulate_trajectory(v0, angle_deg + delta_a, drag_coeff))
    x_minus_a, _, _, _ = find_rim_crossing(simulate_trajectory(v0, angle_deg - delta_a, drag_coeff))

    dxdv = abs((x_plus_v - x_minus_v) / (2 * delta_v))
    dxda = abs((x_plus_a - x_minus_a) / (2 * delta_a))
    return dxdv, dxda


def calc_speed_drop(initial_rpm: float, exit_vel_ms: float, moi_kg_m2: float, efficiency: float = 0.70) -> float:
    """Calculate percentage speed drop after launching a ball."""
    omega_initial = initial_rpm * 2 * np.pi / 60
    E_flywheel = 0.5 * moi_kg_m2 * omega_initial**2
    E_ball = 0.5 * BALL_MASS_KG * exit_vel_ms**2
    E_taken = E_ball / efficiency
    E_new = max(0, E_flywheel - E_taken)
    omega_final = np.sqrt(2 * E_new / moi_kg_m2) if E_new > 0 else 0
    return ((omega_initial - omega_final) / omega_initial) * 100 if omega_initial > 0 else 100


def calc_spinup_time(target_rpm: float, gear_ratio: float, moi_kg_m2: float,
                     idle_rpm: float, config: Dict) -> float:
    """Calculate time to spin up from idle to target RPM."""
    free_speed_rads = config['motor_free_speed_rpm'] * 2 * np.pi / 60
    kt = config['motor_stall_torque_nm'] / config['motor_stall_current_a']
    torque_at_limit = config['motor_current_limit_a'] * kt
    usable_torque = min(config['motor_peak_torque_nm'], torque_at_limit)

    target_rad_s = target_rpm * 2 * np.pi / 60
    motor_target_rad_s = target_rad_s * gear_ratio
    J_motor = moi_kg_m2 / (gear_ratio ** 2)

    dt = 0.0001
    omega = (idle_rpm * 2 * np.pi / 60) * gear_ratio
    t = 0

    while omega < motor_target_rad_s and t < 5:
        speed_fraction = omega / free_speed_rads
        tau_curve = config['motor_stall_torque_nm'] * (1 - speed_fraction)
        tau_motor = min(tau_curve, usable_torque)
        tau_total = config['motor_num_motors'] * tau_motor * config['motor_efficiency']
        alpha = tau_total / J_motor
        omega += alpha * dt
        t += dt

    return t

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_analysis(config: Dict) -> Dict:
    """Run full shooter analysis and return results."""
    drag_coeff = config['drag_coefficient']
    slip_factor = config['slip_factor']
    wheel_diam_m = config['wheel_diameter_in'] * 0.0254
    
    # Calculate total MOI including flywheel and wheels
    flywheel_moi_lb_in2 = config['flywheel_moi_lb_in2']
    
    # Get wheel MOI (per wheel) - estimate from diameter if not specified
    wheel_moi_lb_in2 = config.get('wheel_moi_lb_in2')
    if wheel_moi_lb_in2 is None:
        # Rough approximation: moi ≈ 0.225 * diameter^2
        wheel_moi_lb_in2 = 0.225 * config['wheel_diameter_in'] ** 2
    
    # Total MOI = flywheel MOI + (wheel MOI per wheel × wheels per side × 2 sides)
    wheels_per_side = config.get('wheels_per_side', 1)
    total_wheel_moi_lb_in2 = wheel_moi_lb_in2 * wheels_per_side * 2
    moi_lb_in2 = flywheel_moi_lb_in2 + total_wheel_moi_lb_in2
    
    moi_kg_m2 = moi_lb_in2 / (1 / 0.453592 / (0.0254**2))
    gear_ratio = config['gear_ratio']
    idle_rpm = config['idle_speed_rpm']
    eff_free_speed = config['motor_free_speed_rpm'] / gear_ratio

    # Test angles 50-75
    angles = np.arange(50, 76, 1)
    angle_results = []

    for angle in angles:
        velocities, entry_angles = [], []
        valid = True
        for dist_m in DISTANCES_M:
            v, traj = find_required_velocity(dist_m, angle, drag_coeff)
            _, _, entry, ok = find_rim_crossing(traj)
            if not ok or v > 40:
                valid = False
                break
            velocities.append(v)
            entry_angles.append(entry)

        if valid:
            velocities = np.array(velocities)
            sens_v_list, sens_a_list = [], []
            for i, dist_m in enumerate(DISTANCES_M):
                dxdv, dxda = calc_sensitivity(angle, velocities[i], drag_coeff)
                sens_v_list.append(dxdv)
                sens_a_list.append(dxda)

            v_range = velocities.max() - velocities.min()
            sens_v_mean = np.mean(sens_v_list)
            sens_a_mean = np.mean(sens_a_list)
            score = sens_v_mean * 2.0 + sens_a_mean * 1.0 + v_range / 5.0

            angle_results.append({
                'angle': angle, 'velocities': velocities, 'entry_angles': entry_angles,
                'v_min': velocities.min(), 'v_max': velocities.max(), 'v_range': v_range,
                'sens_v_mean': sens_v_mean, 'sens_a_mean': sens_a_mean, 'score': score
            })

    if not angle_results:
        return {'error': 'No valid angles found'}

    # Find optimal and selected angle
    angle_results.sort(key=lambda x: x['score'])
    optimal = angle_results[0]

    selected_angle = config['selected_angle']
    if selected_angle is not None:
        selected_list = [r for r in angle_results if r['angle'] == int(selected_angle)]
        selected = selected_list[0] if selected_list else optimal
    else:
        selected = optimal

    # Calculate wheel RPMs and related metrics
    wheel_rpms = {}
    shot_data = {}
    for i, dist_ft in enumerate(DISTANCES_FT):
        v_exit = selected['velocities'][i]
        surface_speed = v_exit * slip_factor
        wheel_rpm = (surface_speed / (np.pi * wheel_diam_m)) * 60
        wheel_rpms[dist_ft] = wheel_rpm
        shot_data[dist_ft] = {
            'v_exit_ms': v_exit, 'v_exit_fps': ms_to_fps(v_exit),
            'wheel_rpm': wheel_rpm, 'entry_angle': selected['entry_angles'][i]
        }

    min_rpm, max_rpm = min(wheel_rpms.values()), max(wheel_rpms.values())
    headroom = (1 - max_rpm / eff_free_speed) * 100

    # Speed drop at 20ft
    speed_drop_20 = calc_speed_drop(wheel_rpms[20], shot_data[20]['v_exit_ms'], moi_kg_m2)

    # Spin-up times
    spinup_8 = calc_spinup_time(wheel_rpms[8], gear_ratio, moi_kg_m2, idle_rpm, config) * 1000
    spinup_20 = calc_spinup_time(wheel_rpms[20], gear_ratio, moi_kg_m2, idle_rpm, config) * 1000

    return {
        'optimal_angle': optimal['angle'],
        'optimal_score': optimal['score'],
        'selected_angle': selected['angle'],
        'selected_score': selected['score'],
        'sens_v_mean': selected['sens_v_mean'],
        'sens_a_mean': selected['sens_a_mean'],
        'v_range': selected['v_range'],
        'min_rpm': min_rpm,
        'max_rpm': max_rpm,
        'headroom': headroom,
        'eff_free_speed': eff_free_speed,
        'speed_drop_20': speed_drop_20,
        'spinup_8_ms': spinup_8,
        'spinup_20_ms': spinup_20,
        'wheel_rpms': wheel_rpms,
        'shot_data': shot_data,
        'all_angles': angle_results,
    }

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def status_icon(value: float, good: float, warn: float, lower_better: bool = True) -> str:
    """Return status icon based on thresholds."""
    if lower_better:
        if value <= good: return "[OK]"
        if value <= warn: return "[!]"
        return "[X]"
    else:
        if value >= good: return "[OK]"
        if value >= warn: return "[!]"
        return "[X]"


def display_results(config: Dict, results: Dict):
    """Display compact results summary."""
    if 'error' in results:
        print(f"\nError: {results['error']}")
        return

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Configuration line
    print(f"\nConfiguration:")
    print(f"  Motor: {config['motor_name']} (x{config['motor_num_motors']}) @ {config['motor_current_limit_a']}A")
    print(f"  Gear: {config['gear_ratio']}:1 | Flywheel: {config['flywheel_moi_lb_in2']:.1f} lb-in^2 | Wheel: {config['wheel_diameter_in']}\"")
    print(f"  Drag: {config['drag_coefficient']:.2f} | Slip: {config['slip_factor']:.2f} | Idle: {config['idle_speed_rpm']} RPM")

    # Angle info
    angle_note = "(optimal)" if results['selected_angle'] == results['optimal_angle'] else f"(optimal: {results['optimal_angle']})"
    print(f"\nAngle: {results['selected_angle']} deg {angle_note}")

    # Performance metrics
    print(f"\nPerformance Metrics:")
    icon = status_icon(results['selected_score'], 4.0, 6.0)
    print(f"  Consistency Score: {results['selected_score']:.2f} {icon} (lower is better)")
    icon = status_icon(results['sens_v_mean'], 1.0, 1.5)
    print(f"  Velocity Sensitivity: {results['sens_v_mean']:.3f} m/(m/s) {icon}")
    icon = status_icon(results['sens_a_mean'], 0.1, 0.2)
    print(f"  Angle Sensitivity: {results['sens_a_mean']:.3f} m/deg {icon}")

    print(f"\n  Spin-up Time ({config['gear_ratio']}:1 ratio):")
    icon = status_icon(results['spinup_8_ms'], 300, 500)
    print(f"    8ft shot:  {results['spinup_8_ms']:.0f} ms {icon}")
    icon = status_icon(results['spinup_20_ms'], 400, 600)
    print(f"    20ft shot: {results['spinup_20_ms']:.0f} ms {icon}")

    icon = status_icon(results['speed_drop_20'], 15, 25)
    print(f"\n  Speed Drop: {results['speed_drop_20']:.1f}% {icon} (per shot at 20ft)")

    print(f"\n  RPM Requirements:")
    print(f"    Range: {results['min_rpm']:.0f} - {results['max_rpm']:.0f} RPM")
    icon = status_icon(results['headroom'], 30, 10, lower_better=False)
    print(f"    Headroom: {results['headroom']:.1f}% {icon}")

    # Recommendations
    print(f"\nRecommendations:")
    recs = get_recommendations(config, results)
    for rec in recs:
        print(f"  {rec}")


def get_recommendations(config: Dict, results: Dict) -> list:
    """Generate recommendations based on results."""
    recs = []

    if results['headroom'] < 10:
        recs.append("[!] Low headroom - increase gear ratio or use faster motor")
    if results['headroom'] < 0:
        recs.append("[X] CRITICAL: Exceeds motor free speed! Must increase gear ratio")

    if results['speed_drop_20'] > 25:
        recs.append("[!] High speed drop - consider more flywheel mass")
    elif results['speed_drop_20'] > 15:
        recs.append("[-] Speed drop slightly high - could add flywheel mass")

    if results['spinup_20_ms'] > 600:
        recs.append("[!] Slow spin-up - reduce flywheel mass or increase gear ratio")

    if results['selected_angle'] != results['optimal_angle']:
        diff = results['selected_score'] - results['optimal_score']
        if diff > 0.5:
            recs.append(f"[-] Angle {results['optimal_angle']} deg has better consistency (score: {results['optimal_score']:.2f})")

    if not recs:
        recs.append("[OK] Configuration looks good!")

    return recs

# =============================================================================
# INPUT HELPERS
# =============================================================================

def prompt_float(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Prompt for float with default and validation."""
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        try:
            v = float(val)
            if min_val is not None and v < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("Invalid number. Try again.")


def prompt_int(prompt: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """Prompt for int with default and validation."""
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        try:
            v = int(val)
            if min_val is not None and v < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("Invalid integer. Try again.")


def prompt_choice(prompt: str, options: Dict, default: str) -> str:
    """Prompt for choice from options."""
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return default
        if val in options:
            return val
        print(f"Invalid choice. Options: {', '.join(options.keys())}")

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_initial_config() -> Dict:
    """Get initial configuration from user."""
    print("=" * 80)
    print("FRC 2026 SHOOTER TUNING TOOL")
    print("=" * 80)
    print("\nConfigure your shooter (press Enter for defaults):\n")

    config = get_default_config()

    # Motor
    print("Motor Presets:")
    for k, m in MOTOR_PRESETS.items():
        tag = " [DEFAULT]" if k == '1' else ""
        print(f"  {k}. {m['name']} ({m['free_speed_rpm']} RPM){tag}")
    choice = prompt_choice("Select motor", MOTOR_PRESETS, '1')
    motor = MOTOR_PRESETS[choice]
    config.update({
        'motor_name': motor['name'],
        'motor_free_speed_rpm': motor['free_speed_rpm'],
        'motor_stall_torque_nm': motor['stall_torque_nm'],
        'motor_peak_torque_nm': motor['peak_torque_nm'],
        'motor_stall_current_a': motor['stall_current_a'],
    })
    print()

    # Angle
    angle_str = input("Shooter angle (Enter for optimal) [optimal]: ").strip()
    if angle_str == "" or angle_str.lower() == "optimal":
        config['selected_angle'] = None
    else:
        try:
            config['selected_angle'] = float(angle_str)
        except ValueError:
            config['selected_angle'] = None
    print()

    # Number of motors
    config['motor_num_motors'] = prompt_int("Number of motors", 2, 1, 4)
    config['motor_current_limit_a'] = prompt_float("Current limit (A)", motor['default_current_limit'], 10, 100)
    config['idle_speed_rpm'] = prompt_float("Idle speed (RPM)", 500, 0, 3000)
    print()

    # Gear ratio
    print("Gear Ratios: 1=1.0:1, 2=1.5:1, 3=2.0:1, 4=2.5:1, 5=3.0:1")
    gr_choice = prompt_choice("Select gear ratio", GEAR_RATIOS, '3')
    config['gear_ratio'] = GEAR_RATIOS[gr_choice]
    print()

    # Flywheel
    print("Flywheel Presets:")
    for k, f in FLYWHEEL_PRESETS.items():
        if f['moi_lb_in2']:
            tag = " [DEFAULT]" if k == '4' else ""
            print(f"  {k}. {f['name']} ({f['moi_lb_in2']:.1f} lb-in^2){tag}")
        else:
            print(f"  {k}. {f['name']}")
    fw_choice = prompt_choice("Select flywheel", FLYWHEEL_PRESETS, '4')
    if fw_choice == '5':
        config['flywheel_moi_lb_in2'] = prompt_float("Custom MOI (lb-in^2)", 10.0, 1.0, 50.0)
        config['flywheel_name'] = 'Custom'
    else:
        config['flywheel_moi_lb_in2'] = FLYWHEEL_PRESETS[fw_choice]['moi_lb_in2']
        config['flywheel_name'] = FLYWHEEL_PRESETS[fw_choice]['name']
    print()

    # Drag and slip
    config['drag_coefficient'] = prompt_float("Drag coefficient (0.3-0.7, foam ball ~0.5)", 0.50, 0.3, 0.7)
    config['slip_factor'] = prompt_float("Slip/compression factor (1.0-1.3)", 1.15, 1.0, 1.3)
    config['wheel_diameter_in'] = prompt_float("Wheel diameter (inches)", 4.0, 2.0, 8.0)
    print()

    # Advanced (optional)
    config['motor_efficiency'] = prompt_float("Motor efficiency", 0.85, 0.5, 1.0)

    return config


def update_motor(config: Dict) -> Dict:
    """Update motor configuration."""
    print("\nMotor Presets:")
    for k, m in MOTOR_PRESETS.items():
        current = " [CURRENT]" if m['name'] == config['motor_name'] else ""
        print(f"  {k}. {m['name']} ({m['free_speed_rpm']} RPM){current}")
    choice = prompt_choice("Select motor", MOTOR_PRESETS, '1')
    motor = MOTOR_PRESETS[choice]
    config.update({
        'motor_name': motor['name'],
        'motor_free_speed_rpm': motor['free_speed_rpm'],
        'motor_stall_torque_nm': motor['stall_torque_nm'],
        'motor_peak_torque_nm': motor['peak_torque_nm'],
        'motor_stall_current_a': motor['stall_current_a'],
    })
    config['motor_num_motors'] = prompt_int("Number of motors", config['motor_num_motors'], 1, 4)
    config['motor_current_limit_a'] = prompt_float("Current limit (A)", motor['default_current_limit'], 10, 100)
    return config


def update_angle(config: Dict) -> Dict:
    """Update angle configuration."""
    current = config['selected_angle'] if config['selected_angle'] else "optimal"
    angle_str = input(f"\nShooter angle (or 'optimal') [{current}]: ").strip()
    if angle_str == "":
        pass  # keep current
    elif angle_str.lower() == "optimal":
        config['selected_angle'] = None
    else:
        try:
            config['selected_angle'] = float(angle_str)
        except ValueError:
            print("Invalid angle, keeping current.")
    return config


def update_gear_ratio(config: Dict) -> Dict:
    """Update gear ratio."""
    print("\nGear Ratios:")
    for k, v in GEAR_RATIOS.items():
        current = " [CURRENT]" if v == config['gear_ratio'] else ""
        print(f"  {k}. {v}:1{current}")
    print("  6. Custom")
    choice = input(f"Select gear ratio [3]: ").strip()
    if choice == "":
        pass
    elif choice == '6':
        config['gear_ratio'] = prompt_float("Custom gear ratio", config['gear_ratio'], 0.5, 5.0)
    elif choice in GEAR_RATIOS:
        config['gear_ratio'] = GEAR_RATIOS[choice]
    return config


def update_flywheel(config: Dict) -> Dict:
    """Update flywheel configuration."""
    print("\nFlywheel Presets:")
    for k, f in FLYWHEEL_PRESETS.items():
        if f['moi_lb_in2']:
            current = " [CURRENT]" if f['moi_lb_in2'] == config['flywheel_moi_lb_in2'] else ""
            print(f"  {k}. {f['name']} ({f['moi_lb_in2']:.1f} lb-in^2){current}")
        else:
            print(f"  {k}. {f['name']}")
    choice = input("Select flywheel [4]: ").strip()
    if choice == "":
        pass
    elif choice == '5':
        config['flywheel_moi_lb_in2'] = prompt_float("Custom MOI (lb-in^2)", config['flywheel_moi_lb_in2'], 1.0, 50.0)
        config['flywheel_name'] = 'Custom'
    elif choice in FLYWHEEL_PRESETS:
        config['flywheel_moi_lb_in2'] = FLYWHEEL_PRESETS[choice]['moi_lb_in2']
        config['flywheel_name'] = FLYWHEEL_PRESETS[choice]['name']
    return config


def update_drag(config: Dict) -> Dict:
    """Update drag coefficient."""
    config['drag_coefficient'] = prompt_float("\nDrag coefficient (0.3-0.7)", config['drag_coefficient'], 0.3, 0.7)
    return config


def update_slip(config: Dict) -> Dict:
    """Update slip/compression factor."""
    config['slip_factor'] = prompt_float("\nSlip/compression factor (1.0-1.3)", config['slip_factor'], 1.0, 1.3)
    return config


def update_wheel(config: Dict) -> Dict:
    """Update wheel diameter."""
    config['wheel_diameter_in'] = prompt_float("\nWheel diameter (inches)", config['wheel_diameter_in'], 2.0, 8.0)
    return config

# =============================================================================
# SAVE/LOAD CONFIGURATION
# =============================================================================

def save_config(config: Dict):
    """Save configuration to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shooter_config_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {filename}")


def load_config() -> Optional[Dict]:
    """Load configuration from JSON file."""
    files = [f for f in os.listdir('.') if f.startswith('shooter_config_') and f.endswith('.json')]
    if not files:
        print("\nNo saved configurations found.")
        return None

    print("\nSaved configurations:")
    for i, f in enumerate(sorted(files, reverse=True)[:10], 1):
        print(f"  {i}. {f}")

    choice = input("Select file number (or filename): ").strip()
    try:
        idx = int(choice) - 1
        filename = sorted(files, reverse=True)[idx]
    except (ValueError, IndexError):
        filename = choice

    try:
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"Loaded {filename}")
        return config
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# =============================================================================
# DETAILED ANALYSIS VIEW
# =============================================================================

def show_detailed_analysis(config: Dict, results: Dict):
    """Show detailed analysis similar to original script."""
    if 'error' in results:
        print(f"\nError: {results['error']}")
        return

    print("\n" + "=" * 80)
    print("DETAILED SHOT TABLE")
    print("=" * 80)
    print(f"\n{'Distance':>10} | {'Exit Vel':>10} | {'Exit Vel':>10} | {'Wheel RPM':>10} | {'Entry Angle':>12}")
    print(f"{'(ft)':>10} | {'(m/s)':>10} | {'(ft/s)':>10} | {'':>10} | {'(deg)':>12}")
    print("-" * 65)

    for dist_ft in DISTANCES_FT:
        d = results['shot_data'][dist_ft]
        print(f"{dist_ft:>10} | {d['v_exit_ms']:>10.2f} | {d['v_exit_fps']:>10.1f} | {d['wheel_rpm']:>10.0f} | {d['entry_angle']:>12.1f}")

    print("\n" + "=" * 80)
    print("ANGLE COMPARISON (Top 5)")
    print("=" * 80)
    print(f"\n{'Rank':>4} | {'Angle':>6} | {'Score':>8} | {'V_range':>10} | {'Sens_V':>10} | {'Sens_A':>10}")
    print("-" * 60)
    for i, r in enumerate(results['all_angles'][:5], 1):
        print(f"{i:>4} | {r['angle']:>5} deg | {r['score']:>8.2f} | {r['v_range']:>10.2f} | {r['sens_v_mean']:>10.3f} | {r['sens_a_mean']:>10.3f}")

    print("\n" + "=" * 80)
    print("LOOKUP TABLE FOR ROBOT CODE")
    print("=" * 80)
    print("\n// Distance (ft) -> Wheel RPM")
    print("const SHOOTER_RPM_TABLE = {")
    for dist_ft in DISTANCES_FT:
        print(f"    {dist_ft}: {results['wheel_rpms'][dist_ft]:.0f},")
    print("};")

# =============================================================================
# GENERATE PLOTS
# =============================================================================

def generate_plots(config: Dict, results: Dict):
    """Generate analysis plots."""
    if 'error' in results:
        print(f"\nError: {results['error']}")
        return

    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: RPM requirements
    ax1 = axes[0, 0]
    rpms = [results['wheel_rpms'][d] for d in DISTANCES_FT]
    ax1.bar(DISTANCES_FT, rpms, color='steelblue', alpha=0.8, width=1.5)
    ax1.axhline(results['eff_free_speed'], color='red', linestyle='--',
                label=f"Eff. Free Speed ({results['eff_free_speed']:.0f} RPM)")
    ax1.set_xlabel('Distance (ft)')
    ax1.set_ylabel('Required Wheel RPM')
    ax1.set_title(f'Wheel RPM Requirements ({config["motor_name"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Consistency score by angle
    ax2 = axes[0, 1]
    angles_plot = [r['angle'] for r in results['all_angles']]
    scores = [r['score'] for r in results['all_angles']]
    ax2.bar(angles_plot, scores, color='steelblue', alpha=0.7)
    ax2.axvline(results['optimal_angle'], color='green', linestyle='--', linewidth=2,
                label=f"Optimal ({results['optimal_angle']} deg)")
    if results['selected_angle'] != results['optimal_angle']:
        ax2.axvline(results['selected_angle'], color='orange', linestyle='--', linewidth=2,
                    label=f"Selected ({results['selected_angle']} deg)")
    ax2.set_xlabel('Shooter Angle (deg)')
    ax2.set_ylabel('Consistency Score (lower = better)')
    ax2.set_title('Consistency by Angle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spin-up time vs gear ratio
    ax3 = axes[1, 0]
    moi_kg_m2 = config['flywheel_moi_lb_in2'] / (1 / 0.453592 / (0.0254**2))
    gear_ratios = np.linspace(1.0, 3.5, 20)
    spinup_8 = [calc_spinup_time(results['wheel_rpms'][8], r, moi_kg_m2,
                                  config['idle_speed_rpm'], config) * 1000 for r in gear_ratios]
    spinup_20 = [calc_spinup_time(results['wheel_rpms'][20], r, moi_kg_m2,
                                   config['idle_speed_rpm'], config) * 1000 for r in gear_ratios]
    ax3.plot(gear_ratios, spinup_8, 'b-', linewidth=2, label='8 ft shot')
    ax3.plot(gear_ratios, spinup_20, 'r-', linewidth=2, label='20 ft shot')
    ax3.axvline(config['gear_ratio'], color='green', linestyle='--', label=f"Current ({config['gear_ratio']}:1)")
    ax3.axhline(500, color='gray', linestyle=':', alpha=0.7, label='500 ms target')
    ax3.set_xlabel('Gear Ratio')
    ax3.set_ylabel('Spin-up Time (ms)')
    ax3.set_title('Spin-up Time vs Gear Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Exit velocity vs distance
    ax4 = axes[1, 1]
    vels = [results['shot_data'][d]['v_exit_fps'] for d in DISTANCES_FT]
    ax4.plot(DISTANCES_FT, vels, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Distance (ft)')
    ax4.set_ylabel('Exit Velocity (ft/s)')
    ax4.set_title(f'Exit Velocity at {results["selected_angle"]} deg (with drag)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'shooter_tuning_plots.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {filename}")
    plt.close()

# =============================================================================
# MAIN TUNING LOOP
# =============================================================================

def show_menu():
    """Display tuning menu."""
    print("\n" + "=" * 80)
    print("TUNING MENU")
    print("=" * 80)
    print("  1. Change shooter angle")
    print("  2. Change motor type/count")
    print("  3. Change gearbox ratio")
    print("  4. Change flywheel configuration")
    print("  5. Change drag coefficient")
    print("  6. Change slip/compression factor")
    print("  7. Change wheel diameter")
    print("  8. View detailed analysis")
    print("  9. Generate plots")
    print("  s. Save configuration")
    print("  l. Load configuration")
    print("  r. Start over (new configuration)")
    print("  q. Quit")
    print()


def main():
    """Main tuning loop."""
    config = get_initial_config()

    while True:
        print("\nRunning analysis...")
        results = run_analysis(config)
        display_results(config, results)
        show_menu()

        choice = input("Select option: ").strip().lower()

        if choice == '1':
            config = update_angle(config)
        elif choice == '2':
            config = update_motor(config)
        elif choice == '3':
            config = update_gear_ratio(config)
        elif choice == '4':
            config = update_flywheel(config)
        elif choice == '5':
            config = update_drag(config)
        elif choice == '6':
            config = update_slip(config)
        elif choice == '7':
            config = update_wheel(config)
        elif choice == '8':
            show_detailed_analysis(config, results)
            input("\nPress Enter to continue...")
        elif choice == '9':
            generate_plots(config, results)
            input("\nPress Enter to continue...")
        elif choice == 's':
            save_config(config)
        elif choice == 'l':
            loaded = load_config()
            if loaded:
                config = loaded
        elif choice == 'r':
            config = get_initial_config()
        elif choice == 'q':
            save_prompt = input("Save configuration before quitting? [y/N]: ").strip().lower()
            if save_prompt == 'y':
                save_config(config)
            print("Goodbye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
