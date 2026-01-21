import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# =============================================================================
# MOTOR PRESETS DICTIONARY
# =============================================================================

MOTOR_PRESETS = {
    '1': {
        'name': 'Kraken X44',
        'free_speed_rpm': 7530,
        'stall_torque_nm': 4.05,
        'peak_torque_nm': 2.0,
        'stall_current_a': 275,
        'default_current_limit': 40,
    },
    '2': {
        'name': 'Kraken X60',
        'free_speed_rpm': 5800,
        'stall_torque_nm': 9.37,
        'peak_torque_nm': 4.7,
        'stall_current_a': 483,
        'default_current_limit': 60,
    },
    '3': {
        'name': 'Falcon 500',
        'free_speed_rpm': 6380,
        'stall_torque_nm': 4.69,
        'peak_torque_nm': 2.35,
        'stall_current_a': 257,
        'default_current_limit': 40,
    },
    '4': {
        'name': 'NEO',
        'free_speed_rpm': 5676,
        'stall_torque_nm': 2.6,
        'peak_torque_nm': 1.3,
        'stall_current_a': 105,
        'default_current_limit': 40,
    },
    '5': {
        'name': 'NEO Vortex',
        'free_speed_rpm': 6784,
        'stall_torque_nm': 3.6,
        'peak_torque_nm': 1.8,
        'stall_current_a': 211,
        'default_current_limit': 40,
    },
}

# =============================================================================
# FLYWHEEL CONFIGURATION PRESETS
# =============================================================================

FLYWHEEL_PRESETS = {
    '1': {
        'name': 'Base (Stealth + AM insert)',
        'description': '4" Stealth wheels with AndyMark flywheel inserts',
        'moi_lb_in2': 4.68,  # Calculated from script
    },
    '2': {
        'name': '+ WCP Flywheels (2x)',
        'description': 'Base + 2x WCP Stainless Steel Flywheels',
        'moi_lb_in2': 10.08,  # Base + 2*2.7
    },
    '3': {
        'name': '+ Dual Stealth per side',
        'description': 'Dual stacked Stealth wheels per side',
        'moi_lb_in2': 9.36,  # Base * 2
    },
    '4': {
        'name': 'WCP + Dual Stealth (Recommended)',
        'description': 'WCP Flywheels + Dual Stealth for best balance',
        'moi_lb_in2': 14.76,  # Base * 2 + 2*2.7
    },
}

# =============================================================================
# GEAR RATIO PRESETS
# =============================================================================

GEAR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]

# =============================================================================
# INTERACTIVE CONFIGURATION
# =============================================================================

def get_user_configuration():
    """Prompt user for configuration options interactively."""
    print("=" * 80)
    print("FRC 2026 SHOOTER CONFIGURATION")
    print("=" * 80)
    print()

    # Motor selection
    print("Select Motor Preset:")
    for key, motor in MOTOR_PRESETS.items():
        default_tag = " [DEFAULT]" if key == '1' else ""
        print(f"  {key}. {motor['name']} ({motor['free_speed_rpm']} RPM, {motor['stall_torque_nm']} N·m stall){default_tag}")
    print()

    while True:
        choice = input("Enter motor choice [1-5] (default: 1): ").strip()
        if choice == "":
            choice = '1'
        if choice in MOTOR_PRESETS:
            selected_motor = MOTOR_PRESETS[choice]
            break
        print("Invalid choice. Please enter 1-5.")

    print()

    # Shooter angle
    while True:
        angle_input = input("Shooter angle in degrees (leave empty for optimal calculated angle) [65]: ").strip()
        if angle_input == "":
            selected_angle = 65
            break
        try:
            selected_angle = float(angle_input)
            if 40 <= selected_angle <= 80:
                break
            print("Angle should be between 40 and 80 degrees.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print()

    # Number of motors
    while True:
        num_input = input("Number of motors [2]: ").strip()
        if num_input == "":
            num_motors = 2
            break
        try:
            num_motors = int(num_input)
            if 1 <= num_motors <= 4:
                break
            print("Number of motors should be between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print()

    # Current limit
    default_current = selected_motor['default_current_limit']
    while True:
        current_input = input(f"Current limit per motor in Amps [{default_current}]: ").strip()
        if current_input == "":
            current_limit = default_current
            break
        try:
            current_limit = float(current_input)
            if 10 <= current_limit <= 100:
                break
            print("Current limit should be between 10 and 100 Amps.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print()

    # Idle speed
    while True:
        idle_input = input("Idle speed in RPM [500]: ").strip()
        if idle_input == "":
            idle_speed = 500
            break
        try:
            idle_speed = float(idle_input)
            if 0 <= idle_speed <= 3000:
                break
            print("Idle speed should be between 0 and 3000 RPM.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print()

    # Efficiency
    while True:
        eff_input = input("Motor efficiency [0.85]: ").strip()
        if eff_input == "":
            efficiency = 0.85
            break
        try:
            efficiency = float(eff_input)
            if 0.5 <= efficiency <= 1.0:
                break
            print("Efficiency should be between 0.5 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print()
    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Motor: {selected_motor['name']} (x{num_motors})")
    print(f"Angle: {selected_angle}°" + (" (user-specified)" if selected_angle != 65 else ""))
    print(f"Current Limit: {current_limit}A per motor")
    print(f"Idle Speed: {idle_speed} RPM")
    print(f"Efficiency: {efficiency}")
    print()

    # Confirmation
    confirm = input("Proceed with analysis? [Y/n]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("Configuration cancelled.")
        exit(0)

    print()

    return {
        'motor_name': selected_motor['name'],
        'motor_free_speed_rpm': selected_motor['free_speed_rpm'],
        'motor_stall_torque_nm': selected_motor['stall_torque_nm'],
        'motor_peak_torque_nm': selected_motor['peak_torque_nm'],
        'motor_stall_current_a': selected_motor['stall_current_a'],
        'motor_num_motors': num_motors,
        'motor_current_limit_a': current_limit,
        'motor_efficiency': efficiency,
        'selected_angle': selected_angle if selected_angle != 65 else None,  # None means use optimal
        'idle_speed_rpm': idle_speed,
    }

# Get user configuration
config = get_user_configuration()

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# Physics
G = 9.81  # m/s²

# Game piece - 2026 FUEL (Dense Polyurethane Foam)
BALL_DIAMETER_M = 5.9 * 0.0254  # 5.9 inches to meters
BALL_RADIUS_M = BALL_DIAMETER_M / 2
BALL_MASS_KG = 0.5 * 0.453592  # 0.5 lbs to kg
BALL_CROSS_SECTION = np.pi * BALL_RADIUS_M**2  # m²

# Air resistance parameters (from drag analysis)
AIR_DENSITY = 1.2  # kg/m³ (indoor)
DRAG_COEFFICIENT = 0.50  # Sphere with foam texture
DRAG_CONSTANT = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * BALL_CROSS_SECTION  # k in F_d = k*v²

# Target geometry
RIM_HEIGHT_M = 72 * 0.0254  # 72 inches to meters
HUB_DIAMETER_M = 40 * 0.0254  # 40 inches to meters
HUB_RADIUS_M = HUB_DIAMETER_M / 2

# Shooter geometry
EXIT_HEIGHT_M = 20 * 0.0254  # 20 inches to meters
DELTA_Y = RIM_HEIGHT_M - EXIT_HEIGHT_M  # Height to clear

# Shooting distances
MIN_DISTANCE_FT = 8
MAX_DISTANCE_FT = 20
DISTANCES_FT = [8, 10, 12, 14, 16, 18, 20]
DISTANCES_M = [d * 0.3048 for d in DISTANCES_FT]

# =============================================================================
# APPLY USER CONFIGURATION
# =============================================================================
SELECTED_ANGLE = config['selected_angle']
MOTOR_NAME = config['motor_name']
MOTOR_FREE_SPEED_RPM = config['motor_free_speed_rpm']
MOTOR_STALL_TORQUE_NM = config['motor_stall_torque_nm']
MOTOR_PEAK_TORQUE_NM = config['motor_peak_torque_nm']
MOTOR_STALL_CURRENT_A = config['motor_stall_current_a']
MOTOR_NUM_MOTORS = config['motor_num_motors']
MOTOR_CURRENT_LIMIT_A = config['motor_current_limit_a']
MOTOR_EFFICIENCY = config['motor_efficiency']
# Derived motor constants (calculated from user inputs)
MOTOR_FREE_SPEED_RADS = MOTOR_FREE_SPEED_RPM * 2 * np.pi / 60
MOTOR_KT = MOTOR_STALL_TORQUE_NM / MOTOR_STALL_CURRENT_A  # Torque constant (N·m/A)
MOTOR_TORQUE_AT_CURRENT_LIMIT = MOTOR_CURRENT_LIMIT_A * MOTOR_KT  # Torque at current limit
# Use the LESSER of peak torque or current-limited torque
MOTOR_USABLE_TORQUE = min(MOTOR_PEAK_TORQUE_NM, MOTOR_TORQUE_AT_CURRENT_LIMIT)

# =============================================================================
# IDLE SPEED CONFIGURATION
# =============================================================================
IDLE_SPEED_RPM = config['idle_speed_rpm']

# Unit conversions
def m_to_ft(m): return m / 0.3048
def ft_to_m(ft): return ft * 0.3048
def ms_to_fps(ms): return ms / 0.3048
def fps_to_ms(fps): return fps * 0.3048

print("=" * 80)
print("FRC 2026 SHOOTER ANALYSIS WITH AIR RESISTANCE")
print("=" * 80)

print("\n--- Game Piece Parameters ---")
print(f"  Diameter: {BALL_DIAMETER_M*1000:.1f} mm ({BALL_DIAMETER_M/0.0254:.1f} in)")
print(f"  Mass: {BALL_MASS_KG*1000:.1f} g ({BALL_MASS_KG/0.453592:.2f} lbs)")
print(f"  Cross-sectional area: {BALL_CROSS_SECTION*10000:.2f} cm²")

print("\n--- Air Resistance Model ---")
print(f"  Drag coefficient (Cd): {DRAG_COEFFICIENT}")
print(f"  Air density (ρ): {AIR_DENSITY} kg/m³")
print(f"  Drag constant (k): {DRAG_CONSTANT:.5f} kg/m")
print(f"  Drag force: F_d = {DRAG_CONSTANT:.5f} × v² [N]")
print(f"  Terminal velocity: {np.sqrt(BALL_MASS_KG * G / DRAG_CONSTANT):.1f} m/s ({ms_to_fps(np.sqrt(BALL_MASS_KG * G / DRAG_CONSTANT)):.1f} ft/s)")

print("\n--- Target Geometry ---")
print(f"  Rim height: {RIM_HEIGHT_M/0.0254:.0f} in ({RIM_HEIGHT_M:.3f} m)")
print(f"  Hub diameter: {HUB_DIAMETER_M/0.0254:.0f} in ({HUB_DIAMETER_M:.3f} m)")
print(f"  Exit height: {EXIT_HEIGHT_M/0.0254:.0f} in ({EXIT_HEIGHT_M:.3f} m)")
print(f"  Height to clear: {DELTA_Y/0.0254:.1f} in ({DELTA_Y:.3f} m)")

# =============================================================================
# TRAJECTORY SIMULATION WITH DRAG
# =============================================================================

def simulate_trajectory_with_drag(v0: float, angle_deg: float, dt: float = 0.0001) -> Dict:
    """
    Simulate trajectory with quadratic air resistance.
    
    F_drag = -k * v² * (v_hat)  (opposes velocity direction)
    
    Returns dict with trajectory data and key metrics.
    """
    theta = np.radians(angle_deg)

    # Initial conditions
    x, y = 0.0, 0.0
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    # Storage
    xs, ys, ts = [x], [y], [0.0]
    vxs, vys = [vx], [vy]
    t = 0.0

    # Simulate until ball falls below exit height or goes too far
    max_time = 5.0
    while t < max_time and y >= -EXIT_HEIGHT_M and x < 30:
        # Current speed
        v = np.sqrt(vx**2 + vy**2)

        # Drag acceleration (opposes velocity)
        if v > 0:
            a_drag = DRAG_CONSTANT * v**2 / BALL_MASS_KG
            ax_drag = -a_drag * vx / v
            ay_drag = -a_drag * vy / v
        else:
            ax_drag, ay_drag = 0, 0

        # Total acceleration
        ax = ax_drag
        ay = -G + ay_drag

        # Update velocity
        vx += ax * dt
        vy += ay * dt

        # Update position
        x += vx * dt
        y += vy * dt
        t += dt

        # Store (subsample for memory)
        if len(ts) < 10000 or t - ts[-1] > 0.001:
            xs.append(x)
            ys.append(y)
            ts.append(t)
            vxs.append(vx)
            vys.append(vy)

    return {
        'x': np.array(xs),
        'y': np.array(ys),
        't': np.array(ts),
        'vx': np.array(vxs),
        'vy': np.array(vys),
        'v0': v0,
        'angle': angle_deg
    }


def find_rim_crossing(traj: Dict, target_height: float = DELTA_Y) -> Tuple[float, float, float, bool]:
    """
    Find where trajectory crosses the rim height (descending).
    Returns: (x_position, vy_at_crossing, entry_angle_deg, is_valid)
    """
    xs, ys, vxs, vys = traj['x'], traj['y'], traj['vx'], traj['vy']

    # Find descending crossing of target height
    for i in range(1, len(ys)):
        if ys[i-1] >= target_height and ys[i] < target_height and vys[i] < 0:
            # Linear interpolation
            frac = (target_height - ys[i-1]) / (ys[i] - ys[i-1])
            x_cross = xs[i-1] + frac * (xs[i] - xs[i-1])
            vy_cross = vys[i-1] + frac * (vys[i] - vys[i-1])
            vx_cross = vxs[i-1] + frac * (vxs[i] - vxs[i-1])
            entry_angle = np.degrees(np.arctan2(vy_cross, vx_cross))
            return x_cross, vy_cross, entry_angle, True

    return np.inf, 0, 0, False


def find_required_velocity_with_drag(target_distance_m: float, angle_deg: float,
                                      tolerance: float = 0.01) -> Tuple[float, Dict]:
    """
    Binary search to find exit velocity that lands at target distance with drag.
    """
    v_low, v_high = 1.0, 50.0

    for _ in range(50):  # Max iterations
        v_mid = (v_low + v_high) / 2
        traj = simulate_trajectory_with_drag(v_mid, angle_deg)
        x_land, _, _, valid = find_rim_crossing(traj)

        if not valid:
            v_low = v_mid
            continue

        if abs(x_land - target_distance_m) < tolerance:
            return v_mid, traj
        elif x_land < target_distance_m:
            v_low = v_mid
        else:
            v_high = v_mid

    return v_mid, traj


def find_required_velocity_no_drag(target_distance_m: float, angle_deg: float) -> float:
    """
    Analytical solution without drag for comparison.
    """
    theta = np.radians(angle_deg)
    x = target_distance_m
    y = DELTA_Y

    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    denominator = x * tan_theta - y
    if denominator <= 0:
        return np.inf

    v0_squared = (G * x**2) / (2 * cos_theta**2 * denominator)
    return np.sqrt(v0_squared) if v0_squared > 0 else np.inf


# =============================================================================
# MAIN ANALYSIS: FIND OPTIMAL ANGLE WITH DRAG
# =============================================================================

print("\n" + "=" * 80)
print("TRAJECTORY ANALYSIS WITH AIR RESISTANCE")
print("=" * 80)

# Test angles
angles = np.arange(50, 76, 1)
results = []

print("\nCalculating trajectories for each angle and distance...")

for angle in angles:
    angle_data = {
        'angle': angle,
        'velocities_drag': [],
        'velocities_no_drag': [],
        'entry_angles': [],
        'trajectories': [],
        'valid': True
    }

    for dist_m in DISTANCES_M:
        # With drag
        v_drag, traj = find_required_velocity_with_drag(dist_m, angle)
        x_land, _, entry_angle, valid = find_rim_crossing(traj)

        if not valid or v_drag > 40:
            angle_data['valid'] = False
            break

        # Without drag (for comparison)
        v_no_drag = find_required_velocity_no_drag(dist_m, angle)

        angle_data['velocities_drag'].append(v_drag)
        angle_data['velocities_no_drag'].append(v_no_drag)
        angle_data['entry_angles'].append(entry_angle)
        angle_data['trajectories'].append(traj)

    if angle_data['valid']:
        angle_data['velocities_drag'] = np.array(angle_data['velocities_drag'])
        angle_data['velocities_no_drag'] = np.array(angle_data['velocities_no_drag'])
        angle_data['entry_angles'] = np.array(angle_data['entry_angles'])
        angle_data['v_min'] = angle_data['velocities_drag'].min()
        angle_data['v_max'] = angle_data['velocities_drag'].max()
        angle_data['v_range'] = angle_data['v_max'] - angle_data['v_min']
        angle_data['drag_overhead'] = (angle_data['velocities_drag'] / angle_data['velocities_no_drag']).mean()
        results.append(angle_data)

print(f"\nValid angles found: {len(results)}")

# =============================================================================
# SENSITIVITY ANALYSIS WITH DRAG
# =============================================================================

def calc_sensitivity_with_drag(angle_deg: float, v0: float,
                                delta_v: float = 0.1, delta_angle: float = 0.5) -> Tuple[float, float]:
    """
    Calculate landing position sensitivity to velocity and angle errors with drag.
    """
    # Base case
    traj_base = simulate_trajectory_with_drag(v0, angle_deg)
    x_base, _, _, _ = find_rim_crossing(traj_base)

    # Velocity sensitivity
    traj_plus_v = simulate_trajectory_with_drag(v0 + delta_v, angle_deg)
    traj_minus_v = simulate_trajectory_with_drag(v0 - delta_v, angle_deg)
    x_plus_v, _, _, _ = find_rim_crossing(traj_plus_v)
    x_minus_v, _, _, _ = find_rim_crossing(traj_minus_v)
    dxdv = (x_plus_v - x_minus_v) / (2 * delta_v)

    # Angle sensitivity
    traj_plus_a = simulate_trajectory_with_drag(v0, angle_deg + delta_angle)
    traj_minus_a = simulate_trajectory_with_drag(v0, angle_deg - delta_angle)
    x_plus_a, _, _, _ = find_rim_crossing(traj_plus_a)
    x_minus_a, _, _, _ = find_rim_crossing(traj_minus_a)
    dxda = (x_plus_a - x_minus_a) / (2 * delta_angle)

    return abs(dxdv), abs(dxda)

print("\nCalculating sensitivities...")

for r in results:
    sens_v_list = []
    sens_a_list = []

    for i, dist_m in enumerate(DISTANCES_M):
        v0 = r['velocities_drag'][i]
        dxdv, dxda = calc_sensitivity_with_drag(r['angle'], v0)
        sens_v_list.append(dxdv)
        sens_a_list.append(dxda)

    r['sens_v_mean'] = np.mean(sens_v_list)
    r['sens_a_mean'] = np.mean(sens_a_list)

    # Composite consistency score (lower = better)
    # Weight velocity sensitivity more heavily
    r['score'] = (
        r['sens_v_mean'] * 2.0 +
        r['sens_a_mean'] * 1.0 +
        r['v_range'] / 5.0
    )

# Sort by score
results_sorted = sorted(results, key=lambda x: x['score'])

# =============================================================================
# PRINT RESULTS TABLE
# =============================================================================

print("\n" + "-" * 110)
print(f"{'Angle':>6} | {'V_min':>8} | {'V_max':>8} | {'V_range':>8} | {'Drag OH':>8} | {'Sens_V':>8} | {'Sens_A':>8} | {'Score':>8}")
print(f"{'(deg)':>6} | {'(m/s)':>8} | {'(m/s)':>8} | {'(m/s)':>8} | {'(ratio)':>8} | {'(m/m/s)':>8} | {'(m/deg)':>8} | {'':>8}")
print("-" * 110)

for r in results:
    print(f"{r['angle']:>6} | {r['v_min']:>8.2f} | {r['v_max']:>8.2f} | {r['v_range']:>8.2f} | "
          f"{r['drag_overhead']:>8.2f} | {r['sens_v_mean']:>8.3f} | {r['sens_a_mean']:>8.3f} | {r['score']:>8.2f}")

print("\n" + "=" * 80)
print("TOP 5 ANGLES BY CONSISTENCY SCORE (lower = better)")
print("=" * 80)

print(f"\n{'Rank':>4} | {'Angle':>6} | {'Score':>8} | {'V_range':>10} | {'Sens_V':>10} | {'Sens_A':>10}")
print("-" * 65)
for i, r in enumerate(results_sorted[:5]):
    print(f"{i+1:>4} | {r['angle']:>6}° | {r['score']:>8.2f} | {r['v_range']:>10.2f} | {r['sens_v_mean']:>10.3f} | {r['sens_a_mean']:>10.3f}")

optimal = results_sorted[0]

# Determine which angle to use for detailed analysis
if SELECTED_ANGLE is not None:
    # Find the results for the user-specified angle
    selected_results = [r for r in results if r['angle'] == SELECTED_ANGLE]
    if selected_results:
        selected = selected_results[0]
        print(f"\n*** USING USER-SPECIFIED ANGLE: {SELECTED_ANGLE}° ***")
    else:
        print(f"\n⚠️  WARNING: Angle {SELECTED_ANGLE}° not found in valid results!")
        print(f"    Valid angles: {[r['angle'] for r in results]}")
        print(f"    Falling back to optimal angle: {optimal['angle']}°")
        selected = optimal
else:
    selected = optimal

print("\n" + "=" * 80)
print(f"OPTIMAL ANGLE: {optimal['angle']}° (WITH AIR RESISTANCE)")
if SELECTED_ANGLE is not None and selected['angle'] != optimal['angle']:
    print(f"SELECTED ANGLE: {selected['angle']}° (USER-SPECIFIED)")
print("=" * 80)

# =============================================================================
# DETAILED SHOT TABLE
# =============================================================================

print("\n" + "=" * 80)
print(f"DETAILED SHOT TABLE FOR {selected['angle']}° (WITH DRAG)")
if selected['angle'] != optimal['angle']:
    print(f"(Note: Optimal angle is {optimal['angle']}°)")
print("=" * 80)

print(f"\n{'Distance':>10} | {'No-Drag':>10} | {'With-Drag':>10} | {'Drag OH':>10} | {'Entry Angle':>12} | {'With-Drag':>10}")
print(f"{'(ft)':>10} | {'(m/s)':>10} | {'(m/s)':>10} | {'(%)':>10} | {'(deg)':>12} | {'(ft/s)':>10}")
print("-" * 75)

shot_data = {}
for i, dist_ft in enumerate(DISTANCES_FT):
    v_no_drag = selected['velocities_no_drag'][i]
    v_drag = selected['velocities_drag'][i]
    overhead = (v_drag / v_no_drag - 1) * 100
    entry = selected['entry_angles'][i]

    shot_data[dist_ft] = {
        'v_no_drag_ms': v_no_drag,
        'v_drag_ms': v_drag,
        'v_drag_fps': ms_to_fps(v_drag),
        'overhead_pct': overhead,
        'entry_angle': entry
    }

    print(f"{dist_ft:>10} | {v_no_drag:>10.2f} | {v_drag:>10.2f} | {overhead:>10.1f}% | {entry:>12.1f} | {ms_to_fps(v_drag):>10.1f}")

print("\n" + "=" * 80)
print("COMPARISON: NO-DRAG vs WITH-DRAG")
print("=" * 80)

print(f"""
Key findings with air resistance:

• Drag overhead at {selected['angle']}°: {(selected['drag_overhead']-1)*100:.1f}% additional velocity needed
• This means my original no-drag analysis UNDERESTIMATED required speeds by ~{(selected['drag_overhead']-1)*100:.0f}%

Velocity requirements WITH DRAG:
  • 8 ft shot:  {shot_data[8]['v_drag_ms']:.2f} m/s ({shot_data[8]['v_drag_fps']:.1f} ft/s)
  • 12 ft shot: {shot_data[12]['v_drag_ms']:.2f} m/s ({shot_data[12]['v_drag_fps']:.1f} ft/s)
  • 16 ft shot: {shot_data[16]['v_drag_ms']:.2f} m/s ({shot_data[16]['v_drag_fps']:.1f} ft/s)
  • 20 ft shot: {shot_data[20]['v_drag_ms']:.2f} m/s ({shot_data[20]['v_drag_fps']:.1f} ft/s)
""")

# =============================================================================
# WHEEL SPEED CALCULATIONS WITH DRAG
# =============================================================================

print("\n" + "=" * 80)
print("WHEEL SPEED REQUIREMENTS (WITH DRAG)")
print("=" * 80)

# Wheel parameters
WHEEL_DIAMETER_IN = 4.0
WHEEL_DIAMETER_M = WHEEL_DIAMETER_IN * 0.0254
SLIP_FACTOR = 1.15  # Wheel surface speed / ball exit speed

print(f"\nWheel diameter: {WHEEL_DIAMETER_IN}\"")
print(f"Slip factor: {SLIP_FACTOR}x (wheel faster than ball for low compression)")

print(f"\n{'Distance':>10} | {'Exit Vel':>10} | {'Exit Vel':>10} | {'Surface Spd':>12} | {'Wheel RPM':>10} | {'% of Free':>10}")
print(f"{'(ft)':>10} | {'(m/s)':>10} | {'(ft/s)':>10} | {'(m/s)':>12} | {'':>10} | {f'({MOTOR_NAME})':>10}")
print("-" * 75)

wheel_rpms = {}

for dist_ft in DISTANCES_FT:
    v_exit = shot_data[dist_ft]['v_drag_ms']
    v_exit_fps = shot_data[dist_ft]['v_drag_fps']
    surface_speed = v_exit * SLIP_FACTOR

    # RPM = (surface speed) / (π × diameter) × 60
    wheel_rpm = (surface_speed / (np.pi * WHEEL_DIAMETER_M)) * 60
    pct_free = (wheel_rpm / MOTOR_FREE_SPEED_RPM) * 100

    wheel_rpms[dist_ft] = wheel_rpm

    print(f"{dist_ft:>10} | {v_exit:>10.2f} | {v_exit_fps:>10.1f} | {surface_speed:>12.2f} | {wheel_rpm:>10.0f} | {pct_free:>10.1f}%")

max_rpm = max(wheel_rpms.values())
min_rpm = min(wheel_rpms.values())

print(f"\nRPM range: {min_rpm:.0f} - {max_rpm:.0f} RPM")
print(f"{MOTOR_NAME} free speed: {MOTOR_FREE_SPEED_RPM} RPM")
print(f"Headroom at max (direct drive): {(1 - max_rpm/MOTOR_FREE_SPEED_RPM)*100:.1f}%")

# =============================================================================
# MOTOR SPECIFICATIONS (Using user-configured values)
# =============================================================================

print("\n--- Motor Configuration ---")
print(f"  Motor: {MOTOR_NAME} (x{MOTOR_NUM_MOTORS})")
print(f"  Free speed: {MOTOR_FREE_SPEED_RPM} RPM")
print(f"  Stall torque: {MOTOR_STALL_TORQUE_NM} N·m (reference only)")
print(f"  Peak sustainable torque: {MOTOR_PEAK_TORQUE_NM} N·m (~50% of stall)")
print(f"  Current limit: {MOTOR_CURRENT_LIMIT_A}A → {MOTOR_TORQUE_AT_CURRENT_LIMIT:.2f} N·m")
print(f"  Usable torque per motor: {MOTOR_USABLE_TORQUE:.2f} N·m")
print(f"  Total usable torque: {MOTOR_USABLE_TORQUE * MOTOR_NUM_MOTORS:.2f} N·m")

# =============================================================================
# FLYWHEEL ANALYSIS WITH UPDATED VELOCITIES
# =============================================================================

print("\n" + "=" * 80)
print("FLYWHEEL ENERGY ANALYSIS (WITH DRAG-CORRECTED VELOCITIES)")
print("=" * 80)

# Stealth wheel + flywheel insert specs
STEALTH_WHEEL_MASS_KG = 0.3 * 0.453592
STEALTH_WHEEL_RADIUS_M = (WHEEL_DIAMETER_IN / 2) * 0.0254
FLYWHEEL_INSERT_MASS_KG = 0.8 * 0.453592
FLYWHEEL_INSERT_RADIUS_M = (3.3 / 2) * 0.0254

# Moment of inertia (single wheel assembly)
I_stealth = 0.5 * STEALTH_WHEEL_MASS_KG * STEALTH_WHEEL_RADIUS_M**2 * 1.3
I_flywheel = 0.5 * FLYWHEEL_INSERT_MASS_KG * FLYWHEEL_INSERT_RADIUS_M**2
I_single = I_stealth + I_flywheel
I_both = 2 * I_single

# Convert to lb·in²
KG_M2_TO_LB_IN2 = 1 / 0.453592 / (0.0254**2)
I_both_lbin2 = I_both * KG_M2_TO_LB_IN2

# WCP Stainless Flywheel
WCP_FLYWHEEL_MOI_LBIN2 = 2.7
WCP_FLYWHEEL_MOI_KGM2 = WCP_FLYWHEEL_MOI_LBIN2 / KG_M2_TO_LB_IN2

print("\nBase configuration (4\" Stealth + AM flywheel insert, both wheels):")
print(f"  Total MOI: {I_both_lbin2:.2f} lb·in² ({I_both*1e4:.2f} kg·cm²)")

def calc_speed_drop(initial_rpm, exit_velocity_ms, I_system, efficiency=0.70):
    """Calculate percentage speed drop after launching a ball."""
    omega_initial = initial_rpm * 2 * np.pi / 60
    E_flywheel = 0.5 * I_system * omega_initial**2
    E_ball = 0.5 * BALL_MASS_KG * exit_velocity_ms**2
    E_taken = E_ball / efficiency
    E_new = E_flywheel - E_taken
    if E_new <= 0:
        return 100.0
    omega_final = np.sqrt(2 * E_new / I_system)
    return ((omega_initial - omega_final) / omega_initial) * 100

print(f"\n{'Distance':>10} | {'Exit Vel':>10} | {'Wheel RPM':>10} | {'Speed Drop':>12} | {'Status':>15}")
print(f"{'(ft)':>10} | {'(m/s)':>10} | {'':>10} | {'(%)':>12} | {'':>15}")
print("-" * 65)

for dist_ft in [8, 12, 16, 20]:
    v_exit = shot_data[dist_ft]['v_drag_ms']
    rpm = wheel_rpms[dist_ft]
    drop = calc_speed_drop(rpm, v_exit, I_both)
    status = "⚠️ TOO HIGH" if drop > 20 else "✓ OK" if drop < 15 else "⚠️ MARGINAL"
    print(f"{dist_ft:>10} | {v_exit:>10.2f} | {rpm:>10.0f} | {drop:>12.1f}% | {status:>15}")

print("\n" + "=" * 80)
print("⚠️  CRITICAL: FLYWHEEL MASS UPGRADE REQUIRED")
print("=" * 80)

print("""
The speed drop is VERY HIGH with base configuration due to:
1. Higher exit velocities needed (drag compensation)
2. Ball kinetic energy scales with v² - higher speeds = much more energy per shot

RECOMMENDED UPGRADES:
""")

# Calculate for different configurations
configs = [
    ("Base (Stealth + AM insert)", I_both, I_both_lbin2),
    ("+ WCP Stainless Flywheels (2x)", I_both + 2*WCP_FLYWHEEL_MOI_KGM2, I_both_lbin2 + 2*WCP_FLYWHEEL_MOI_LBIN2),
    ("+ Dual Stealth per side", I_both * 2, I_both_lbin2 * 2),
    ("WCP + Dual Stealth (RECOMMENDED)", I_both * 2 + 2*WCP_FLYWHEEL_MOI_KGM2, I_both_lbin2 * 2 + 2*WCP_FLYWHEEL_MOI_LBIN2),
]

print(f"{'Configuration':>40} | {'MOI (lb·in²)':>12} | {'Drop @ 20ft':>12}")
print("-" * 70)

for name, I_kg, I_lb in configs:
    drop_20 = calc_speed_drop(wheel_rpms[20], shot_data[20]['v_drag_ms'], I_kg)
    status = "✓" if drop_20 < 15 else "⚠️" if drop_20 < 25 else "❌"
    print(f"{name:>40} | {I_lb:>12.1f} | {drop_20:>11.1f}% {status}")

# Use recommended config for further analysis
I_recommended = I_both * 2 + 2*WCP_FLYWHEEL_MOI_KGM2
I_recommended_lbin2 = I_both_lbin2 * 2 + 2*WCP_FLYWHEEL_MOI_LBIN2

print("\n→ RECOMMENDED: WCP Flywheels + Dual Stealth Wheels")
print(f"  Total MOI: {I_recommended_lbin2:.1f} lb·in²")
print(f"  Speed drop at 20ft: {calc_speed_drop(wheel_rpms[20], shot_data[20]['v_drag_ms'], I_recommended):.1f}%")

# =============================================================================
# SPIN-UP TIME ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("SPIN-UP TIME ANALYSIS (WITH RECOMMENDED FLYWHEEL MASS)")
print("=" * 80)

def calc_spinup_time(target_rpm, gear_ratio, I_system):
    """
    Calculate time to reach target RPM using PEAK torque curve (not stall).
    Starts from IDLE_SPEED_RPM and accelerates to target.
    Uses conservative motor parameters for sustainable operation.
    """
    target_rad_s = target_rpm * 2 * np.pi / 60
    motor_target_rad_s = target_rad_s * gear_ratio
    J_motor = I_system / (gear_ratio ** 2)

    dt = 0.0001
    omega = (IDLE_SPEED_RPM * 2 * np.pi / 60) * gear_ratio  # Start from idle
    t = 0

    while omega < motor_target_rad_s and t < 5:
        # Calculate torque from motor curve (linear from stall to free speed)
        # But cap at the usable torque (considering current limit)
        speed_fraction = omega / MOTOR_FREE_SPEED_RADS
        tau_from_curve = MOTOR_STALL_TORQUE_NM * (1 - speed_fraction)

        # Apply all limits: peak torque and current limit
        tau_motor = min(tau_from_curve, MOTOR_USABLE_TORQUE)

        # Total torque from all motors
        tau_total = MOTOR_NUM_MOTORS * tau_motor * MOTOR_EFFICIENCY

        alpha = tau_total / J_motor
        omega += alpha * dt
        t += dt

    return t

print(f"\nUsing recommended flywheel mass: {I_recommended_lbin2:.1f} lb·in²")
print(f"{MOTOR_NUM_MOTORS}x {MOTOR_NAME} motors, {MOTOR_CURRENT_LIMIT_A}A limit")
if IDLE_SPEED_RPM > 0:
    print(f"Idle speed: {IDLE_SPEED_RPM} RPM (spin-up measured from idle)")
else:
    print("Idle speed: 0 RPM (spin-up measured from complete stop)")

print(f"\n{'Gear Ratio':>12} | {'Eff. Free Spd':>14} | {'8ft Spinup':>12} | {'20ft Spinup':>12} | {'Headroom':>10} | {'% Peak Torque':>14}")
print("-" * 95)

for ratio in [1.0, 1.5, 2.0, 2.5]:
    eff_free = MOTOR_FREE_SPEED_RPM / ratio
    t_8 = calc_spinup_time(wheel_rpms[8], ratio, I_recommended) * 1000
    t_20 = calc_spinup_time(wheel_rpms[20], ratio, I_recommended) * 1000
    headroom = (1 - max_rpm / eff_free) * 100

    # Calculate what % of motor's max torque we're using at operating point
    operating_speed_fraction = (max_rpm * ratio) / MOTOR_FREE_SPEED_RPM
    torque_at_operating = MOTOR_STALL_TORQUE_NM * (1 - operating_speed_fraction) if operating_speed_fraction < 1 else 0
    pct_of_peak = (MOTOR_USABLE_TORQUE / MOTOR_STALL_TORQUE_NM) * 100 if torque_at_operating > 0 else 0

    status = "✓" if headroom > 30 else "⚠️" if headroom > 10 else "❌"
    print(f"{ratio:>12.1f}:1 | {eff_free:>14.0f} | {t_8:>12.0f} ms | {t_20:>12.0f} ms | {headroom:>9.1f}% {status} | {pct_of_peak:>13.0f}%")

# =============================================================================
# GENERATE PLOTS
# =============================================================================

fig, axes = plt.subplots(3, 3, figsize=(18, 16))

# Plot 1: Trajectories with drag
ax1 = axes[0, 0]
for dist_ft in [8, 12, 16, 20]:
    idx = DISTANCES_FT.index(dist_ft)
    traj = selected['trajectories'][idx]
    ax1.plot(m_to_ft(traj['x']), m_to_ft(traj['y'] + EXIT_HEIGHT_M),
             label=f'{dist_ft} ft', linewidth=2)

ax1.axhline(m_to_ft(RIM_HEIGHT_M), color='gray', linestyle='--', alpha=0.7, label='Rim height')
ax1.fill_between([m_to_ft(DISTANCES_M[-1]) - m_to_ft(HUB_RADIUS_M),
                   m_to_ft(DISTANCES_M[-1]) + m_to_ft(HUB_RADIUS_M)],
                  [m_to_ft(RIM_HEIGHT_M), m_to_ft(RIM_HEIGHT_M)], [0, 0],
                  alpha=0.2, color='orange', label='Hub')
ax1.set_xlabel('Horizontal Distance (ft)')
ax1.set_ylabel('Height (ft)')
ax1.set_title(f'Trajectories at {selected["angle"]}° WITH Air Resistance')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 25)
ax1.set_ylim(0, 15)

# Plot 2: No-drag vs With-drag comparison
ax2 = axes[0, 1]
x_dists = DISTANCES_FT
v_no_drag = [selected['velocities_no_drag'][i] for i in range(len(DISTANCES_FT))]
v_drag = [selected['velocities_drag'][i] for i in range(len(DISTANCES_FT))]

width = 0.35
x = np.arange(len(x_dists))
ax2.bar(x - width/2, [ms_to_fps(v) for v in v_no_drag], width, label='No Drag', color='steelblue', alpha=0.7)
ax2.bar(x + width/2, [ms_to_fps(v) for v in v_drag], width, label='With Drag', color='coral', alpha=0.7)
ax2.set_xlabel('Distance (ft)')
ax2.set_ylabel('Required Exit Velocity (ft/s)')
ax2.set_title('Exit Velocity: No-Drag vs With-Drag')
ax2.set_xticks(x)
ax2.set_xticklabels(x_dists)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Wheel RPM requirements
ax3 = axes[0, 2]
rpms = [wheel_rpms[d] for d in DISTANCES_FT]
ax3.bar(DISTANCES_FT, rpms, color='steelblue', alpha=0.8, width=1.5)
ax3.axhline(MOTOR_FREE_SPEED_RPM, color='red', linestyle='--', label=f'{MOTOR_NAME} Free ({MOTOR_FREE_SPEED_RPM} RPM)')
ax3.axhline(MOTOR_FREE_SPEED_RPM/1.5, color='orange', linestyle='--', label=f'1.5:1 Eff. ({MOTOR_FREE_SPEED_RPM/1.5:.0f} RPM)')
ax3.axhline(MOTOR_FREE_SPEED_RPM/2, color='green', linestyle='--', label=f'2:1 Eff. ({MOTOR_FREE_SPEED_RPM/2:.0f} RPM)')
ax3.set_xlabel('Distance (ft)')
ax3.set_ylabel('Required Wheel RPM')
ax3.set_title(f'Wheel RPM Requirements ({MOTOR_NAME})')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Sensitivity analysis across angles
ax4 = axes[1, 0]
angles_plot = [r['angle'] for r in results]
sens_v = [r['sens_v_mean'] for r in results]
sens_a = [r['sens_a_mean'] for r in results]
ax4.plot(angles_plot, sens_v, 'b-', label='Velocity Sensitivity (m/m/s)', linewidth=2)
ax4.plot(angles_plot, sens_a, 'r-', label='Angle Sensitivity (m/deg)', linewidth=2)
ax4.axvline(optimal['angle'], color='green', linestyle='--', linewidth=2, label=f"Optimal ({optimal['angle']}°)")
ax4.set_xlabel('Shooter Angle (deg)')
ax4.set_ylabel('Sensitivity (landing error)')
ax4.set_title('Sensitivity Analysis (With Drag)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Speed drop comparison
ax5 = axes[1, 1]
configs_plot = ['Base', '+WCP FW', '+Dual Stealth', 'WCP+Dual\n(REC)']
mois = [I_both_lbin2, I_both_lbin2 + 2*WCP_FLYWHEEL_MOI_LBIN2,
        I_both_lbin2 * 2, I_recommended_lbin2]
drops = [calc_speed_drop(wheel_rpms[20], shot_data[20]['v_drag_ms'], I)
         for I in [I_both, I_both + 2*WCP_FLYWHEEL_MOI_KGM2, I_both * 2, I_recommended]]

colors = ['red' if d > 25 else 'orange' if d > 15 else 'green' for d in drops]
bars = ax5.bar(configs_plot, drops, color=colors, alpha=0.7)
ax5.axhline(15, color='darkgreen', linestyle='--', linewidth=2, label='Target (<15%)')
ax5.axhline(25, color='darkorange', linestyle='--', linewidth=2, label='Marginal (<25%)')
ax5.set_ylabel('Speed Drop Per Shot (%)')
ax5.set_title('Flywheel Configuration Comparison (20ft shot)')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_ylim(0, 50)

# Add MOI labels on bars
for bar, moi in zip(bars, mois):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{moi:.1f} lb·in²', ha='center', va='bottom', fontsize=9)

# Plot 6: Consistency score by angle
ax6 = axes[1, 2]
scores = [r['score'] for r in results]
ax6.bar(angles_plot, scores, color='steelblue', alpha=0.7)
ax6.axvline(optimal['angle'], color='green', linestyle='--', linewidth=2, label=f"Optimal ({optimal['angle']}°)")
ax6.set_xlabel('Shooter Angle (deg)')
ax6.set_ylabel('Consistency Score (lower = better)')
ax6.set_title('Overall Consistency by Angle')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Spin-up time vs gear ratio
ax7 = axes[2, 0]
gear_ratios = np.linspace(1.0, 3.0, 20)
spinup_8ft = [calc_spinup_time(wheel_rpms[8], r, I_recommended) * 1000 for r in gear_ratios]
spinup_20ft = [calc_spinup_time(wheel_rpms[20], r, I_recommended) * 1000 for r in gear_ratios]
ax7.plot(gear_ratios, spinup_8ft, 'b-', linewidth=2, label='8 ft shot')
ax7.plot(gear_ratios, spinup_20ft, 'r-', linewidth=2, label='20 ft shot')
ax7.axhline(500, color='gray', linestyle='--', alpha=0.7, label='500 ms target')
ax7.set_xlabel('Gear Ratio')
ax7.set_ylabel('Spin-up Time (ms)')
ax7.set_title(f'Spin-up Time vs Gear Ratio ({MOTOR_NAME})')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_ylim(0, max(spinup_20ft) * 1.1)

# Plot 8: Spin-up time vs distance for different gear ratios
ax8 = axes[2, 1]
for ratio in [1.0, 1.5, 2.0, 2.5]:
    spinup_times = [calc_spinup_time(wheel_rpms[d], ratio, I_recommended) * 1000 for d in DISTANCES_FT]
    ax8.plot(DISTANCES_FT, spinup_times, 'o-', linewidth=2, markersize=6, label=f'{ratio}:1')
ax8.axhline(500, color='gray', linestyle='--', alpha=0.7, label='500 ms target')
ax8.set_xlabel('Distance (ft)')
ax8.set_ylabel('Spin-up Time (ms)')
ax8.set_title('Spin-up Time vs Distance by Gear Ratio')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Spin-up curve (RPM vs time) for 20ft shot at 2:1 ratio
ax9 = axes[2, 2]
def calc_spinup_curve(target_rpm, gear_ratio, I_system):
    """Return time and RPM arrays for spin-up curve, starting from idle speed."""
    target_rad_s = target_rpm * 2 * np.pi / 60
    motor_target_rad_s = target_rad_s * gear_ratio
    J_motor = I_system / (gear_ratio ** 2)

    dt = 0.0001
    omega = (IDLE_SPEED_RPM * 2 * np.pi / 60) * gear_ratio  # Start from idle
    t = 0
    times, rpms = [0], [IDLE_SPEED_RPM]

    while omega < motor_target_rad_s and t < 2:
        speed_fraction = omega / MOTOR_FREE_SPEED_RADS
        tau_from_curve = MOTOR_STALL_TORQUE_NM * (1 - speed_fraction)
        tau_motor = min(tau_from_curve, MOTOR_USABLE_TORQUE)
        tau_total = MOTOR_NUM_MOTORS * tau_motor * MOTOR_EFFICIENCY
        alpha = tau_total / J_motor
        omega += alpha * dt
        t += dt
        if len(times) < 2000 or t - times[-1] > 0.005:
            times.append(t * 1000)  # Convert to ms
            rpms.append(omega / gear_ratio * 60 / (2 * np.pi))  # Wheel RPM
    return times, rpms

for dist_ft in [8, 12, 16, 20]:
    times, rpms = calc_spinup_curve(wheel_rpms[dist_ft], 2.0, I_recommended)
    ax9.plot(times, rpms, linewidth=2, label=f'{dist_ft} ft ({wheel_rpms[dist_ft]:.0f} RPM)')
ax9.set_xlabel('Time (ms)')
ax9.set_ylabel('Wheel RPM')
ax9.set_title('Spin-up Curves at 2:1 Ratio (Conservative Torque)')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.set_xlim(0, None)

plt.tight_layout()
plt.savefig('shooter_analysis_with_drag.png', dpi=150, bbox_inches='tight')
print("\n[Plot saved to shooter_analysis_with_drag.png]")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL SPECIFICATION SUMMARY (WITH AIR RESISTANCE)")
print("=" * 80)

angle_note = f" (optimal: {optimal['angle']}°)" if selected['angle'] != optimal['angle'] else ""
print(f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    RECOMMENDED SHOOTER CONFIGURATION                            ║
║                        (WITH AIR RESISTANCE CORRECTION)                         ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  GEOMETRY                                                                      ║
║    • Fixed angle: {selected['angle']}°{angle_note}                                                       ║
║    • Exit height: 20" from floor                                               ║
║    • Shooting range: 8-20 ft                                                   ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  EXIT VELOCITIES (CORRECTED FOR DRAG)                                          ║
║    • 8 ft shot:  {shot_data[8]['v_drag_ms']:.1f} m/s ({shot_data[8]['v_drag_fps']:.0f} ft/s) — was {ms_to_fps(selected['velocities_no_drag'][0]):.0f} ft/s without drag       ║
║    • 12 ft shot: {shot_data[12]['v_drag_ms']:.1f} m/s ({shot_data[12]['v_drag_fps']:.0f} ft/s) — was {ms_to_fps(selected['velocities_no_drag'][2]):.0f} ft/s without drag       ║
║    • 16 ft shot: {shot_data[16]['v_drag_ms']:.1f} m/s ({shot_data[16]['v_drag_fps']:.0f} ft/s) — was {ms_to_fps(selected['velocities_no_drag'][4]):.0f} ft/s without drag       ║
║    • 20 ft shot: {shot_data[20]['v_drag_ms']:.1f} m/s ({shot_data[20]['v_drag_fps']:.0f} ft/s) — was {ms_to_fps(selected['velocities_no_drag'][6]):.0f} ft/s without drag       ║
║    • Drag overhead: ~{(selected['drag_overhead']-1)*100:.0f}% additional velocity required                        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  WHEELS                                                                        ║
║    • Type: 4" AndyMark Stealth Wheels (DUAL STACKED per side)                  ║
║    • Durometer: 50A (Blue) or 60A (Black)                                      ║
║    • ADD: 2x WCP Stainless Flywheels (2.7 lb·in² each)                         ║
║    • Total MOI: ~{I_recommended_lbin2:.0f} lb·in² for <15% speed drop                              ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  MOTORS                                                                        ║
║    • Type: {MOTOR_NUM_MOTORS}x {MOTOR_NAME} (one per wheel shaft)                                 ║
║    • Gearing: 2:1 reduction recommended (faster spin-up, good headroom)        ║
║    • Current limit: {MOTOR_CURRENT_LIMIT_A}A per motor                                            ║
║    • Using {MOTOR_USABLE_TORQUE:.1f} N·m per motor (peak torque, limited by {MOTOR_CURRENT_LIMIT_A}A current limit)              ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  WHEEL SPEED REQUIREMENTS                                                      ║
║    • RPM range: {min_rpm:.0f} - {max_rpm:.0f} RPM                                              ║
║    • At 2:1 ratio: Effective free speed = {MOTOR_FREE_SPEED_RPM/2:.0f} RPM                        ║
║    • Headroom: {(1 - max_rpm/(MOTOR_FREE_SPEED_RPM/2))*100:.0f}% at max distance                                        ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  PERFORMANCE (with recommended config, conservative torque)                    ║
║    • Spin-up time (0 to 20ft speed): ~{calc_spinup_time(wheel_rpms[20], 2.0, I_recommended)*1000:.0f} ms                              ║
║    • Speed drop per shot: ~{calc_speed_drop(wheel_rpms[20], shot_data[20]['v_drag_ms'], I_recommended):.0f}%                                              ║
║    • Recovery between shots: ~100-150 ms                                       ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  DIFFERENTIAL CONTROL                                                          ║
║    • Base: Equal top/bottom speeds                                             ║
║    • Backspin: Top wheel 5-15% faster for steeper entry                        ║
║    • Use differential for fine-tuning, not primary distance control            ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  PARTS LIST                                                                    ║
║    • 4x 4" Stealth Wheel 50A or 60A                                            ║
║    • 2x WCP Stainless Flywheel 4"                                              ║
║    • {MOTOR_NUM_MOTORS}x {MOTOR_NAME} motors                                                      ║
║    • 2:1 belt reduction (HTD 5mm recommended)                                  ║
╚════════════════════════════════════════════════════════════════════════════════╝

⚠️  NOTE: Motor calculations use CONSERVATIVE (peak) torque values:
    • Using {MOTOR_USABLE_TORQUE:.1f} N·m per motor (not {MOTOR_STALL_TORQUE_NM:.1f} N·m stall torque)
    • Current limit: {MOTOR_CURRENT_LIMIT_A}A per motor
    • This ensures motors won't overheat during extended use
""")

# Save numerical data for reference
print("\n" + "=" * 80)
print("LOOKUP TABLE: WHEEL RPM vs DISTANCE")
print("=" * 80)
print("\nProgram this into your robot controller:")
print("\n// Distance (ft) -> Wheel RPM (with drag correction)")
print("const SHOOTER_RPM_TABLE = {")
for dist_ft in DISTANCES_FT:
    print(f"    {dist_ft}: {wheel_rpms[dist_ft]:.0f},")
print("};")
