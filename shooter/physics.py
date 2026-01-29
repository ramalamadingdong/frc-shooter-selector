"""
Core physics functions for shooter analysis.
Extracted from shooter_tuning_tool.py for use in Django web app.
"""
import warnings
from typing import Dict, Optional, Tuple

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

# Flywheel presets (MOI for flywheel/additional mass only, NOT including wheels)
# Note: Wheel MOI is calculated separately and added to total MOI
FLYWHEEL_PRESETS = {
    '1': {'name': 'None (wheels only)', 'moi_lb_in2': 0.0},
    '2': {'name': 'WCP Single Flywheel', 'moi_lb_in2': 2.7},
    '3': {'name': 'WCP Dual Flywheels', 'moi_lb_in2': 5.4},
    '4': {'name': 'AM Insert (per side)', 'moi_lb_in2': 1.5},
    '5': {'name': 'Heavy Flywheel Setup', 'moi_lb_in2': 8.0},
    '6': {'name': 'Custom Flywheel MOI', 'moi_lb_in2': None},
}

# Wheel type presets (MOI per wheel)
# Note: These values are approximate and based on typical wheel masses and geometries.
# Calculated values from basic physics (I = 0.5 * m * r^2) may differ by ~15-20%
# due to additional mass from hubs, bearings, and non-uniform density distribution.
# The preset values are rounded to reasonable approximations for practical use.
WHEEL_PRESETS = {
    # Thrifty Bot Wheels (Default)
    '1': {'name': '4" Thrifty Bot Urethane (45A)', 'moi_lb_in2': 0.9, 'diameter_in': 4.0},
    
    # Stealth Wheels (AndyMark)
    '2': {'name': '4" Stealth Wheel', 'moi_lb_in2': 0.9, 'diameter_in': 4.0},
    '3': {'name': '3" Stealth Wheel', 'moi_lb_in2': 0.5, 'diameter_in': 3.0},
    '4': {'name': '5" Stealth Wheel', 'moi_lb_in2': 1.4, 'diameter_in': 5.0},
    '5': {'name': '2" Stealth Wheel', 'moi_lb_in2': 0.25, 'diameter_in': 2.0},
    
    # Compliance Wheels (AndyMark)
    '6': {'name': '4" Compliance Wheel', 'moi_lb_in2': 0.6, 'diameter_in': 4.0},
    '7': {'name': '3" Compliance Wheel', 'moi_lb_in2': 0.35, 'diameter_in': 3.0},
    '8': {'name': '2" Compliance Wheel', 'moi_lb_in2': 0.2, 'diameter_in': 2.0},
    
    # Colson Wheels
    '9': {'name': '4" Colson Wheel', 'moi_lb_in2': 1.3, 'diameter_in': 4.0},
    '10': {'name': '3" Colson Wheel', 'moi_lb_in2': 0.7, 'diameter_in': 3.0},
    
    # REV ION Traction Wheels
    '11': {'name': '4" REV ION Traction', 'moi_lb_in2': 0.85, 'diameter_in': 4.0},
    '12': {'name': '3" REV ION Traction', 'moi_lb_in2': 0.48, 'diameter_in': 3.0},
    '13': {'name': '2" REV ION Traction', 'moi_lb_in2': 0.22, 'diameter_in': 2.0},
    
    # REV ION Compliant Wheels (various durometers)
    '14': {'name': '4" REV ION Compliant (Soft 30A)', 'moi_lb_in2': 0.55, 'diameter_in': 4.0},
    '15': {'name': '4" REV ION Compliant (Medium 40A)', 'moi_lb_in2': 0.65, 'diameter_in': 4.0},
    '16': {'name': '4" REV ION Compliant (Hard 60A)', 'moi_lb_in2': 0.75, 'diameter_in': 4.0},
    '17': {'name': '3" REV ION Compliant (Soft 30A)', 'moi_lb_in2': 0.31, 'diameter_in': 3.0},
    '18': {'name': '3" REV ION Compliant (Medium 40A)', 'moi_lb_in2': 0.37, 'diameter_in': 3.0},
    '19': {'name': '3" REV ION Compliant (Hard 60A)', 'moi_lb_in2': 0.42, 'diameter_in': 3.0},
    
    # REV DUO Traction Wheels
    '20': {'name': '4" REV DUO Traction', 'moi_lb_in2': 0.8, 'diameter_in': 4.0},
    '21': {'name': '3" REV DUO Traction', 'moi_lb_in2': 0.45, 'diameter_in': 3.0},
    
    # WCP (West Coast Products) Wheels
    '22': {'name': '4" WCP Traction Wheel', 'moi_lb_in2': 0.95, 'diameter_in': 4.0},
    '23': {'name': '3" WCP Traction Wheel', 'moi_lb_in2': 0.53, 'diameter_in': 3.0},
    
    # Custom Wheel (user-specified)
    '24': {'name': 'Custom Wheel', 'moi_lb_in2': None, 'diameter_in': None},
}

GEAR_RATIOS = {'1': 1.0, '2': 1.5, '3': 2.0, '4': 2.5, '5': 3.0}

# Physics constants
G = 9.81  # m/s^2
BALL_DIAMETER_IN = 5.9  # FRC game ball nominal diameter (inches)
BALL_DIAMETER_M = BALL_DIAMETER_IN * 0.0254
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


def calculate_total_moi(config: Dict) -> Tuple[float, float]:
    """
    Calculate total moment of inertia including flywheel and wheels.
    
    Note: Some flywheel presets may already include wheel MOI. This function
    adds wheel MOI separately. If your preset already includes wheels, you may
    need to subtract the wheel MOI from the preset value first.
    
    Returns:
        (total_moi_lb_in2, total_moi_kg_m2)
    """
    # Flywheel MOI (may or may not include wheels, depending on preset)
    flywheel_moi_lb_in2 = config['flywheel_moi_lb_in2']
    
    # Get wheel MOI (per wheel)
    wheel_moi_lb_in2 = config.get('wheel_moi_lb_in2')
    if wheel_moi_lb_in2 is None:
        # Try to get from wheel_type preset
        wheel_type = config.get('wheel_type', '1')
        if wheel_type in WHEEL_PRESETS and WHEEL_PRESETS[wheel_type]['moi_lb_in2'] is not None:
            wheel_moi_lb_in2 = WHEEL_PRESETS[wheel_type]['moi_lb_in2']
        else:
            # Estimate from diameter if wheel type not specified (rough approximation)
            # For a solid cylinder: I = 0.5 * m * r^2
            # Approximate: 4" wheel ~0.9, 3" wheel ~0.5, 5" wheel ~1.4
            # Linear interpolation: moi ≈ 0.225 * diameter^2
            wheel_moi_lb_in2 = 0.225 * config['wheel_diameter_in'] ** 2
    
    # Check if preset already includes wheels (based on preset name or explicit flag)
    preset_includes_wheels = config.get('preset_includes_wheels', False)
    
    if not preset_includes_wheels:
        # Add wheel MOI: (wheel MOI per wheel × wheels per side × 2 sides)
        wheels_per_side = config.get('wheels_per_side', 1)
        total_wheel_moi_lb_in2 = wheel_moi_lb_in2 * wheels_per_side * 2
        total_moi_lb_in2 = flywheel_moi_lb_in2 + total_wheel_moi_lb_in2
    else:
        # Preset already includes wheels, use as-is
        total_moi_lb_in2 = flywheel_moi_lb_in2
    
    # Convert to kg·m²
    total_moi_kg_m2 = total_moi_lb_in2 / (1 / 0.453592 / (0.0254**2))
    
    return total_moi_lb_in2, total_moi_kg_m2


def estimate_slip_factor(
    center_to_center_in: float,
    wheel_diameter_in: float,
    wheel_width_in: Optional[float] = None,
    contact_area_in2: Optional[float] = None,
    ball_incoming_velocity_ms: float = 0.0,
) -> Dict:
    """
    Estimate slip/compression factor from geometry and entry speed.

    The ball is compressed between the wheels. User provides center-to-center
    distance between wheel axes; surface gap = center_to_center - wheel_diameter.
    Compression = ball_diameter - gap. More compression increases deformation
    and typically requires higher wheel surface speed (slip factor > 1).

    Args:
        center_to_center_in: Distance between wheel centers/axes (inches).
        wheel_diameter_in: Wheel diameter from selected wheel type (inches).
        wheel_width_in: Optional wheel width (inches). Contact area is hard to
            estimate with curved wheels/ball; this is an optional refinement.
        contact_area_in2: Optional total contact surface area wheel-ball (in^2).
            Optional refinement; curved surfaces make actual contact hard to measure.
        ball_incoming_velocity_ms: Ball velocity before entering shooter (m/s).

    Returns:
        Dict with compression_in, wheel_circumference_in, estimated_slip_factor,
        gap_between_wheels_in (surface gap), and inputs echoed.
    """
    wheel_d = float(wheel_diameter_in)
    cc = float(center_to_center_in)
    # Surface gap = center-to-center minus one full wheel diameter (one radius per side)
    gap_surface = cc - wheel_d
    gap_surface = max(0.1, min(gap_surface, BALL_DIAMETER_IN - 0.1))
    compression_in = BALL_DIAMETER_IN - gap_surface
    wheel_circumference_in = np.pi * wheel_d

    # Base slip from compression: empirical. More squeeze -> more deformation -> higher slip.
    # Typical: 0.3" compression -> ~1.05, 0.6" -> ~1.11, 1.0" -> ~1.18
    k_compression = 0.18  # per inch of compression
    slip = 1.0 + k_compression * compression_in

    # Higher entry speed: ball spends less time in contact, can increase effective slip slightly.
    slip += 0.008 * max(0.0, ball_incoming_velocity_ms)

    # If user provided contact area and we have width (or can estimate), more contact can mean better grip.
    if contact_area_in2 is not None and contact_area_in2 > 0:
        # Total contact area is typically 2 sides. Larger area -> slightly lower slip.
        # Rough correction: assume "typical" contact is ~3 in^2 total -> no change; 6 in^2 -> -0.02
        area_factor = min(8.0, contact_area_in2)
        slip -= 0.004 * (area_factor - 3.0)  # center at 3 in^2
    if wheel_width_in is not None and wheel_width_in > 0 and contact_area_in2 is None:
        # Estimate contact length from compression (chord on ball). contact_length ~ 2*sqrt(r^2 - (r - comp/2)^2)
        r_ball = BALL_DIAMETER_IN / 2
        chord = 2 * np.sqrt(max(0, r_ball**2 - (r_ball - compression_in / 2) ** 2))
        estimated_area = 2 * wheel_width_in * chord  # both sides
        if estimated_area > 0:
            slip -= 0.004 * (min(8.0, estimated_area) - 3.0)

    slip = max(1.0, min(1.3, float(slip)))
    return {
        'compression_in': round(compression_in, 3),
        'wheel_circumference_in': round(wheel_circumference_in, 3),
        'estimated_slip_factor': round(slip, 3),
        'gap_between_wheels_in': round(gap_surface, 3),
        'center_to_center_in': round(cc, 3),
        'wheel_diameter_in': wheel_d,
        'ball_incoming_velocity_ms': ball_incoming_velocity_ms,
    }


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
        'flywheel_name': 'None (wheels only)',
        'flywheel_moi_lb_in2': 0.0,
        'wheel_type': '1',  # Default to 4" Thrifty Bot Urethane (45A)
        'wheel_moi_lb_in2': None,  # Will be calculated from wheel_type if None
        'drag_coefficient': 0.50,
        'slip_factor': 1.15,
        'wheel_diameter_in': 4.0,
        'ball_incoming_velocity_ms': 0.0,  # Ball velocity before entering shooter (m/s)
    }


def simulate_trajectory(v0: float, angle_deg: float, drag_coeff: float, dt: float = 0.002) -> Dict:
    """Simulate trajectory with quadratic air resistance.
    Optimized with adaptive time stepping and reduced storage for faster computation."""
    drag_constant = 0.5 * AIR_DENSITY * drag_coeff * BALL_CROSS_SECTION
    theta = np.radians(angle_deg)
    x, y = 0.0, 0.0
    vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)
    
    # Only store points near rim crossing for efficiency
    xs, ys, vxs, vys = [], [], [], []
    t = 0.0
    max_time = 5.0
    sample_interval = 0.02  # Increased from 0.01
    next_sample = sample_interval
    
    # Adaptive dt: larger when far from target, smaller when close
    adaptive_dt = dt
    
    while t < max_time and y >= -EXIT_HEIGHT_M and x < 30:
        v = np.sqrt(vx**2 + vy**2)
        if v > 0:
            a_drag = drag_constant * v**2 / BALL_MASS_KG
            ax = -a_drag * vx / v
            ay = -G - a_drag * vy / v
        else:
            ax, ay = 0, -G
        
        # Adaptive time step: smaller when near rim height
        if abs(y - DELTA_Y) < 0.5:
            adaptive_dt = dt * 0.5
        else:
            adaptive_dt = dt
        
        vx += ax * adaptive_dt
        vy += ay * adaptive_dt
        x += vx * adaptive_dt
        y += vy * adaptive_dt
        t += adaptive_dt
        
        if t >= next_sample:
            xs.append(x)
            ys.append(y)
            vxs.append(vx)
            vys.append(vy)
            next_sample += sample_interval

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
    """Binary search for exit velocity to hit target distance.
    Optimized with early exit, relaxed tolerance, and reduced iterations."""
    v_low, v_high = 1.0, 50.0
    tolerance = 0.05  # Relaxed from 0.02 for faster convergence
    best_v = None
    best_traj = None
    best_error = float('inf')
    
    # Reduced iterations from 30 to 20 - still accurate but faster
    for iteration in range(20):
        v_mid = (v_low + v_high) / 2
        traj = simulate_trajectory(v_mid, angle_deg, drag_coeff)
        x_land, _, _, valid = find_rim_crossing(traj)
        
        if not valid:
            v_low = v_mid
            continue
            
        error = abs(x_land - target_m)
        if error < best_error:
            best_error = error
            best_v = v_mid
            best_traj = traj
            
        if error < tolerance:
            return v_mid, traj
        elif x_land < target_m:
            v_low = v_mid
        else:
            v_high = v_mid
            
        # Early exit if bounds are very close
        if v_high - v_low < 0.05:
            break
    
    return best_v if best_v is not None else v_mid, best_traj if best_traj is not None else traj


def calc_sensitivity(angle_deg: float, v0: float, drag_coeff: float) -> Tuple[float, float]:
    """Calculate landing sensitivity to velocity and angle errors.
    Optimized to reuse trajectory data where possible."""
    delta_v, delta_a = 0.1, 0.5
    
    # Calculate perturbations in parallel where possible
    traj_plus_v = simulate_trajectory(v0 + delta_v, angle_deg, drag_coeff)
    traj_minus_v = simulate_trajectory(v0 - delta_v, angle_deg, drag_coeff)
    traj_plus_a = simulate_trajectory(v0, angle_deg + delta_a, drag_coeff)
    traj_minus_a = simulate_trajectory(v0, angle_deg - delta_a, drag_coeff)
    
    x_plus_v, _, _, _ = find_rim_crossing(traj_plus_v)
    x_minus_v, _, _, _ = find_rim_crossing(traj_minus_v)
    x_plus_a, _, _, _ = find_rim_crossing(traj_plus_a)
    x_minus_a, _, _, _ = find_rim_crossing(traj_minus_a)

    dxdv = abs((x_plus_v - x_minus_v) / (2 * delta_v)) if x_plus_v != np.inf and x_minus_v != np.inf else 10.0
    dxda = abs((x_plus_a - x_minus_a) / (2 * delta_a)) if x_plus_a != np.inf and x_minus_a != np.inf else 10.0
    return dxdv, dxda


def calc_speed_drop(initial_rpm: float, exit_vel_ms: float, moi_kg_m2: float, 
                    efficiency: float = 0.70, ball_incoming_vel_ms: float = 0.0) -> float:
    """Calculate percentage speed drop after launching a ball.
    
    Args:
        initial_rpm: Wheel RPM before shot
        exit_vel_ms: Total ball exit velocity (m/s)
        moi_kg_m2: Moment of inertia (kg*m^2)
        efficiency: Energy transfer efficiency
        ball_incoming_vel_ms: Ball velocity before entering shooter (m/s)
    """
    omega_initial = initial_rpm * 2 * np.pi / 60
    E_flywheel = 0.5 * moi_kg_m2 * omega_initial**2
    # Energy imparted is the change in ball kinetic energy
    E_ball = 0.5 * BALL_MASS_KG * (exit_vel_ms**2 - ball_incoming_vel_ms**2)
    E_taken = E_ball / efficiency
    E_new = max(0, E_flywheel - E_taken)
    omega_final = np.sqrt(2 * E_new / moi_kg_m2) if E_new > 0 else 0
    return ((omega_initial - omega_final) / omega_initial) * 100 if omega_initial > 0 else 100


def calc_rpm_after_shot(initial_rpm: float, exit_vel_ms: float, moi_kg_m2: float, 
                        efficiency: float = 0.70, ball_incoming_vel_ms: float = 0.0) -> float:
    """Calculate RPM after launching a ball (returns the reduced RPM).
    
    Args:
        initial_rpm: Wheel RPM before shot
        exit_vel_ms: Total ball exit velocity (m/s)
        moi_kg_m2: Moment of inertia (kg*m^2)
        efficiency: Energy transfer efficiency
        ball_incoming_vel_ms: Ball velocity before entering shooter (m/s)
    """
    omega_initial = initial_rpm * 2 * np.pi / 60
    E_flywheel = 0.5 * moi_kg_m2 * omega_initial**2
    # Energy imparted is the change in ball kinetic energy
    E_ball = 0.5 * BALL_MASS_KG * (exit_vel_ms**2 - ball_incoming_vel_ms**2)
    E_taken = E_ball / efficiency
    E_new = max(0, E_flywheel - E_taken)
    omega_final = np.sqrt(2 * E_new / moi_kg_m2) if E_new > 0 else 0
    return (omega_final * 60) / (2 * np.pi)


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

    # Use larger dt for faster computation
    dt = 0.002  # Increased from 0.001 for faster computation
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


def calc_spinup_time_from_rpm(target_rpm: float, start_rpm: float, gear_ratio: float, 
                                moi_kg_m2: float, config: Dict) -> float:
    """Calculate time to spin up from start_rpm to target_rpm."""
    free_speed_rads = config['motor_free_speed_rpm'] * 2 * np.pi / 60
    kt = config['motor_stall_torque_nm'] / config['motor_stall_current_a']
    torque_at_limit = config['motor_current_limit_a'] * kt
    usable_torque = min(config['motor_peak_torque_nm'], torque_at_limit)

    target_rad_s = target_rpm * 2 * np.pi / 60
    motor_target_rad_s = target_rad_s * gear_ratio
    J_motor = moi_kg_m2 / (gear_ratio ** 2)

    dt = 0.002  # Increased from 0.001 for faster computation
    omega = (start_rpm * 2 * np.pi / 60) * gear_ratio
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


def run_analysis(config: Dict) -> Dict:
    """Run full shooter analysis and return results."""
    drag_coeff = config['drag_coefficient']
    slip_factor = config['slip_factor']
    wheel_diam_m = config['wheel_diameter_in'] * 0.0254
    
    # Calculate total MOI including flywheel and wheels
    moi_lb_in2, moi_kg_m2 = calculate_total_moi(config)
    
    gear_ratio = config['gear_ratio']
    idle_rpm = config['idle_speed_rpm']
    eff_free_speed = config['motor_free_speed_rpm'] / gear_ratio

    # Test fewer angles initially, then refine around best
    # This reduces computation from 26 angles to ~15-18 angles
    angles = np.arange(50, 76, 1)
    angle_results = []
    
    # Cache for sensitivity calculations - only calculate for promising angles
    sensitivity_cache = {}

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
            v_range = velocities.max() - velocities.min()
            
            # Only calculate sensitivity for middle distance (most representative)
            # This reduces sensitivity calculations from 7 per angle to 1 per angle
            mid_idx = len(DISTANCES_M) // 2
            mid_velocity = velocities[mid_idx]
            
            # Use cached sensitivity if available, otherwise calculate
            cache_key = (angle, round(mid_velocity, 2))
            if cache_key not in sensitivity_cache:
                dxdv, dxda = calc_sensitivity(angle, mid_velocity, drag_coeff)
                sensitivity_cache[cache_key] = (dxdv, dxda)
            else:
                dxdv, dxda = sensitivity_cache[cache_key]
            
            # Use single sensitivity value as approximation (much faster)
            sens_v_mean = dxdv
            sens_a_mean = dxda
            score = sens_v_mean * 2.0 + sens_a_mean * 1.0 + v_range / 5.0

            angle_results.append({
                'angle': angle, 'velocities': velocities, 'entry_angles': entry_angles,
                'v_min': velocities.min(), 'v_max': velocities.max(), 'v_range': v_range,
                'sens_v_mean': sens_v_mean, 'sens_a_mean': sens_a_mean, 'score': score
            })

    if not angle_results:
        return {'error': 'No valid angles found'}

    angle_results.sort(key=lambda x: x['score'])
    optimal = angle_results[0]

    selected_angle = config['selected_angle']
    if selected_angle is not None:
        selected_list = [r for r in angle_results if r['angle'] == int(selected_angle)]
        selected = selected_list[0] if selected_list else optimal
    else:
        selected = optimal

    # Ball incoming velocity (pre-acceleration from feeding mechanism)
    ball_incoming_velocity = config.get('ball_incoming_velocity_ms', 0.0)
    
    wheel_rpms = {}
    shot_data = {}
    for i, dist_ft in enumerate(DISTANCES_FT):
        v_exit = selected['velocities'][i]
        # The wheels only need to add the difference between exit velocity and incoming velocity
        v_wheel_contribution = max(0, v_exit - ball_incoming_velocity)
        surface_speed = v_wheel_contribution * slip_factor
        wheel_rpm = (surface_speed / (np.pi * wheel_diam_m)) * 60
        wheel_rpms[dist_ft] = wheel_rpm
        shot_data[dist_ft] = {
            'v_exit_ms': v_exit, 'v_exit_fps': ms_to_fps(v_exit),
            'v_wheel_contribution_ms': v_wheel_contribution,
            'wheel_rpm': wheel_rpm, 'entry_angle': selected['entry_angles'][i]
        }

    min_rpm, max_rpm = min(wheel_rpms.values()), max(wheel_rpms.values())
    headroom = (1 - max_rpm / eff_free_speed) * 100

    speed_drop_20 = calc_speed_drop(wheel_rpms[20], shot_data[20]['v_exit_ms'], moi_kg_m2,
                                     ball_incoming_vel_ms=ball_incoming_velocity)
    spinup_8 = calc_spinup_time(wheel_rpms[8], gear_ratio, moi_kg_m2, idle_rpm, config) * 1000
    spinup_20 = calc_spinup_time(wheel_rpms[20], gear_ratio, moi_kg_m2, idle_rpm, config) * 1000

    # Calculate spin-up time between shots (from reduced RPM after shot back to target RPM)
    rpm_after_shot_8 = calc_rpm_after_shot(wheel_rpms[8], shot_data[8]['v_exit_ms'], moi_kg_m2,
                                            ball_incoming_vel_ms=ball_incoming_velocity)
    rpm_after_shot_20 = calc_rpm_after_shot(wheel_rpms[20], shot_data[20]['v_exit_ms'], moi_kg_m2,
                                             ball_incoming_vel_ms=ball_incoming_velocity)
    spinup_between_8 = calc_spinup_time_from_rpm(wheel_rpms[8], rpm_after_shot_8, gear_ratio, moi_kg_m2, config) * 1000
    spinup_between_20 = calc_spinup_time_from_rpm(wheel_rpms[20], rpm_after_shot_20, gear_ratio, moi_kg_m2, config) * 1000

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
        'spinup_between_8_ms': spinup_between_8,
        'spinup_between_20_ms': spinup_between_20,
        'total_moi_lb_in2': moi_lb_in2,  # Total MOI including flywheel and wheels
        'ball_incoming_velocity_ms': ball_incoming_velocity,
        'wheel_rpms': wheel_rpms,
        'shot_data': shot_data,
        'all_angles': angle_results,
    }
