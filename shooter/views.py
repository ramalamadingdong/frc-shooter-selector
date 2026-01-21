import json
import logging
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from shooter.physics import (
    get_default_config, run_analysis, MOTOR_PRESETS,
    FLYWHEEL_PRESETS, GEAR_RATIOS, WHEEL_PRESETS
)

logger = logging.getLogger(__name__)


def index(request):
    """Main page with configuration form."""
    default_config = get_default_config()
    context = {
        'default_config': default_config,
        'motor_presets': MOTOR_PRESETS,
        'flywheel_presets': FLYWHEEL_PRESETS,
        'wheel_presets': WHEEL_PRESETS,
        'gear_ratios': GEAR_RATIOS,
    }
    return render(request, 'shooter/index.html', context)


@csrf_exempt
def analyze(request):
    """API endpoint to run analysis."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        logger.info('Analysis request received')
        data = json.loads(request.body)
        config = {
            'motor_name': data.get('motor_name', 'Kraken X44'),
            'motor_free_speed_rpm': float(data.get('motor_free_speed_rpm', 7530)),
            'motor_stall_torque_nm': float(data.get('motor_stall_torque_nm', 4.05)),
            'motor_peak_torque_nm': float(data.get('motor_peak_torque_nm', 2.0)),
            'motor_stall_current_a': float(data.get('motor_stall_current_a', 275)),
            'wheels_per_side': int(data.get('wheels_per_side', 1)),  # Number of wheels per side/axis
            'motor_num_motors': int(data.get('motor_num_motors', 2)),
            'motor_current_limit_a': float(data.get('motor_current_limit_a', 40)),
            'motor_efficiency': float(data.get('motor_efficiency', 0.85)),
            'selected_angle': float(data['selected_angle']) if data.get('selected_angle') else None,
            'idle_speed_rpm': float(data.get('idle_speed_rpm', 500)),
            'gear_ratio': float(data.get('gear_ratio', 2.0)),
            'flywheel_name': data.get('flywheel_name', 'WCP + Dual Stealth (Recommended)'),
            'flywheel_moi_lb_in2': float(data.get('flywheel_moi_lb_in2') or 14.76),
            'wheel_type': data.get('wheel_type', '1'),  # Default to 4" Stealth Wheel
            'wheel_moi_lb_in2': float(data['wheel_moi_lb_in2']) if data.get('wheel_moi_lb_in2') else None,
            'drag_coefficient': float(data.get('drag_coefficient', 0.50)),
            'slip_factor': float(data.get('slip_factor', 1.15)),
            'wheel_diameter_in': float(data.get('wheel_diameter_in', 4.0)),
        }

        logger.info(f'Running analysis with config: motor={config["motor_name"]}, angle={config["selected_angle"]}')
        results = run_analysis(config)
        logger.info('Analysis completed successfully')
        
        # Convert numpy arrays to lists for JSON serialization
        if 'error' not in results:
            results['wheel_rpms'] = {str(k): float(v) for k, v in results['wheel_rpms'].items()}
            results['shot_data'] = {
                str(k): {
                    'v_exit_ms': float(v['v_exit_ms']),
                    'v_exit_fps': float(v['v_exit_fps']),
                    'wheel_rpm': float(v['wheel_rpm']),
                    'entry_angle': float(v['entry_angle']),
                }
                for k, v in results['shot_data'].items()
            }
            results['all_angles'] = [
                {
                    'angle': int(r['angle']),
                    'v_min': float(r['v_min']),
                    'v_max': float(r['v_max']),
                    'v_range': float(r['v_range']),
                    'sens_v_mean': float(r['sens_v_mean']),
                    'sens_a_mean': float(r['sens_a_mean']),
                    'score': float(r['score']),
                }
                for r in results['all_angles']
            ]
            results['optimal_angle'] = int(results['optimal_angle'])
            results['selected_angle'] = int(results['selected_angle'])
            results['optimal_score'] = float(results['optimal_score'])
            results['selected_score'] = float(results['selected_score'])
            results['sens_v_mean'] = float(results['sens_v_mean'])
            results['sens_a_mean'] = float(results['sens_a_mean'])
            results['v_range'] = float(results['v_range'])
            results['min_rpm'] = float(results['min_rpm'])
            results['max_rpm'] = float(results['max_rpm'])
            results['headroom'] = float(results['headroom'])
            results['eff_free_speed'] = float(results['eff_free_speed'])
            results['speed_drop_20'] = float(results['speed_drop_20'])
            results['spinup_8_ms'] = float(results['spinup_8_ms'])
            results['spinup_20_ms'] = float(results['spinup_20_ms'])
            results['spinup_between_8_ms'] = float(results['spinup_between_8_ms'])
            results['spinup_between_20_ms'] = float(results['spinup_between_20_ms'])
            results['total_moi_lb_in2'] = float(results['total_moi_lb_in2'])

        return JsonResponse({'config': config, 'results': results})
    except json.JSONDecodeError as e:
        logger.error(f'JSON decode error: {e}')
        return JsonResponse({'error': f'Invalid JSON in request: {str(e)}'}, status=400)
    except ValueError as e:
        logger.error(f'Value error: {e}')
        return JsonResponse({'error': f'Invalid parameter value: {str(e)}'}, status=400)
    except Exception as e:
        logger.exception('Unexpected error in analyze endpoint')
        return JsonResponse({'error': f'Analysis failed: {str(e)}'}, status=500)
