# FRC 2026 Shooter Analysis

This project performs comprehensive ballistics analysis for the FRC 2026 shooter mechanism, accounting for air resistance and providing optimization recommendations.

## Setup

### Requirements
- Python 3.9+
- `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Environment Setup

1. **Create and activate the environment:**
   ```bash
   uv sync
   ```

2. **Run the analysis:**
   ```bash
   uv run python shooter_with_drag.py
   ```

3. **Check code quality with Ruff:**
   ```bash
   # Check for issues
   uv run ruff check shooter_with_drag.py
   
   # Format code
   uv run ruff format shooter_with_drag.py
   ```

## Project Structure

- `shooter_with_drag.py` - Main analysis script
- `pyproject.toml` - Project configuration and dependencies
- `.python-version` - Python version specification (3.11)

## Configuration

### Adjustable Parameters in `shooter_with_drag.py`

**Angle:**
```python
SELECTED_ANGLE = 65  # Set to None to use calculated optimal angle
```

**Motor Selection** (uncomment desired motor):
```python
# Kraken X44 (default)
# Kraken X60 (more torque, less speed)
# Falcon 500
# NEO
# NEO Vortex
```

## Output

The script generates:
- Console output with detailed analysis tables
- `shooter_analysis_with_drag.png` - 3×3 grid of analysis plots:
  - Row 1: Trajectories, velocity comparison, wheel RPM requirements
  - Row 2: Sensitivity analysis, speed drop comparison, consistency score
  - Row 3: Spin-up vs gear ratio, spin-up vs distance, spin-up curves

## Key Features

- **Air resistance modeling** - Accounts for quadratic drag on foam game pieces
- **Motor thermal derating** - Uses 50% peak torque (not stall) for sustainable operation
- **Configurable motors** - Easy motor preset selection
- **Comprehensive sensitivity analysis** - Shows impact of velocity and angle errors
- **Spin-up time analysis** - Plots across different gear ratios
- **Flywheel configurations** - Compares different mass options

## Notes

All motor calculations use **conservative peak torque values** (not stall torque) with thermal derating to ensure sustainable operation without overheating.
