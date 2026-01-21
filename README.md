# FRC 2026 Shooter Analysis

This project performs comprehensive ballistics analysis for the FRC 2026 shooter mechanism, accounting for air resistance and providing optimization recommendations.

## Features

- **Web Interface** - Modern Django web application with interactive UI
- **Command Line Tool** - Original Python script for batch analysis
- **Air resistance modeling** - Accounts for quadratic drag on foam game pieces
- **Conservative motor torque** - Uses peak torque values (not stall) for sustainable operation
- **Configurable motors** - Easy motor preset selection
- **Comprehensive sensitivity analysis** - Shows impact of velocity and angle errors
- **Spin-up time analysis** - Calculates across different gear ratios
- **Flywheel configurations** - Compares different mass options

## Setup

### Requirements
- Python 3.9+
- `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/)) OR pip

### Web Application (Django)

1. **Install dependencies:**
   ```bash
   uv sync
   # OR
   pip install -r requirements.txt
   ```

2. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

3. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

4. **Open in browser:**
   Navigate to `http://127.0.0.1:8000/`

### Command Line Tool

1. **Run the analysis:**
   ```bash
   uv run python shooter_tuning_tool.py
   # OR
   python shooter_tuning_tool.py
   ```

2. **Run the detailed analysis script:**
   ```bash
   uv run python shooter_with_drag.py
   ```

## Project Structure

- `shooter_web/` - Django project settings
- `shooter/` - Django app with views, URLs, and physics module
- `templates/shooter/` - HTML templates
- `shooter_tuning_tool.py` - Interactive command-line tool
- `shooter_with_drag.py` - Detailed analysis script
- `shooter/physics.py` - Core physics functions (shared between CLI and web)
- `pyproject.toml` - Project configuration and dependencies
- `requirements.txt` - Python dependencies for pip

## Web Application Usage

1. Configure your shooter parameters using the form on the left
2. Click "Run Analysis" to calculate optimal settings
3. View results including:
   - Performance metrics with status indicators
   - Shot data table with velocities and RPMs
   - RPM lookup table for robot code
   - Interactive charts for angle comparison and RPM requirements
   - Recommendations for optimization

## Command Line Tool Usage

The `shooter_tuning_tool.py` provides an interactive menu-driven interface:
- Configure motor, gear ratio, flywheel, and other parameters
- View detailed analysis results
- Generate plots
- Save/load configurations

## Output

### Web Application
- Real-time analysis results displayed in the browser
- Interactive charts using Plotly
- RPM lookup tables ready for robot code

### Command Line Tool
- Console output with detailed analysis tables
- `shooter_tuning_plots.png` - Analysis plots
- `shooter_config_*.json` - Saved configurations

### Detailed Analysis Script
- `shooter_analysis_with_drag.png` - 3×3 grid of analysis plots

## Notes

All motor calculations use **conservative peak torque values** (not stall torque) to ensure sustainable operation without overheating.
