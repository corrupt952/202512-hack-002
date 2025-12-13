# GokiRun

A cockroach simulation that runs on your macOS desktop. Based on scientific research of real cockroach escape behavior.

## Features

- Realistic escape behavior based on published research (Domenici et al., Camhi & Nolen)
- Four preferred escape trajectories (90, 120, 150, 180 degrees)
- Two-stage threat response (pause for weak stimuli, escape for strong stimuli)
- Hides under windows (recognizes as safe zones)
- Click to squash with mysterious revival effect
- Metal compute shaders for parallel processing

## Quick start

### Prerequisites

- macOS 11.0 or later (Apple Silicon)
- Xcode Command Line Tools

### Installation

```bash
git clone https://github.com/user/GokiRun.git
cd GokiRun
./build.sh
```

### Usage

```bash
# Run with default count (10)
./GokiRunMetal

# Run with custom count
./GokiRunMetal -c 20
./GokiRunMetal --count 50
```

### Permissions

The app requires accessibility permissions for global mouse monitoring. Grant access in System Settings > Privacy & Security > Accessibility.

## Behavior

### States

| State | Description |
|-------|-------------|
| IDLE | Stationary, antennae scanning |
| RANDOM_WALK | Exploring, frequent direction changes |
| WALL_FOLLOW | Moving along screen edges (safety) |
| ESCAPE | Running away from mouse cursor |
| STOP | Frozen, checking surroundings |
| DEAD | Squashed by click |
| REVIVING | Mysterious revival in progress |

### Interactions

- **Mouse cursor**: Acts as threat, triggers escape or pause behavior
- **Click**: Squashes the nearest cockroach
- **Windows**: Cockroaches hide under other application windows

## Research references

- Domenici et al. (2008) - Preferred escape trajectories
- Domenici et al. (2010) - Away vs towards responses
- Camhi & Nolen (1981) - Two-stage threat response
- Daltorio et al. (2013) - RAMBLER exploration algorithm

## License

MIT
