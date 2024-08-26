# SeisComP Event XML Generator

## Overview

This project provides a Python-based tool for generating synthetic seismic event data in SeisComP XML format. It's designed to create realistic earthquake event data, including multiple origins, focal mechanisms, and various magnitude types. This tool is particularly useful for testing seismic data processing systems, developing algorithms, and for educational purposes in seismology.

## Features

- Generate synthetic earthquake events with multiple origins
- Create realistic focal mechanisms with moment tensors
- Calculate and associate various magnitude types (ML, mb, Ms, Mw)
- Produce output in SeisComP XML format
- Configurable event parameters (location, depth, time, magnitudes, etc.)
- Station selection based on distance from the event

## Requirements

- Python 3.7+
- ObsPy
- NumPy
- ConfigParser

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/comoglu/eventXML-generator.git
   cd seiscomp-event-xml-generator
   ```

2. Install the required Python packages:
   ```
   pip install obspy numpy configparser
   ```

## Configuration

Before running the script, you need to set up the configuration file `config.ini`. This file allows you to specify various parameters for the synthetic event generation. Here's an example of what it might contain:

```ini
[Event]
latitude = -7.69
longitude = 125.88
depth = 10
time = now

[Magnitudes]
ML = 5.5
mb = 5.2
Ms = 5.7

[FocalMechanism]
strike1 = 30
dip1 = 60
rake1 = 90

[Inventory]
path = /path/to/your/inventory.xml
min_distance = 0
max_distance = 50

[Agency]
id = GA
author_prefix = auto_

[MultipleOrigins]
number_of_origins = 10
initial_station_count = 10
station_increase_per_origin = 5
time_between_origins = 60

[Uncertainties]
origin_time = 0.1
origin_latitude = 0.01
origin_depth = 5000

[Noise]
pick_time_std = 0.5
station_magnitude_std = 0.2
```

Adjust these values according to your needs.

## Usage

To generate a synthetic event, run the main script:

```
python seismic_event_generator.py
```

This will create a SeisComP XML file named `synthetic_event_multiple_origins.xml` in the current directory.

## Output

The script generates a SeisComP XML file containing:

- Multiple origins for the event
- Associated picks and arrivals
- Various magnitude types
- Focal mechanisms with moment tensors
- Station information used for the event

## Contributing

Contributions to improve the script or add new features are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool uses ObsPy for seismological computations and data structures.
- Thanks to the seismological community for providing resources and knowledge that made this tool possible.

## Disclaimer

This tool generates synthetic data and should not be used for real-time seismic monitoring or official reporting. Always verify and validate the generated data before using it in any critical applications.
