# eventXML-generator

A synthetic seismic event generator with a graphical user interface for creating and dispatching custom seismic events.

## Description

The eventXML-generator is a Python-based tool designed to generate synthetic seismic events and output them in QuakeML format. It features a user-friendly GUI for easy configuration and execution of the event generation process.

## Features

- Interactive map for selecting event location
- Customizable event parameters (magnitude, depth, time, etc.)
- Focal mechanism configuration with real-time beachball diagram
- Station inventory management
- Noise and uncertainty modeling
- Multiple origin simulation
- Event dispatch functionality

## Requirements

- Python 3.6+
- PyQt5
- ObsPy
- Folium
- Matplotlib
- NumPy
- Configparser

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:comoglu/eventXML-generator.git
   ```

2. Navigate to the project directory:
   ```
   cd eventXML-generator
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the GUI:
   ```
   python GUI.py
   ```

2. Configure the event parameters using the various tabs in the interface.

3. Click "Save Config" to save your configuration to `config.ini`.

4. Click "Run Generator" to create the synthetic event.

5. Once the event is generated, you can click "Dispatch Event" to send it to your seismic processing system.

## Configuration

The `config.ini` file contains all the parameters for event generation. You can edit this file directly or use the GUI to modify the settings.

## Files

- `GUI.py`: The main graphical user interface for the application.
- `synthetic_seismic_event_generator.py`: The core script for generating synthetic seismic events.
- `config.ini`: Configuration file for event parameters.

## Contributing

Contributions to the eventXML-generator are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Mustafa Comoglu

## Acknowledgments

- ObsPy community for their comprehensive seismology toolkit
- PyQt5 for the GUI framework
- Folium for the interactive map functionality
