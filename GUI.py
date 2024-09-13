import sys
import os
import configparser
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QGridLayout,
                             QTabWidget, QGroupBox, QMessageBox)
from PyQt5.QtCore import QProcess, Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from obspy.imaging.beachball import beach
import folium
import io

class Bridge(QObject):
    locationChanged = pyqtSignal(float, float)

    @pyqtSlot(float, float)
    def updateLocation(self, lat, lng):
        print(f"Location updated: {lat}, {lng}")
        self.locationChanged.emit(lat, lng)

class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        layout = QVBoxLayout()
        self.setLayout(layout)

        coordinate = (0, 0)
        self.map = folium.Map(
            zoom_start=2,
            location=coordinate
        )

        self.marker = folium.Marker(
            coordinate,
            draggable=True
        )
        self.marker.add_to(self.map)

        self.webView = QWebEngineView()
        self.channel = QWebChannel()
        self.bridge = Bridge()
        self.channel.registerObject('pyObj', self.bridge)
        self.webView.page().setWebChannel(self.channel)

        self.bridge.locationChanged.connect(self.parent.update_location)

        layout.addWidget(self.webView)

        self.update_map(coordinate)

    def update_map(self, coordinate):
        self.map.location = coordinate
        self.marker.location = coordinate
        data = io.BytesIO()
        self.map.save(data, close_file=False)
        self.webView.setHtml(data.getvalue().decode())

        self.webView.page().runJavaScript('''
            function initMap() {
                var map = document.getElementsByTagName('div')[0];
                if (!map) {
                    console.error('Map div not found');
                    return;
                }
                map.style.height = '100%';
                map.style.width = '100%';
                if (map.leaflet_map) {
                    map.leaflet_map.invalidateSize();
                    map.leaflet_map.on('click', function(e) {
                        if (window.pyObj) {
                            window.pyObj.updateLocation(e.latlng.lat, e.latlng.lng);
                        }
                    });
                    var marker = map.leaflet_map.markers[0];
                    if (marker) {
                        marker.on('dragend', function(e) {
                            if (window.pyObj) {
                                window.pyObj.updateLocation(e.target._latlng.lat, e.target._latlng.lng);
                            }
                        });
                    } else {
                        console.error('Marker not found');
                    }
                } else {
                    console.error('Leaflet map not initialized');
                }
            }

            setTimeout(initMap, 500);
        ''')

class BeachballWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax = self.figure.add_subplot(111)
        self.strike, self.dip, self.rake = 0, 90, 0
        self.draw_beachball()

    def draw_beachball(self):
        self.ax.clear()
        b = beach([self.strike, self.dip, self.rake], width=200, linewidth=2, facecolor='r')
        self.ax.add_collection(b)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-105, 105)
        self.ax.set_ylim(-105, 105)
        self.ax.axis('off')
        self.canvas.draw()

    def mousePressEvent(self, event):
        self.update_focal_mechanism(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.update_focal_mechanism(event)

    def update_focal_mechanism(self, event):
        if self.ax.contains(event)[0]:
            x, y = self.ax.transData.inverted().transform([event.x(), event.y()])
            r = np.sqrt(x**2 + y**2)
            if r <= 100:
                azimuth = np.degrees(np.arctan2(x, y)) % 360
                plunge = 90 - r * 90 / 100
                
                if event.modifiers() & Qt.ShiftModifier:
                    self.dip = plunge
                    self.rake = azimuth
                else:
                    self.strike = azimuth
                    self.dip = plunge

                self.draw_beachball()
                self.parent().update_focal_mechanism_values(self.strike, self.dip, self.rake)

class SeismicEventGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.config = configparser.ConfigParser()
        self.load_config()
        self.output_file = ""

    def init_ui(self):
        self.setWindowTitle('Seismic Event Generator')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Event tab
        event_tab = QWidget()
        event_layout = QGridLayout()
        event_tab.setLayout(event_layout)
        tabs.addTab(event_tab, "Event")

        event_layout.addWidget(QLabel("Latitude:"), 0, 0)
        self.latitude_edit = QLineEdit()
        self.latitude_edit.textChanged.connect(self.update_map_from_input)
        event_layout.addWidget(self.latitude_edit, 0, 1)

        event_layout.addWidget(QLabel("Longitude:"), 1, 0)
        self.longitude_edit = QLineEdit()
        self.longitude_edit.textChanged.connect(self.update_map_from_input)
        event_layout.addWidget(self.longitude_edit, 1, 1)

        event_layout.addWidget(QLabel("Depth:"), 2, 0)
        self.depth_edit = QLineEdit()
        event_layout.addWidget(self.depth_edit, 2, 1)

        event_layout.addWidget(QLabel("Time:"), 3, 0)
        self.time_edit = QLineEdit()
        event_layout.addWidget(self.time_edit, 3, 1)

        self.map_widget = MapWidget(self)
        event_layout.addWidget(self.map_widget, 0, 2, 4, 1)

        # Magnitudes tab
        magnitudes_tab = QWidget()
        magnitudes_layout = QGridLayout()
        magnitudes_tab.setLayout(magnitudes_layout)
        tabs.addTab(magnitudes_tab, "Magnitudes")

        self.magnitude_edits = {}
        for i, mag_type in enumerate(['ML', 'mb', 'Mwp']):
            magnitudes_layout.addWidget(QLabel(f"{mag_type}:"), i, 0)
            self.magnitude_edits[mag_type] = QLineEdit()
            magnitudes_layout.addWidget(self.magnitude_edits[mag_type], i, 1)

        # Focal Mechanism tab
        focal_tab = QWidget()
        focal_layout = QGridLayout()
        focal_tab.setLayout(focal_layout)
        tabs.addTab(focal_tab, "Focal Mechanism")

        self.focal_edits = {}
        for i, param in enumerate(['strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2']):
            focal_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.focal_edits[param] = QLineEdit()
            self.focal_edits[param].textChanged.connect(self.update_beachball)
            focal_layout.addWidget(self.focal_edits[param], i, 1)

        focal_layout.addWidget(QLabel("Mww Magnitudes:"), 6, 0)
        self.mww_magnitudes_edit = QLineEdit()
        focal_layout.addWidget(self.mww_magnitudes_edit, 6, 1)

        self.beachball = BeachballWidget(self)
        focal_layout.addWidget(self.beachball, 0, 2, 7, 1)

        # Inventory tab
        inventory_tab = QWidget()
        inventory_layout = QGridLayout()
        inventory_tab.setLayout(inventory_layout)
        tabs.addTab(inventory_tab, "Inventory")

        inventory_layout.addWidget(QLabel("Inventory Path:"), 0, 0)
        self.inventory_path_edit = QLineEdit()
        inventory_layout.addWidget(self.inventory_path_edit, 0, 1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_inventory)
        inventory_layout.addWidget(browse_button, 0, 2)

        inventory_layout.addWidget(QLabel("Min Distance:"), 1, 0)
        self.min_distance_edit = QLineEdit()
        inventory_layout.addWidget(self.min_distance_edit, 1, 1)

        inventory_layout.addWidget(QLabel("Max Distance:"), 2, 0)
        self.max_distance_edit = QLineEdit()
        inventory_layout.addWidget(self.max_distance_edit, 2, 1)

        # Noise tab
        noise_tab = QWidget()
        noise_layout = QGridLayout()
        noise_tab.setLayout(noise_layout)
        tabs.addTab(noise_tab, "Noise")

        noise_layout.addWidget(QLabel("Pick Time STD:"), 0, 0)
        self.pick_time_std_edit = QLineEdit()
        noise_layout.addWidget(self.pick_time_std_edit, 0, 1)

        noise_layout.addWidget(QLabel("Station Magnitude STD:"), 1, 0)
        self.station_magnitude_std_edit = QLineEdit()
        noise_layout.addWidget(self.station_magnitude_std_edit, 1, 1)

        # Agency tab
        agency_tab = QWidget()
        agency_layout = QGridLayout()
        agency_tab.setLayout(agency_layout)
        tabs.addTab(agency_tab, "Agency")

        agency_layout.addWidget(QLabel("Agency ID:"), 0, 0)
        self.agency_id_edit = QLineEdit()
        agency_layout.addWidget(self.agency_id_edit, 0, 1)

        agency_layout.addWidget(QLabel("Agency ID (lowercase):"), 1, 0)
        self.agency_id_lowercase_edit = QLineEdit()
        agency_layout.addWidget(self.agency_id_lowercase_edit, 1, 1)

        agency_layout.addWidget(QLabel("Author Prefix:"), 2, 0)
        self.author_prefix_edit = QLineEdit()
        agency_layout.addWidget(self.author_prefix_edit, 2, 1)

        # Uncertainties tab
        uncertainties_tab = QWidget()
        uncertainties_layout = QGridLayout()
        uncertainties_tab.setLayout(uncertainties_layout)
        tabs.addTab(uncertainties_tab, "Uncertainties")

        self.uncertainty_edits = {}
        for i, param in enumerate(['origin_time', 'origin_latitude', 'origin_longitude', 'origin_depth', 'origin_min_horizontal', 'origin_max_horizontal']):
            uncertainties_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.uncertainty_edits[param] = QLineEdit()
            uncertainties_layout.addWidget(self.uncertainty_edits[param], i, 1)

        # Phases tab
        phases_tab = QWidget()
        phases_layout = QGridLayout()
        phases_tab.setLayout(phases_layout)
        tabs.addTab(phases_tab, "Phases")

        phases_layout.addWidget(QLabel("S-wave Cutoff:"), 0, 0)
        self.s_wave_cutoff_edit = QLineEdit()
        phases_layout.addWidget(self.s_wave_cutoff_edit, 0, 1)

        # Multiple Origins tab
        multiple_origins_tab = QWidget()
        multiple_origins_layout = QGridLayout()
        multiple_origins_tab.setLayout(multiple_origins_layout)
        tabs.addTab(multiple_origins_tab, "Multiple Origins")

        self.multiple_origins_edits = {}
        for i, param in enumerate(['number_of_origins', 'initial_station_count', 'station_increase_per_origin', 'time_between_origins']):
            multiple_origins_layout.addWidget(QLabel(f"{param}:"), i, 0)
            self.multiple_origins_edits[param] = QLineEdit()
            multiple_origins_layout.addWidget(self.multiple_origins_edits[param], i, 1)

        # Buttons
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)

        save_config_button = QPushButton("Save Config")
        save_config_button.clicked.connect(self.save_config)
        button_layout.addWidget(save_config_button)

        run_button = QPushButton("Run Generator")
        run_button.clicked.connect(self.run_generator)
        button_layout.addWidget(run_button)

        self.dispatch_button = QPushButton("Dispatch Event")
        self.dispatch_button.clicked.connect(self.dispatch_event)
        self.dispatch_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.dispatch_button)

        # Output area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

    def load_config(self):
        if os.path.exists('config.ini'):
            self.config.read('config.ini')
            self.latitude_edit.setText(self.config.get('Event', 'latitude', fallback=''))
            self.longitude_edit.setText(self.config.get('Event', 'longitude', fallback=''))
            self.depth_edit.setText(self.config.get('Event', 'depth', fallback=''))
            self.time_edit.setText(self.config.get('Event', 'time', fallback=''))

            for mag_type in self.magnitude_edits:
                self.magnitude_edits[mag_type].setText(self.config.get('Magnitudes', mag_type.lower(), fallback=''))

            for param in self.focal_edits:
                self.focal_edits[param].setText(self.config.get('FocalMechanism', param, fallback=''))

            self.mww_magnitudes_edit.setText(self.config.get('FocalMechanism', 'mww_magnitudes', fallback=''))

            self.inventory_path_edit.setText(self.config.get('Inventory', 'path', fallback=''))
            self.min_distance_edit.setText(self.config.get('Inventory', 'min_distance', fallback=''))
            self.max_distance_edit.setText(self.config.get('Inventory', 'max_distance', fallback=''))

            self.pick_time_std_edit.setText(self.config.get('Noise', 'pick_time_std', fallback=''))
            self.station_magnitude_std_edit.setText(self.config.get('Noise', 'station_magnitude_std', fallback=''))

            self.agency_id_edit.setText(self.config.get('Agency', 'id', fallback=''))
            self.agency_id_lowercase_edit.setText(self.config.get('Agency', 'id_lowercase', fallback=''))
            self.author_prefix_edit.setText(self.config.get('Agency', 'author_prefix', fallback=''))

            for param in self.uncertainty_edits:
                self.uncertainty_edits[param].setText(self.config.get('Uncertainties', param, fallback=''))

            self.s_wave_cutoff_edit.setText(self.config.get('Phases', 's_wave_cutoff', fallback=''))

            for param in self.multiple_origins_edits:
                self.multiple_origins_edits[param].setText(self.config.get('MultipleOrigins', param, fallback=''))

    def save_config(self):
        self.config['Event'] = {
            'latitude': self.latitude_edit.text(),
            'longitude': self.longitude_edit.text(),
            'depth': self.depth_edit.text(),
            'time': self.time_edit.text()
        }

        self.config['Magnitudes'] = {
            mag_type.lower(): edit.text() for mag_type, edit in self.magnitude_edits.items()
        }

        self.config['FocalMechanism'] = {
            param: edit.text() for param, edit in self.focal_edits.items()
        }
        self.config['FocalMechanism']['mww_magnitudes'] = self.mww_magnitudes_edit.text()

        self.config['Inventory'] = {
            'path': self.inventory_path_edit.text(),
            'min_distance': self.min_distance_edit.text(),
            'max_distance': self.max_distance_edit.text()
        }

        self.config['Noise'] = {
            'pick_time_std': self.pick_time_std_edit.text(),
            'station_magnitude_std': self.station_magnitude_std_edit.text()
        }

        self.config['Agency'] = {
            'id': self.agency_id_edit.text(),
            'id_lowercase': self.agency_id_lowercase_edit.text(),
            'author_prefix': self.author_prefix_edit.text()
        }

        self.config['Uncertainties'] = {
            param: edit.text() for param, edit in self.uncertainty_edits.items()
        }

        self.config['Phases'] = {
            's_wave_cutoff': self.s_wave_cutoff_edit.text()
        }

        self.config['MultipleOrigins'] = {
            param: edit.text() for param, edit in self.multiple_origins_edits.items()
        }

        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

        self.output_text.append("Configuration saved to config.ini")

    def update_beachball(self):
        try:
            strike = float(self.focal_edits['strike1'].text())
            dip = float(self.focal_edits['dip1'].text())
            rake = float(self.focal_edits['rake1'].text())
            self.beachball.strike, self.beachball.dip, self.beachball.rake = strike, dip, rake
            self.beachball.draw_beachball()
        except ValueError:
            pass

    def update_focal_mechanism_values(self, strike, dip, rake):
        self.focal_edits['strike1'].setText(f"{strike:.2f}")
        self.focal_edits['dip1'].setText(f"{dip:.2f}")
        self.focal_edits['rake1'].setText(f"{rake:.2f}")

    def browse_inventory(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Inventory File", "", "XML Files (*.xml)")
        if filename:
            self.inventory_path_edit.setText(filename)

    def run_generator(self):
        self.save_config()
        self.output_text.clear()
        self.output_text.append("Running Seismic Event Generator...")
        
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self.handle_stdout)
        process.readyReadStandardError.connect(self.handle_stderr)
        process.finished.connect(self.process_finished)
        process.start("python", ["synthetic_seismic_event_generator.py"])

    def handle_stdout(self):
        process = self.sender()
        stdout = process.readAllStandardOutput()
        output = bytes(stdout).decode("utf8")
        self.output_text.append(output)
        
        # Check if the output contains the name of the generated file
        for line in output.split('\n'):
            if line.startswith("QuakeML file has been written to "):
                self.output_file = line.split("to ")[-1].strip()
                break

    def handle_stderr(self):
        process = self.sender()
        stderr = process.readAllStandardError()
        output = bytes(stderr).decode("utf8")
        self.output_text.append(output)

    def process_finished(self):
        self.output_text.append("Seismic Event Generator finished execution.")
        if self.output_file:
            self.dispatch_button.setEnabled(True)
        else:
            self.output_text.append("Warning: No output file was detected.")

    def dispatch_event(self):
        if not self.output_file:
            QMessageBox.warning(self, "Error", "No output file available to dispatch.")
            return
        
        self.output_text.append(f"Dispatching event from file: {self.output_file}")
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self.handle_stdout)
        process.readyReadStandardError.connect(self.handle_stderr)
        process.finished.connect(self.dispatch_finished)
        process.start("scdispatch", ["-i", self.output_file])

    def dispatch_finished(self):
        self.output_text.append("Event dispatch process completed.")
        self.dispatch_button.setEnabled(False)
        self.output_file = ""

    def update_location(self, lat, lng):
        self.latitude_edit.setText(f"{lat:.4f}")
        self.longitude_edit.setText(f"{lng:.4f}")
        print(f"Location updated in GUI: {lat:.4f}, {lng:.4f}")

    def update_map_from_input(self):
        try:
            lat = float(self.latitude_edit.text())
            lon = float(self.longitude_edit.text())
            self.map_widget.update_map((lat, lon))
        except ValueError:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SeismicEventGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())
