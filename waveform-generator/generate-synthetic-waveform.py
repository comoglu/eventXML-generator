import numpy as np
from obspy import read_events, read_inventory, Stream, Trace, UTCDateTime
from obspy.core.event import Origin
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from obspy.signal.filter import bandpass
from obspy.signal.invsim import simulate_seismometer
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from scipy.signal import convolve
import matplotlib.pyplot as plt

def generate_synthetic_waveforms(event_file, inventory_file, output_file):
    # Read the event and inventory
    catalog = read_events(event_file)
    event = catalog[0]
    inventory = read_inventory(inventory_file)

    # Get event parameters
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]

    # Create synthetic waveforms for each station
    stream = Stream()
    for network in inventory:
        for station in network:
            traces = create_synthetic_traces(origin, magnitude, station, inventory)
            if traces:
                stream += traces

    # Write the waveforms to a file
    stream.write(output_file, format="MSEED")
    print(f"Synthetic waveforms written to {output_file}")

def create_synthetic_traces(origin, magnitude, station, inventory):
    # Calculate distance and travel time
    distance = locations2degrees(origin.latitude, origin.longitude, 
                                 station.latitude, station.longitude)
    
    # Use TauPy to calculate travel times
    model = TauPyModel(model="iasp91")
    phases = ["P", "S", "PP", "SS", "Rayleigh", "Love"]
    arrivals = model.get_travel_times(source_depth_in_km=origin.depth/1000,
                                      distance_in_degree=distance,
                                      phase_list=phases)
    
    if not arrivals:
        return None

    # Generate synthetic waveforms
    duration = 600  # 10 minutes of data
    sampling_rate = 80.0 if np.random.rand() > 0.5 else 40.0  # Randomly choose 40 or 80 Hz
    npts = int(duration * sampling_rate)
    t = np.arange(npts) / sampling_rate
    
    components = ['Z', 'N', 'E']
    traces = []

    for component in components:
        data = np.zeros(npts)
        
        for arrival in arrivals:
            phase_data = create_phase_data(t, arrival, magnitude.mag, distance, component)
            data += phase_data
        
        # Add attenuation and scattering effects
        data = apply_attenuation_and_scattering(data, distance, sampling_rate)
        
        # Add site-specific effects
        data = apply_site_effects(data, station, component)
        
        # Add some noise
        noise = np.random.normal(0, 0.05 * np.max(np.abs(data)), npts)
        data += noise
        
        # Apply a bandpass filter
        data = bandpass(data, 0.1, 20, sampling_rate, corners=4)
        
        # Create trace
        tr = Trace(data=data)
        tr.stats.network = station.network_code
        tr.stats.station = station.code
        tr.stats.channel = f"BH{component}"
        tr.stats.starttime = origin.time
        tr.stats.sampling_rate = sampling_rate
        
        # Apply instrument response
        tr = apply_instrument_response(tr, station, inventory)
        
        traces.append(tr)
    
    return traces

def create_phase_data(t, arrival, magnitude, distance, component):
    # Create a more complex pulse for each phase
    arrival_time = arrival.time
    f = 2.0  # dominant frequency
    tau = (t - arrival_time) * f * np.pi
    amplitude = 10**(magnitude - 5) * np.exp(-0.002 * distance)  # Distance-dependent amplitude
    
    if arrival.name in ['P', 'PP']:
        wave = amplitude * (1 - 2 * tau**2) * np.exp(-tau**2) * (1 if component == 'Z' else 0.5)
    elif arrival.name in ['S', 'SS']:
        wave = amplitude * 0.8 * tau * np.exp(-tau**2) * (0.5 if component == 'Z' else 1)
    elif arrival.name == 'Rayleigh':
        wave = amplitude * 0.5 * np.sin(2*np.pi*f*(t-arrival_time)) * np.exp(-(t-arrival_time)/10)
    elif arrival.name == 'Love':
        wave = amplitude * 0.3 * np.sin(2*np.pi*f*(t-arrival_time)) * np.exp(-(t-arrival_time)/8) * (1 if component != 'Z' else 0)
    else:
        wave = np.zeros_like(t)
    
    return wave

def apply_attenuation_and_scattering(data, distance, sampling_rate):
    # Frequency-dependent Q model
    def Q(f):
        return 100 + 10 * f  # Simple frequency-dependent Q model

    # Create attenuation operator
    f = np.fft.fftfreq(len(data), 1/sampling_rate)
    omega = 2 * np.pi * np.abs(f)
    v = 3.5  # Average velocity in km/s
    t_star = distance / (Q(np.abs(f)) * v)
    atten_op = np.exp(-omega * t_star / 2)
    
    # Apply attenuation in frequency domain
    data_fd = np.fft.fft(data)
    data_fd *= atten_op
    attenuated_data = np.real(np.fft.ifft(data_fd))
    
    # Add scattering using energy-flux model (simplified)
    t = np.arange(len(data)) / sampling_rate
    scattering = np.sqrt(t) * np.exp(-t / 20) * np.random.randn(len(t))
    scattered_data = np.convolve(attenuated_data, scattering, mode='same')
    
    return scattered_data

def apply_site_effects(data, station, component):
    # Simple site amplification based on station elevation (just an example)
    elevation = station.elevation
    amplification = 1 + 0.0001 * elevation  # Higher elevation, slightly higher amplification
    
    # Add resonance effects (simplified)
    t = np.arange(len(data)) / station.channels[0].sample_rate
    resonance_freq = 2 + np.random.rand() * 3  # Random resonance between 2-5 Hz
    resonance = 0.1 * np.sin(2 * np.pi * resonance_freq * t)
    
    # Apply Konno-Ohmachi smoothing to simulate complex site response
    freq = np.fft.rfftfreq(len(data), d=1/station.channels[0].sample_rate)
    smoothed_response = konno_ohmachi_smoothing(np.random.rand(len(freq)) + 1, freq, bandwidth=40, normalize=True)
    
    data_fd = np.fft.rfft(data)
    data_fd *= smoothed_response
    site_response_data = np.fft.irfft(data_fd)
    
    return amplification * site_response_data + resonance

def apply_instrument_response(trace, station, inventory):
    # Get the response for this specific channel
    response = inventory.get_response(trace.id, trace.stats.starttime)
    
    # Simulate seismometer
    if response:
        try:
            # Remove sensitivity
            trace.data = trace.data / response.instrument_sensitivity.value
            
            # Simulate seismometer to velocity
            trace.data = simulate_seismometer(
                trace.data,
                trace.stats.sampling_rate,
                paz_remove=None,  # We've already removed sensitivity
                paz_simulate=response.get_paz(),
                water_level=60.0
            )
        except Exception as e:
            print(f"Error applying instrument response for {trace.id}: {e}")
            # If there's an error, we'll just return the original trace
            pass
    
    return trace

if __name__ == "__main__":
    event_file = "synthetic_event_multiple_origins.xml"
    inventory_file = "./inventory/AU.xml"
    output_file = "synthetic_waveforms.mseed"
    generate_synthetic_waveforms(event_file, inventory_file, output_file)
