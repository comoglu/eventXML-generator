import numpy as np
import os
from obspy import read_events, read_inventory, Stream, Trace, UTCDateTime, Inventory
from obspy.core.event import Origin
from obspy.geodetics import locations2degrees, kilometers2degrees
from obspy.taup import TauPyModel
from obspy.signal.filter import bandpass
from obspy.signal.invsim import simulate_seismometer
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
from obspy.core.inventory import PolesZerosResponseStage
from obspy.geodetics.base import gps2dist_azimuth
import multiprocessing as mp
from functools import partial
import cProfile
import pstats

def read_all_inventories(inventory_dir):
    combined_inventory = Inventory()
    for filename in os.listdir(inventory_dir):
        if filename.endswith('.xml'):
            file_path = os.path.join(inventory_dir, filename)
            try:
                inv = read_inventory(file_path)
                combined_inventory += inv
                print(f"Successfully read inventory from {filename}")
            except Exception as e:
                print(f"Error reading inventory from {filename}: {e}")
    return combined_inventory

def get_available_channels(inventory):
    """
    Extract all available network.station.location.channel combinations from the inventory.
    """
    channels = set()
    for network in inventory:
        for station in network:
            for channel in station:
                channel_id = f"{network.code}.{station.code}.{channel.location_code}.{channel.code}"
                channels.add(channel_id)
    return channels

def generate_synthetic_waveforms(event_file, inventory_dir, output_file, channel_list=None):
    catalog = read_events(event_file)
    event = catalog[0]
    inventory = read_all_inventories(inventory_dir)
    
    if not inventory:
        print("No valid inventory files found. Exiting.")
        return

    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]

    # If channel_list is not provided, extract it from the inventory
    if channel_list is None:
        available_channels = get_available_channels(inventory)
    else:
        available_channels = set(channel_list)

    # Pre-calculate travel times
    model = TauPyModel(model="iasp91")
    phases = ["P", "S", "PP", "SS"]

    # Use multiprocessing to generate waveforms
    with mp.Pool(processes=mp.cpu_count()) as pool:
        partial_create_traces = partial(create_synthetic_traces, 
                                        origin, magnitude, inventory, 
                                        model, phases, available_channels)
        all_traces = pool.starmap(partial_create_traces, 
                                  [(network, station) for network in inventory for station in network])

    # Flatten the list of traces
    stream = Stream([tr for traces in all_traces if traces for tr in traces])

    stream.write(output_file, format="MSEED")
    print(f"Synthetic waveforms written to {output_file}")

def create_synthetic_traces(origin, magnitude, inventory, model, phases, available_channels, network, station):
    # Calculate distance and azimuth
    distance_m, azimuth, _ = gps2dist_azimuth(origin.latitude, origin.longitude,
                                              station.latitude, station.longitude)
    distance_deg = kilometers2degrees(distance_m / 1000)

    # Calculate travel times
    arrivals = model.get_travel_times(source_depth_in_km=origin.depth / 1000,
                                      distance_in_degree=distance_deg,
                                      phase_list=phases)

    # Set up time array
    duration = 600
    sampling_rate = 80.0 if np.random.rand() > 0.5 else 40.0
    npts = int(duration * sampling_rate)
    t = np.arange(npts) / sampling_rate

    traces = []
    for component in ['Z', 'N', 'E']:
        data = np.zeros(npts)

        # Add body waves
        for arrival in arrivals:
            data += create_phase_data(t, arrival, magnitude.mag, distance_deg, component)

        # Add surface waves
        data += create_surface_waves(t, distance_deg, magnitude.mag, component)

        # Apply attenuation and scattering
        data = apply_attenuation_and_scattering(data, distance_deg, sampling_rate)

        # Apply site effects
        data = apply_site_effects(data, station, component)

        # Add some noise
        noise = np.random.normal(0, 0.05 * np.max(np.abs(data)), npts)
        data += noise

        # Apply bandpass filter
        data = bandpass(data, 0.1, 20, sampling_rate, corners=4)

        # Create trace
        tr = Trace(data=data)
        tr.stats.network = network.code
        tr.stats.station = station.code
        tr.stats.channel = f"BH{component}"
        tr.stats.starttime = origin.time
        tr.stats.sampling_rate = sampling_rate

        # Add origin and station information to trace stats
        tr.stats.origin_time = origin.time
        tr.stats.origin_latitude = origin.latitude
        tr.stats.origin_longitude = origin.longitude
        tr.stats.origin_depth = origin.depth
        tr.stats.distance = distance_deg
        tr.stats.azimuth = azimuth
        tr.stats.back_azimuth = (azimuth + 180) % 360  # Calculate back azimuth

        # Apply instrument response
        tr = apply_instrument_response(tr, inventory, available_channels)

        traces.append(tr)

    return traces

def create_phase_data(t, arrival, magnitude, distance_deg, component):
    arrival_time = arrival.time
    f = 2.0  # dominant frequency
    tau = (t - arrival_time) * f * np.pi
    amplitude = 10**(magnitude - 5) * np.exp(-0.002 * distance_deg)
    
    if arrival.name in ['P', 'PP']:
        wave = amplitude * (1 - 2 * tau**2) * np.exp(-tau**2) * (1 if component == 'Z' else 0.5)
    elif arrival.name in ['S', 'SS']:
        wave = amplitude * 0.8 * tau * np.exp(-tau**2) * (0.5 if component == 'Z' else 1)
    else:
        wave = np.zeros_like(t)
    
    return wave

def create_surface_waves(t, distance_deg, magnitude, component):
    rayleigh_velocity = 3.0
    love_velocity = 3.5
    
    rayleigh_arrival_time = distance_deg * 111.19 / rayleigh_velocity
    love_arrival_time = distance_deg * 111.19 / love_velocity
    
    rayleigh_amplitude = 10**(magnitude - 6) * np.exp(-0.001 * distance_deg)
    love_amplitude = 0.7 * rayleigh_amplitude
    
    rayleigh_wave = rayleigh_amplitude * np.sin(2*np.pi*0.1*(t-rayleigh_arrival_time)) * \
                    np.exp(-(t-rayleigh_arrival_time)/50) * \
                    (t > rayleigh_arrival_time)
    
    love_wave = love_amplitude * np.sin(2*np.pi*0.08*(t-love_arrival_time)) * \
                np.exp(-(t-love_arrival_time)/60) * \
                (t > love_arrival_time)
    
    if component == 'Z':
        return rayleigh_wave
    elif component in ['N', 'E']:
        return rayleigh_wave * 0.7 + love_wave
    else:
        return np.zeros_like(t)

def apply_attenuation_and_scattering(data, distance_deg, sampling_rate):
    def Q(f):
        return 100 + 10 * f

    f = np.fft.fftfreq(len(data), 1/sampling_rate)
    omega = 2 * np.pi * np.abs(f)
    v = 3.5
    t_star = distance_deg / (Q(np.abs(f)) * v)
    atten_op = np.exp(-omega * t_star / 2)
    
    data_fd = np.fft.fft(data)
    data_fd *= atten_op
    attenuated_data = np.real(np.fft.ifft(data_fd))
    
    t = np.arange(len(data)) / sampling_rate
    scattering = np.sqrt(t) * np.exp(-t / 20) * np.random.randn(len(t))
    scattered_data = np.convolve(attenuated_data, scattering, mode='same')
    
    return scattered_data

def apply_site_effects(data, station, component):
    elevation = station.elevation
    amplification = 1 + 0.0001 * elevation
    
    t = np.arange(len(data)) / station.channels[0].sample_rate
    resonance_freq = 2 + np.random.rand() * 3
    resonance = 0.1 * np.sin(2 * np.pi * resonance_freq * t)
    
    freq = np.fft.rfftfreq(len(data), d=1/station.channels[0].sample_rate)
    smoothed_response = konno_ohmachi_smoothing(np.random.rand(len(freq)) + 1, freq, bandwidth=40, normalize=True)
    
    data_fd = np.fft.rfft(data)
    data_fd *= smoothed_response
    site_response_data = np.fft.irfft(data_fd)
    
    return amplification * site_response_data + resonance

def apply_instrument_response(trace, inventory, available_channels=None):
    """
    Apply instrument response using only available channel combinations.
    
    :param trace: ObsPy Trace object
    :param inventory: ObsPy Inventory object
    :param available_channels: Set of available channel IDs (optional)
    :return: Trace with instrument response applied
    """
    if available_channels is None:
        available_channels = get_available_channels(inventory)
    
    channel_id = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}"
    
    if channel_id not in available_channels:
        print(f"Channel {channel_id} not found in inventory. Skipping instrument response.")
        return trace
    
    try:
        response = inventory.get_response(channel_id, trace.stats.starttime)
        if response:
            # Apply instrument response (implementation as before)
            trace.data = trace.data / response.instrument_sensitivity.value
            paz = get_paz_from_response(response)
            if paz:
                trace.data = simulate_seismometer(
                    trace.data,
                    trace.stats.sampling_rate,
                    paz_remove=None,
                    paz_simulate=paz,
                    water_level=60.0
                )
            print(f"Successfully applied instrument response for {channel_id}")
        else:
            print(f"No response found for {channel_id}")
    except Exception as e:
        print(f"Error applying instrument response for {channel_id}: {e}")
    
    return trace

def get_paz_from_response(response):
    for stage in response.response_stages:
        if isinstance(stage, PolesZerosResponseStage):
            return {
                'poles': stage.poles,
                'zeros': stage.zeros,
                'gain': stage.normalization_factor,
                'sensitivity': response.instrument_sensitivity.value
            }
    try:
        return response.get_paz()
    except:
        return None

# Usage example:
if __name__ == "__main__":
    event_file = "synthetic_event_multiple_origins.xml"
    inventory_dir = "./test"
    output_file = "synthetic_waveforms1.mseed"

    # Use cProfile to identify performance bottlenecks
    profiler = cProfile.Profile()
    profiler.enable()

    # Option 1: Let the script extract available channels
    generate_synthetic_waveforms(event_file, inventory_dir, output_file)
    
    # Option 2: Provide a list of available channels
    # channel_list = ["AU.ARMA..BHZ", "AU.ARMA..BHN", "AU.ARMA..BHE", ...]
    # generate_synthetic_waveforms(event_file, inventory_dir, output_file, channel_list)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 time-consuming functions
