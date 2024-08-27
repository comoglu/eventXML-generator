#!/usr/bin/env python3

import os
import sys
import numpy as np
import random
import string
import datetime
from obspy import UTCDateTime, Catalog, read_inventory
from obspy.core.event import (
    Event, Origin, Magnitude, Pick, Arrival, WaveformStreamID,
    OriginQuality, OriginUncertainty, QuantityError, Comment,
    FocalMechanism, MomentTensor, NodalPlane, NodalPlanes, PrincipalAxes,
    Axis, SourceTimeFunction, Tensor, CreationInfo, StationMagnitude,
    Amplitude, DataUsed 
)
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.taup import TauPyModel
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')


def create_resource_id(id_string):
    # Remove any existing 'smi:' prefix to avoid nesting
    id_string = id_string.replace('smi:', '')
    
    # Remove the agency prefix if it's already there
    agency_prefix = f"{config['Agency']['id'].lower()}/"
    if id_string.startswith(agency_prefix):
        id_string = id_string[len(agency_prefix):]
    
    return f"smi:{config['Agency']['id'].lower()}/{id_string}"

def get_config_float(section, key, fallback=None):
    try:
        return float(config[section][key])
    except ValueError:
        print(f"Warning: Could not convert {section}.{key} to float. Using fallback value: {fallback}")
        return fallback

def get_config_int(section, key, fallback=None):
    try:
        return int(config[section][key])
    except ValueError:
        print(f"Warning: Could not convert {section}.{key} to int. Using fallback value: {fallback}")
        return fallback

def generate_event_id(agency_id, event_time):
    current_year = event_time.year
    
    # Calculate the fraction of the year that has passed
    year_start = datetime.datetime(current_year, 1, 1)
    year_fraction = (event_time - year_start).total_seconds() / (366 * 24 * 60 * 60)  # Using 366 to account for leap years
    
    # Convert the fraction to a base-26 number with 6 digits
    letter_value = int(year_fraction * (26**6))
    
    # Convert the number to letters
    letters = ""
    for _ in range(6):
        letter_value, remainder = divmod(letter_value, 26)
        letters = string.ascii_lowercase[remainder] + letters
    
    return create_resource_id(f"event/{agency_id}{current_year}{letters}")


def create_magnitude(mag_value, mag_type, origin_id, method_description, station_count, event_id, magnitude_index, uncertainty=None):
    mag = Magnitude()
    mag.mag = mag_value
    mag.magnitude_type = mag_type
    mag.origin_id = origin_id
    mag.method_id = create_resource_id(f"method/{mag_type.lower()}")
    mag.station_count = station_count
    mag.evaluation_mode = "manual"
    mag.evaluation_status = "confirmed"
    if uncertainty:
        mag.mag_errors = QuantityError(uncertainty=uncertainty)
    mag.comments.append(Comment(text=method_description))
    mag.creation_info = CreationInfo(
        agency_id=config['Agency']['id'],
        author=f"{config['Agency']['author_prefix']}olv@testtest",
        creation_time=UTCDateTime()
    )
    mag.resource_id = create_resource_id(f"magnitude/{mag_type.lower()}/{event_id}/{magnitude_index}")
    return mag

def create_magnitudes(event, origin, magnitudes, stations, event_id):
    for i, (mag_type, mag_value) in enumerate(magnitudes.items()):
        # Add small random adjustment to magnitude
        adjusted_mag = mag_value + np.random.uniform(-0.2, 0.2)
        # Ensure magnitude doesn't exceed realistic values
        adjusted_mag = min(adjusted_mag, 9.5)  # 9.5 is about the maximum recorded magnitude

        method_description = f"Network magnitude using {mag_type}"
        mag = create_magnitude(adjusted_mag, mag_type, origin.resource_id, method_description, len(stations), event_id, i, uncertainty=0.1)
        event.magnitudes.append(mag)

        for station in stations:
            sta_mag = StationMagnitude()
            sta_mag.origin_id = origin.resource_id
            sta_mag.mag = adjusted_mag + np.random.normal(0, get_config_float('Noise', 'station_magnitude_std', 0.2))
            sta_mag.magnitude_type = mag_type
            sta_mag.waveform_id = WaveformStreamID(network_code=station['network'], station_code=station['code'])
            sta_mag.resource_id = create_resource_id(f"station_magnitude/{mag_type.lower()}/{event_id}/{i}/{station['code']}")
            event.station_magnitudes.append(sta_mag)

    # Create moment magnitude (Mw) based on the largest magnitude
    largest_mag = max(mag.mag for mag in event.magnitudes)
    scalar_moment = 10**(1.5 * largest_mag + 9.1)
    mw_mag = create_moment_magnitude(scalar_moment, origin.resource_id, event_id, len(stations), len(event.magnitudes))
    event.magnitudes.append(mw_mag)

    # Set the moment magnitude as the preferred magnitude
    event.preferred_magnitude_id = mw_mag.resource_id



def create_pick(time, waveform_id, phase_hint, evaluation_mode="automatic"):
    pick = Pick()
    pick.time = time
    pick.time_errors = QuantityError(uncertainty=0.05)  # 50 ms uncertainty
    pick.waveform_id = waveform_id
    pick.phase_hint = phase_hint
    pick.evaluation_mode = evaluation_mode
    pick.resource_id = create_resource_id(f"pick/{waveform_id.get_seed_string()}/{phase_hint}/{time.timestamp}")
    pick.creation_info = CreationInfo(
        agency_id=config['Agency']['id'],
        author=f"{config['Agency']['author_prefix']}autopick@testtest",
        creation_time=UTCDateTime()
    )
    return pick

def create_arrival(pick_id, phase, azimuth, distance, time_residual, time_weight):
    arrival = Arrival()
    arrival.pick_id = pick_id
    arrival.phase = phase
    arrival.azimuth = azimuth
    arrival.distance = distance
    arrival.time_residual = time_residual
    arrival.time_weight = time_weight
    arrival.earth_model_id = create_resource_id("earthmodel/iasp91")
    arrival.resource_id = create_resource_id(f"arrival/{pick_id}/{phase}")  # Added
    return arrival

def create_focal_mechanism(strike1, dip1, rake1, strike2, dip2, rake2, scalar_moment, event_id):
    fm = FocalMechanism()
    fm.resource_id = create_resource_id(f"focalmechanism/{event_id}")
    
    # Create nodal planes
    np1 = NodalPlane(strike=strike1, dip=dip1, rake=rake1)
    np2 = NodalPlane(strike=strike2, dip=dip2, rake=rake2)
    fm.nodal_planes = NodalPlanes(nodal_plane_1=np1, nodal_plane_2=np2)
    
    # Calculate and set principal axes
    t_axis, p_axis, n_axis = calculate_principal_axes(strike1, dip1, rake1)
    fm.principal_axes = PrincipalAxes(
        t_axis=Axis(azimuth=t_axis[0], plunge=t_axis[1], length=1),
        p_axis=Axis(azimuth=p_axis[0], plunge=p_axis[1], length=-1),
        n_axis=Axis(azimuth=n_axis[0], plunge=n_axis[1], length=0)
    )
    
    # Create moment tensor
    mt = create_moment_tensor(scalar_moment, strike1, dip1, rake1, event_id)
    fm.moment_tensor = mt
    
    # Calculate moment magnitude
    mw = (2/3) * (np.log10(scalar_moment) - 9.1)
    fm.moment_tensor.derived_origin_id = create_resource_id("origin/initial")
    fm.moment_tensor.moment_magnitude_id = create_resource_id(f"magnitude/mw/{mw:.2f}")
    
    fm.creation_info = CreationInfo(agency_id=config['Agency']['id'], author=f"{config['Agency']['author_prefix']}autofm@testtest", creation_time=UTCDateTime())
    return fm, mw

def create_focal_mechanisms(lat, lon, depth, time, scalar_moment, event_id, initial_strike, initial_dip, initial_rake, num_mechanisms=3, variation=15):
    focal_mechanisms = []
    magnitudes = []

    for i in range(num_mechanisms):
        print(f"Creating focal mechanism {i+1}/{num_mechanisms}")

        # Create origin (existing code remains the same)
        origin = Origin()
        origin.resource_id = f"origin/focalmechanism/{event_id}/{i}"


        # Create origin (existing code)
        origin = Origin()
        origin.resource_id = create_resource_id(f"origin/focalmechanism/{event_id}/{i}")
        origin.time = time + np.random.normal(0, 1)
        origin.latitude = lat + np.random.normal(0, 0.01)
        origin.longitude = lon + np.random.normal(0, 0.01)
        origin.depth = depth * 1000 + np.random.normal(0, 500)
        origin.depth_type = "from moment tensor inversion"
        origin.method_id = "FocalMechanism"
        origin.earth_model_id = "gemini-prem"
        
        # Create origin quality
        origin.quality = OriginQuality(
            used_phase_count=np.random.randint(5, 15),
            associated_station_count=np.random.randint(50, 80),
            used_station_count=np.random.randint(5, 15),
            azimuthal_gap=np.random.uniform(90, 180),
            maximum_distance=np.random.uniform(30, 50),
            minimum_distance=np.random.uniform(1, 10)
        )
        
        origin.origin_type = "hypocenter"
        origin.creation_info = CreationInfo(
            agency_id="GA",
            author="scautomt@testtest",
            creation_time=UTCDateTime()
        )
        print(f"Created origin with resource_id: {origin.resource_id}")
        
        # Create focal mechanism
        fm = FocalMechanism()
        fm.resource_id = create_resource_id(f"focalmechanism/{event_id}/{i}")
        
        # Add random variations to the initial focal mechanism
        strike1 = initial_strike + np.random.uniform(-variation, variation)
        dip1 = initial_dip + np.random.uniform(-variation/2, variation/2)
        rake1 = initial_rake + np.random.uniform(-variation, variation)
        
        # Ensure values are within valid ranges
        strike1 = strike1 % 360
        dip1 = max(0, min(90, dip1))
        rake1 = max(-180, min(180, rake1))
        
        # Calculate the auxiliary plane
        strike2, dip2, rake2 = calculate_auxiliary_plane(strike1, dip1, rake1)
        
        # Create nodal planes
        np1 = NodalPlane(strike=strike1, dip=dip1, rake=rake1)
        np2 = NodalPlane(strike=strike2, dip=dip2, rake=rake2)
        fm.nodal_planes = NodalPlanes(nodal_plane_1=np1, nodal_plane_2=np2)
        
        # Calculate and set principal axes
        t_axis, p_axis, n_axis = calculate_principal_axes(strike1, dip1, rake1)
        fm.principal_axes = PrincipalAxes(
            t_axis=Axis(azimuth=t_axis[0], plunge=t_axis[1], length=1),
            p_axis=Axis(azimuth=p_axis[0], plunge=p_axis[1], length=-1),
            n_axis=Axis(azimuth=n_axis[0], plunge=n_axis[1], length=0)
        )
        
        # Create moment tensor
        mt = create_moment_tensor(scalar_moment, strike1, dip1, rake1, f"{event_id}/{i}")
        fm.moment_tensor = mt
        
        # Link moment tensor to the origin
        fm.moment_tensor.derived_origin_id = origin.resource_id

        print(f"Created focal mechanism with resource_id: {fm.resource_id}")
        print(f"Linked moment tensor to origin: {origin.resource_id}")

        # Calculate Mw from scalar moment
        mw = (2/3) * (np.log10(scalar_moment) - 9.1)

        # Create Mw magnitude
        mag_mw = create_magnitude(mw, "Mw", origin.resource_id, "Moment magnitude from focal mechanism", origin.quality.used_station_count, event_id, len(magnitudes) + i*2)
        mag_mw.mag = mw
        mag_mw.magnitude_type = "Mw"
        mag_mw.origin_id = origin.resource_id
        mag_mw.method_id = create_resource_id("method/MT")
        mag_mw.station_count = origin.quality.used_station_count
        mag_mw.azimuthal_gap = origin.quality.azimuthal_gap
        mag_mw.creation_info = origin.creation_info
        mag_mw.resource_id = create_resource_id(f"magnitude/mw/{event_id}/{i}")

        mag_mww = create_magnitude(mw, "Mww", origin.resource_id, "W-phase moment magnitude", origin.quality.used_station_count, event_id, len(magnitudes) + i*2 + 1)

        mag_mww.mag = mw
        mag_mww.magnitude_type = "Mww"
        mag_mww.origin_id = origin.resource_id
        mag_mww.method_id = create_resource_id("method/wphase")

        mag_mww.station_count = origin.quality.used_station_count
        mag_mww.azimuthal_gap = origin.quality.azimuthal_gap
        mag_mww.creation_info = origin.creation_info
        mag_mww.resource_id = create_resource_id(f"magnitude/mww/{event_id}/{i}")


        fm.creation_info = origin.creation_info

        # Calculate misfit
        fm.misfit = calculate_misfit(fm, [])  # We don't have picks here, so passing an empty list

        # Associate the origin with this focal mechanism
        fm.triggering_origin_id = origin.resource_id

        print(f"Created Mw magnitude: {mag_mw.mag}, associated with origin: {mag_mw.origin_id}")
        print(f"Created Mww magnitude: {mag_mww.mag}, associated with origin: {mag_mww.origin_id}")


        # Explicitly link the Mww magnitude to the focal mechanism's moment tensor
        fm.moment_tensor.moment_magnitude_id = mag_mww.resource_id

        focal_mechanisms.append((fm, origin))
        magnitudes.extend([mag_mw, mag_mww])

    return focal_mechanisms, magnitudes

def calculate_misfit(focal_mechanism, picks):
    """
    Calculate a simplified misfit for the focal mechanism.
    In a real scenario, this would involve comparing observed and synthetic waveforms.
    """
    # This is a placeholder implementation
    # In a real scenario, you would calculate the misfit based on the difference
    # between observed and synthetic waveforms generated using the focal mechanism
    
    # For now, we'll return a random value as a placeholder
    return np.random.uniform(0, 1)


def calculate_principal_axes(strike, dip, rake):
    """
    Calculate the principal axes (T, P, N) given the strike, dip, and rake.
    """
    s, d, r = np.radians([strike, dip, rake])
    
    # T axis
    t_pl = np.arcsin(np.sin(d) * np.sin(r))
    t_az = np.arctan2(np.cos(r) * np.cos(s) + np.sin(r) * np.sin(d) * np.sin(s),
                      np.cos(d) * np.sin(s))
    
    # P axis
    p_pl = np.arcsin(-np.cos(d) * np.sin(r))
    p_az = np.arctan2(np.sin(r) * np.cos(s) - np.cos(r) * np.sin(d) * np.sin(s),
                      -np.cos(d) * np.cos(s))
    
    # N axis (null axis)
    n_pl = np.arccos(np.cos(d) * np.cos(r))
    n_az = np.arctan2(-np.cos(r) * np.sin(s) - np.sin(r) * np.sin(d) * np.cos(s),
                      np.cos(r) * np.cos(d) * np.cos(s) - np.sin(r) * np.sin(s))
    
    # Convert back to degrees
    return (np.degrees(t_az) % 360, np.degrees(t_pl)), \
           (np.degrees(p_az) % 360, np.degrees(p_pl)), \
           (np.degrees(n_az) % 360, np.degrees(n_pl))


def calculate_auxiliary_plane(strike, dip, rake):
    """
    Calculate the auxiliary plane given the strike, dip, and rake of the primary plane.
    """
    strike, dip, rake = np.radians([strike, dip, rake])
    
    # Calculate auxiliary plane
    strike2 = np.arctan2(
        -np.cos(rake) * np.sin(strike) - np.sin(rake) * np.sin(dip) * np.cos(strike),
        np.cos(rake) * np.cos(strike) - np.sin(rake) * np.sin(dip) * np.sin(strike)
    )
    strike2 = np.degrees(strike2) % 360

    dip2 = np.arcsin(np.sin(rake) * np.sin(dip))
    dip2 = np.degrees(dip2)

    rake2 = np.arctan2(
        -np.cos(dip) * np.sin(rake),
        -np.sin(dip) * np.cos(rake) * np.sin(strike - np.radians(strike2))
    )
    rake2 = np.degrees(rake2)

    return strike2, dip2, rake2


def create_moment_tensor(scalar_moment, strike, dip, rake, event_id):
    """
    Create a MomentTensor object given the scalar moment and fault plane solution.
    """
    mt = MomentTensor()
    mt.resource_id = create_resource_id(f"momenttensor/{event_id}")
    mt.scalar_moment = scalar_moment
    
    # Convert to radians
    s, d, r = np.radians([strike, dip, rake])
    
    # Calculate moment tensor components
    mrr = scalar_moment * (np.sin(2*d) * np.sin(r))
    mtt = -scalar_moment * (np.sin(d) * np.cos(r) * np.sin(2*s) + np.sin(2*d) * np.sin(r) * np.sin(s)**2)
    mpp = scalar_moment * (np.sin(d) * np.cos(r) * np.sin(2*s) - np.sin(2*d) * np.sin(r) * np.cos(s)**2)
    mrt = -scalar_moment * (np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2*d) * np.sin(r) * np.sin(s))
    mrp = scalar_moment * (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2*d) * np.sin(r) * np.cos(s))
    mtp = -scalar_moment * (np.sin(d) * np.cos(r) * np.cos(2*s) + 0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
    
    mt.tensor = Tensor(m_rr=mrr, m_tt=mtt, m_pp=mpp, m_rt=mrt, m_rp=mrp, m_tp=mtp)
    
    # Set source time function (simplified)
    duration = 2.0  # Example duration in seconds
    mt.source_time_function = SourceTimeFunction(type="triangle", duration=duration)
    
    return mt

def calculate_tensor_components(scalar_moment, strike, dip, rake):
    # Convert to radians
    s, d, r = np.radians([strike, dip, rake])
    
    # Calculate moment tensor components
    mrr = scalar_moment * (np.sin(2*d) * np.sin(r))
    mtt = -scalar_moment * (np.sin(d) * np.cos(r) * np.sin(2*s) + np.sin(2*d) * np.sin(r) * np.sin(s)**2)
    mpp = scalar_moment * (np.sin(d) * np.cos(r) * np.sin(2*s) - np.sin(2*d) * np.sin(r) * np.cos(s)**2)
    mrt = -scalar_moment * (np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2*d) * np.sin(r) * np.sin(s))
    mrp = scalar_moment * (np.cos(d) * np.cos(r) * np.sin(s) - np.cos(2*d) * np.sin(r) * np.cos(s))
    mtp = -scalar_moment * (np.sin(d) * np.cos(r) * np.cos(2*s) + 0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
    
    return Tensor(m_rr=mrr, m_tt=mtt, m_pp=mpp, m_rt=mrt, m_rp=mrp, m_tp=mtp)

def calculate_rupture_duration(scalar_moment):
    # Simple estimation based on scalar moment
    magnitude = (2/3) * (np.log10(scalar_moment) - 9.1)  # Moment magnitude
    return 10**(0.3 * magnitude - 0.774)  # Empirical relation for rupture duration

def create_moment_magnitude(scalar_moment, origin_id, event_id, station_count, magnitude_index):
    mw = (2/3) * (np.log10(scalar_moment) - 9.1)
    mag = Magnitude()
    mag.mag = mw
    mag.magnitude_type = "Mw"
    mag.origin_id = origin_id
    mag.resource_id = create_resource_id(f"magnitude/mw/{event_id}/{magnitude_index}")
    mag.method_id = create_resource_id("method/mw")
    mag.station_count = station_count
    mag.creation_info = CreationInfo(agency_id=config['Agency']['id'], author=f"{config['Agency']['author_prefix']}automag@testtest", creation_time=UTCDateTime())
    return mag

def filter_stations(inventory, event_lat, event_lon, min_distance, max_distance):
    filtered_stations = []
    for network in inventory:
        for station in network:
            distance = locations2degrees(event_lat, event_lon, station.latitude, station.longitude)
            if min_distance <= distance <= max_distance:
                azimuth = calculate_azimuth(event_lat, event_lon, station.latitude, station.longitude)
                channels = [chan for chan in station.channels if chan.code in ['BHZ', 'HHZ', 'SHZ']]
                if channels:
                    filtered_stations.append({
                        'code': station.code,
                        'network': network.code,
                        'latitude': station.latitude,
                        'longitude': station.longitude,
                        'distance': distance,
                        'azimuth': azimuth,
                        'channels': channels
                    })
    
    # Sort stations by distance
    filtered_stations.sort(key=lambda x: x['distance'])
    return filtered_stations


def calculate_azimuth(lat1, lon1, lat2, lon2):
    # Simple azimuth calculation (not accurate for large distances)
    dlon = np.radians(lon2 - lon1)
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth = np.degrees(np.arctan2(y, x))
    return (azimuth + 360) % 360

def create_synthetic_event_with_multiple_origins(lat, lon, depth, initial_time, stations, initial_magnitudes, focal_mechanism):
    event = Event()
    event.event_type = "earthquake"
    
    # Generate the event ID
    agency_id = config['Agency']['id_lowercase']
    event_id = generate_event_id(agency_id, initial_time.datetime)
    event.resource_id = event_id

    number_of_origins = get_config_int('MultipleOrigins', 'number_of_origins', 10)
    initial_station_count = get_config_int('MultipleOrigins', 'initial_station_count', 10)
    station_increase = get_config_int('MultipleOrigins', 'station_increase_per_origin', 5)
    time_between_origins = get_config_float('MultipleOrigins', 'time_between_origins', 60)

    true_origin_time = initial_time
    origin_time_uncertainty = get_config_float('Uncertainties', 'origin_time', 0.1)

    for i in range(number_of_origins):
        current_station_count = min(initial_station_count + i * station_increase, len(stations))
        current_stations = stations[:current_station_count]

        current_time = true_origin_time + np.random.normal(0, origin_time_uncertainty)
        current_magnitudes = {k: v + np.random.uniform(-0.2, 0.2) for k, v in initial_magnitudes.items()}

        origin = create_origin(lat, lon, depth, current_time, current_stations, i, number_of_origins)
        event.origins.append(origin)

        create_picks_and_arrivals(event, origin, current_stations, depth)
        create_magnitudes(event, origin, current_magnitudes, current_stations, event_id)

        if i == number_of_origins - 1:  # For the last origin
            print("Creating focal mechanisms and associated Mw and Mww magnitudes")
            largest_mag = max(mag.mag for mag in event.magnitudes)
            scalar_moment = 10**(1.5 * largest_mag + 9.1)

            focal_mechanisms_with_origins, mw_mww_magnitudes = create_focal_mechanisms(
                lat, lon, depth, current_time, scalar_moment, event_id,
                focal_mechanism['strike1'], focal_mechanism['dip1'], focal_mechanism['rake1'],
                num_mechanisms=3
            )

            print(f"Created {len(focal_mechanisms_with_origins)} focal mechanisms and {len(mw_mww_magnitudes)} magnitudes (Mw and Mww)")

            for fm, origin in focal_mechanisms_with_origins:
                event.focal_mechanisms.append(fm)
                event.origins.append(origin)
                print(f"Added focal mechanism {fm.resource_id} and origin {origin.resource_id} to event")
                print(f"Moment tensor linked to origin: {fm.moment_tensor.derived_origin_id}")

            # Add Mw and Mww magnitudes to the event
            event.magnitudes.extend(mw_mww_magnitudes)
            print(f"Added {len(mw_mww_magnitudes)} magnitudes (Mw and Mww) to event")

            # Use the focal mechanism with the lowest misfit as the preferred one
            preferred_fm, preferred_origin = min(focal_mechanisms_with_origins, key=lambda x: x[0].misfit)
            event.preferred_focal_mechanism_id = preferred_fm.resource_id
            event.preferred_origin_id = preferred_origin.resource_id

            # Set the preferred magnitude to the Mww magnitude associated with the preferred focal mechanism
            preferred_mww = next(mag for mag in mw_mww_magnitudes if mag.origin_id == preferred_origin.resource_id and mag.magnitude_type == "Mww")
            event.preferred_magnitude_id = preferred_mww.resource_id

            print(f"Set preferred focal mechanism: {event.preferred_focal_mechanism_id}")
            print(f"Set preferred origin: {event.preferred_origin_id}")
            print(f"Set preferred magnitude (Mww): {event.preferred_magnitude_id}")

    return event

def create_origin(lat, lon, depth, time, stations, origin_index=0, total_origins=10):
    origin = Origin()
    origin.time = time
    origin.latitude = lat
    origin.longitude = lon
    origin.depth = depth * 1000  # Convert to meters
    origin.depth_type = "from location"
    
    # Decrease time uncertainty as we get more origins
    time_uncertainty = get_config_float('Uncertainties', 'origin_time', 0.1) * (1 - origin_index / total_origins)
    origin.time_errors = QuantityError(uncertainty=time_uncertainty)
    
    # Decrease spatial uncertainties as we get more origins
    lat_lon_uncertainty = get_config_float('Uncertainties', 'origin_latitude', 0.01) * (1 - origin_index / total_origins)
    origin.latitude_errors = QuantityError(uncertainty=lat_lon_uncertainty)
    origin.longitude_errors = QuantityError(uncertainty=lat_lon_uncertainty)
    
    depth_uncertainty = get_config_float('Uncertainties', 'origin_depth', 5000) * (1 - origin_index / total_origins)
    origin.depth_errors = QuantityError(uncertainty=depth_uncertainty)
    
    origin.origin_uncertainty = OriginUncertainty(
        min_horizontal_uncertainty=get_config_float('Uncertainties', 'origin_min_horizontal', 5000) * (1 - origin_index / total_origins),
        max_horizontal_uncertainty=get_config_float('Uncertainties', 'origin_max_horizontal', 10000) * (1 - origin_index / total_origins),
        azimuth_max_horizontal_uncertainty=90,
        preferred_description="uncertainty ellipse"
    )
    origin.quality = OriginQuality(
        associated_phase_count=len(stations) * 2,
        used_phase_count=len(stations) * 2,
        associated_station_count=len(stations),
        used_station_count=len(stations),
        depth_phase_count=0,
        standard_error=0.5 * (1 - origin_index / total_origins),
        azimuthal_gap=calculate_azimuthal_gap(stations),
        minimum_distance=min(s['distance'] for s in stations),
        maximum_distance=max(s['distance'] for s in stations),
        median_distance=np.median([s['distance'] for s in stations])
    )
    origin.evaluation_mode = "manual"
    origin.evaluation_status = "confirmed"
    origin.creation_info = CreationInfo(
        agency_id=config['Agency']['id'],
        author=f"{config['Agency']['author_prefix']}olv@testtest",
        creation_time=UTCDateTime()
    )
    return origin


def create_picks_and_arrivals(event, origin, stations, depth):
    model = TauPyModel(model="iasp91")
    s_wave_cutoff = get_config_float('Phases', 's_wave_cutoff', 15)
    
    for station in stations:
        distance = station['distance']
        azimuth = station['azimuth']
        
        phases_to_pick = ['P', 'S', 'PKP'] if distance <= s_wave_cutoff else ['P']
        
        for phase in phases_to_pick:
            arrivals = model.get_travel_times(source_depth_in_km=depth,
                                              distance_in_degree=distance,
                                              phase_list=[phase])
            if arrivals:
                theoretical_time = origin.time + arrivals[0].time
                noise = np.random.normal(0, get_config_float('Noise', 'pick_time_std', 0.5))
                arrival_time = theoretical_time + noise
                
                # Randomly choose a channel for this pick
                channel = random.choice(station['channels'])
                
                waveform_id = WaveformStreamID(network_code=station['network'],
                                               station_code=station['code'],
                                               channel_code=channel.code)
                
                pick = create_pick(arrival_time, waveform_id, phase)
                event.picks.append(pick)
                
                time_residual = arrival_time - theoretical_time
                arrival = create_arrival(pick.resource_id, phase, azimuth, distance, time_residual, 1)
                origin.arrivals.append(arrival)

                # Create amplitude (simplified)
                amplitude = Amplitude()
                amplitude.generic_amplitude = np.random.uniform(1e-6, 1e-4)
                amplitude.period = np.random.uniform(0.5, 2.0)
                amplitude.pick_id = pick.resource_id
                amplitude.waveform_id = waveform_id
                event.amplitudes.append(amplitude)

def calculate_azimuthal_gap(stations):
    azimuths = sorted([s['azimuth'] for s in stations])
    gaps = np.diff(azimuths)
    gaps = np.append(gaps, 360 + azimuths[0] - azimuths[-1])
    return np.max(gaps)

def generate_synthetic_catalog_with_multiple_origins(lat, lon, depth, time, stations, magnitudes, focal_mechanism):
    event = create_synthetic_event_with_multiple_origins(lat, lon, depth, time, stations, magnitudes, focal_mechanism)
    catalog = Catalog([event])
    catalog.resource_id = create_resource_id("catalog/synthetic")
    catalog.creation_info = CreationInfo(agency_id="GA", author="ObsPy QuakeML Generator", creation_time=UTCDateTime())
    return catalog, event

def main():
    # Event parameters
    lat = get_config_float('Event', 'latitude', -7.69)
    lon = get_config_float('Event', 'longitude', 125.88)
    depth = get_config_float('Event', 'depth', 10)
    time = UTCDateTime(config['Event']['time']) if config['Event']['time'] != 'now' else UTCDateTime.now()
    
    # Specify magnitudes and focal mechanism
    magnitudes = {k: get_config_float('Magnitudes', k, 0) for k in config['Magnitudes']}
    focal_mechanism = {
        f"{key}{i}": get_config_float('FocalMechanism', f'{key}{i}', 0)
        for key in ['strike', 'dip', 'rake'] for i in [1, 2]
    }
    
    # Configuration for station filtering
    inventory_path = config['Inventory']['path']
    min_distance = get_config_float('Inventory', 'min_distance', 0)
    max_distance = get_config_float('Inventory', 'max_distance', 50)
    
    # Read inventory and filter stations
    inventory = read_inventory(inventory_path)
    stations = filter_stations(inventory, lat, lon, min_distance, max_distance)
    
    if not stations:
        print(f"Error: No stations selected after filtering. Check criteria: min_distance={min_distance}, max_distance={max_distance}")
        sys.exit(1)
    
    # Generate catalog with multiple origins
    catalog, event = generate_synthetic_catalog_with_multiple_origins(lat, lon, depth, time, stations, magnitudes, focal_mechanism)
    
    # Write to file
    output_file = "synthetic_event_multiple_origins.xml"
    catalog.write(output_file, format="SC3ML")
    print(f"QuakeML file has been written to {output_file}")

    print_event_summary(event, stations)
    print_network_summary(stations)
    print_focal_mechanism_summary(event.focal_mechanisms[0])
    print_magnitude_summary(event.magnitudes)
    print_origin_times(event.origins)

def print_event_summary(event, stations):
    print("\nGenerated Event Summary:")
    print(f"Number of Origins: {len(event.origins)}")
    print(f"Initial Origin Time: {event.origins[0].time}")
    print(f"Final Origin Time: {event.preferred_origin().time}")
    print(f"Latitude: {event.preferred_origin().latitude:.4f}")
    print(f"Longitude: {event.preferred_origin().longitude:.4f}")
    print(f"Depth: {event.preferred_origin().depth / 1000:.2f} km")
    print(f"Final Magnitude: {event.preferred_magnitude().mag:.2f} {event.preferred_magnitude().magnitude_type}")
    print(f"Total Number of Stations: {len(stations)}")
    print(f"Total Number of Picks: {len(event.picks)}")
    print(f"Total origins: {len(event.origins)}")
    print(f"Total focal mechanisms: {len(event.focal_mechanisms)}")
    print(f"Total magnitudes: {len(event.magnitudes)}")
    print(f"Final Azimuthal Gap: {event.preferred_origin().quality.azimuthal_gap:.2f} degrees")

def print_network_summary(stations):
    print("\nUsed Networks and Stations:")
    network_station_count = {}
    for station in stations:
        network_station_count[station['network']] = network_station_count.get(station['network'], 0) + 1
    
    for network, count in network_station_count.items():
        print(f"Network {network}: {count} stations")
    
    print(f"\nTotal number of stations: {len(stations)}")
    print(f"Number of networks: {len(network_station_count)}")

def print_focal_mechanism_summary(fm):
    if fm:
        print(f"\nFocal Mechanism:")
        print(f"Nodal Plane 1 (Strike, Dip, Rake): {fm.nodal_planes.nodal_plane_1.strike:.2f}, "
              f"{fm.nodal_planes.nodal_plane_1.dip:.2f}, {fm.nodal_planes.nodal_plane_1.rake:.2f}")
        print(f"Nodal Plane 2 (Strike, Dip, Rake): {fm.nodal_planes.nodal_plane_2.strike:.2f}, "
              f"{fm.nodal_planes.nodal_plane_2.dip:.2f}, {fm.nodal_planes.nodal_plane_2.rake:.2f}")
        print(f"Moment Magnitude (Mw): {fm.moment_tensor.moment_magnitude_id}")
        print(f"Scalar Moment: {fm.moment_tensor.scalar_moment:.2e} N-m")
        
        mt = fm.moment_tensor.tensor
        print("\nMoment Tensor Components:")
        print(f"Mrr: {mt.m_rr:.2e}, Mtt: {mt.m_tt:.2e}, Mpp: {mt.m_pp:.2e}")
        print(f"Mrt: {mt.m_rt:.2e}, Mrp: {mt.m_rp:.2e}, Mtp: {mt.m_tp:.2e}")

def print_magnitude_summary(magnitudes):
    print("\nAll Magnitudes:")
    for mag in magnitudes:
        print(f"{mag.magnitude_type}: {mag.mag:.2f}")

def print_origin_times(origins):
    print("\nOrigin Times:")
    for i, origin in enumerate(origins, 1):
        print(f"Origin {i}: {origin.time}")

if __name__ == "__main__":
    main()