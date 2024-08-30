#!/usr/bin/env seiscomp-python

import sys
import random
import numpy as np
from seiscomp import core, datamodel, io, math
import configparser
from glob import glob

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def create_synthetic_event(config):
    ep = datamodel.EventParameters()

    # Create Event
    event = datamodel.Event.Create("synthetic_event")
    ep.add(event)
    event.setType(datamodel.EARTHQUAKE)

    # Set creation info
    creation_info = datamodel.CreationInfo()
    creation_info.setAuthor(f"{config['Agency']['author_prefix']}@{config['Agency']['id']}")
    creation_info.setAgencyID(config['Agency']['id'])
    creation_info.setCreationTime(core.Time.GMT())
    event.setCreationInfo(creation_info)

    # Create multiple origins
    origins = create_multiple_origins(config)
    for origin in origins:
        ep.add(origin)
        event.add(datamodel.OriginReference(origin.publicID()))
        
        # Create magnitudes for each origin
        mags = create_magnitudes(origin, config)
        for mag in mags:
            origin.add(mag)

    # Set preferred origin (last one)
    preferred_origin = origins[-1]
    event.setPreferredOriginID(preferred_origin.publicID())

    # Set preferred magnitude (Mww from the preferred origin)
    mww_mag = None
    for i in range(preferred_origin.magnitudeCount()):
        mag = preferred_origin.magnitude(i)
        if mag.type() == "Mww":
            mww_mag = mag
            break
    if mww_mag:
        event.setPreferredMagnitudeID(mww_mag.publicID())

    # Create a separate origin for focal mechanism
    fm_origin = create_focal_mechanism_origin(config, preferred_origin)
    ep.add(fm_origin)
    event.add(datamodel.OriginReference(fm_origin.publicID()))

    # Create focal mechanism and moment tensor
    fm, mt = create_focal_mechanism(fm_origin, config)
    ep.add(fm)
    event.setPreferredFocalMechanismID(fm.publicID())

    return ep

def create_multiple_origins(config):
    origins = []
    lat = float(config['Event']['latitude'])
    lon = float(config['Event']['longitude'])
    depth_km = float(config['Event']['depth'])
    if config['Event']['time'].lower() == 'now':
        origin_time = core.Time.GMT()
    else:
        origin_time = core.Time.FromString(config['Event']['time'])
    
    num_origins = int(config['MultipleOrigins']['number_of_origins'])
    initial_station_count = int(config['MultipleOrigins']['initial_station_count'])
    station_increase = int(config['MultipleOrigins']['station_increase_per_origin'])
    creation_time_increment = float(config['MultipleOrigins'].get('creation_time_increment', '60'))

    base_creation_time = core.Time.GMT()

    for i in range(num_origins):
        origin = datamodel.Origin.Create()
        
        # Set latitude with uncertainty
        lat_value = lat + np.random.normal(0, float(config['Uncertainties']['origin_latitude']))
        lat_uncertainty = float(config['Uncertainties']['origin_latitude']) * (1 - i/num_origins)
        origin.setLatitude(datamodel.RealQuantity(lat_value, lat_uncertainty))

        # Set longitude with uncertainty
        lon_value = lon + np.random.normal(0, float(config['Uncertainties']['origin_longitude']))
        lon_uncertainty = float(config['Uncertainties']['origin_longitude']) * (1 - i/num_origins)
        origin.setLongitude(datamodel.RealQuantity(lon_value, lon_uncertainty))

        # Set depth with uncertainty (in kilometers)
        depth_value = depth_km + np.random.normal(0, float(config['Uncertainties']['origin_depth']) / 1000)
        depth_uncertainty = float(config['Uncertainties']['origin_depth']) / 1000 * (1 - i/num_origins)
        origin.setDepth(datamodel.RealQuantity(depth_value, depth_uncertainty))
        
        # Set origin time (constant for all origins)
        origin.setTime(datamodel.TimeQuantity(origin_time))

        # Set creation time (incremental)
        creation_time = base_creation_time + core.TimeSpan(i * creation_time_increment)
        creation_info = datamodel.CreationInfo()
        creation_info.setCreationTime(creation_time)
        creation_info.setAgencyID(config['Agency']['id'])
        creation_info.setAuthor(f"{config['Agency']['author_prefix']}@{config['Agency']['id']}")
        origin.setCreationInfo(creation_info)

        # Set evaluation properties
        origin.setEvaluationMode(datamodel.MANUAL)
        origin.setEvaluationStatus(datamodel.CONFIRMED)

        # Add arrivals (simplified)
        station_count = min(initial_station_count + i * station_increase, 500)  # Cap at 500 stations
        for _ in range(station_count):
            arr = datamodel.Arrival()
            arr.setPhase(datamodel.Phase("P"))
            arr.setDistance(random.uniform(0, 180))
            arr.setAzimuth(random.uniform(0, 360))
            arr.setTimeResidual(random.uniform(-1, 1))
            arr.setWeight(1.0)
            origin.add(arr)

        origins.append(origin)

    return origins

def create_magnitudes(origin, config):
    mags = []
    for mag_type, mag_value in config['Magnitudes'].items():
        mag = datamodel.Magnitude.Create()
        mag.setMagnitude(datamodel.RealQuantity(float(mag_value) + np.random.normal(0, float(config['Noise']['station_magnitude_std']))))
        mag.setType(mag_type.upper())
        mag.setOriginID(origin.publicID())
        mag.setStationCount(random.randint(10, 50))
        mags.append(mag)

    # Add Mww magnitude
    mw_value = float(config['Magnitudes'].get('mwp', 6.0))  # Use MwP as a basis, or default to 6.0
    mww = datamodel.Magnitude.Create()
    mww.setMagnitude(datamodel.RealQuantity(mw_value + np.random.normal(0, 0.1)))
    mww.setType("Mww")
    mww.setOriginID(origin.publicID())
    mww.setMethodID("wphase")
    mww.setStationCount(random.randint(20, 100))
    mags.append(mww)

    return mags

def create_focal_mechanism_origin(config, preferred_origin):
    fm_origin = datamodel.Origin.Create()
    fm_origin.setLatitude(preferred_origin.latitude())
    fm_origin.setLongitude(preferred_origin.longitude())
    fm_origin.setDepth(preferred_origin.depth())
    fm_origin.setTime(preferred_origin.time())
    
    fm_origin.setEvaluationMode(datamodel.MANUAL)
    fm_origin.setEvaluationStatus(datamodel.CONFIRMED)

    # Set creation info
    creation_info = datamodel.CreationInfo()
    creation_info.setAuthor(f"{config['Agency']['author_prefix']}@{config['Agency']['id']}")
    creation_info.setAgencyID(config['Agency']['id'])
    creation_info.setCreationTime(core.Time.GMT())
    fm_origin.setCreationInfo(creation_info)

    return fm_origin

def create_focal_mechanism(origin, config):
    fm = datamodel.FocalMechanism.Create()
    fm.setTriggeringOriginID(origin.publicID())

    # Set nodal planes
    np = datamodel.NodalPlanes()
    np1 = datamodel.NodalPlane()
    np1.setStrike(datamodel.RealQuantity(float(config['FocalMechanism']['strike1'])))
    np1.setDip(datamodel.RealQuantity(float(config['FocalMechanism']['dip1'])))
    np1.setRake(datamodel.RealQuantity(float(config['FocalMechanism']['rake1'])))
    np.setNodalPlane1(np1)

    np2 = datamodel.NodalPlane()
    np2.setStrike(datamodel.RealQuantity(float(config['FocalMechanism']['strike2'])))
    np2.setDip(datamodel.RealQuantity(float(config['FocalMechanism']['dip2'])))
    np2.setRake(datamodel.RealQuantity(float(config['FocalMechanism']['rake2'])))
    np.setNodalPlane2(np2)

    fm.setNodalPlanes(np)

    # Create moment tensor
    mt = create_moment_tensor(origin, config)
    fm.add(mt)

    return fm, mt

def create_moment_tensor(origin, config):
    mt = datamodel.MomentTensor.Create()
    mt.setDerivedOriginID(origin.publicID())

    # Calculate scalar moment based on Mw
    mw = float(config['Magnitudes'].get('mwp', 6.0))
    scalar_moment = 10 ** (1.5 * mw + 9.1)

    # Set strike, dip, rake
    strike = float(config['FocalMechanism']['strike1'])
    dip = float(config['FocalMechanism']['dip1'])
    rake = float(config['FocalMechanism']['rake1'])

    # Calculate moment tensor components
    tensor = calculate_tensor_components(scalar_moment, strike, dip, rake)
    mt.setTensor(tensor)

    # Set scalar moment
    mt.setScalarMoment(datamodel.RealQuantity(scalar_moment))

    # Set double couple, ISO, CLVD components (simplified)
    mt.setDoubleCouple(0.8)
    mt.setIso(0.1)
    mt.setClvd(0.1)

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

    tensor = datamodel.Tensor()
    tensor.setMrr(datamodel.RealQuantity(mrr))
    tensor.setMtt(datamodel.RealQuantity(mtt))
    tensor.setMpp(datamodel.RealQuantity(mpp))
    tensor.setMrt(datamodel.RealQuantity(mrt))
    tensor.setMrp(datamodel.RealQuantity(mrp))
    tensor.setMtp(datamodel.RealQuantity(mtp))

    return tensor

def write_to_xml(ep, filename):
    ar = io.XMLArchive()
    if not ar.create(filename):
        print(f"Could not create file: {filename}")
        return False
    ar.setFormattedOutput(True)
    ar.writeObject(ep)
    ar.close()
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py config.ini")
        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    try:
        ep = create_synthetic_event(config)
        print("Successfully created synthetic event")

        # Write to file
        output_file = "synthetic_event_seiscomp.xml"
        if write_to_xml(ep, output_file):
            print(f"Event has been written to {output_file}")
        else:
            print("Failed to write event to file")

        # Print event information
        for i in range(ep.eventCount()):
            event = ep.event(i)
            print(f"\nEvent {i+1} ID: {event.publicID()}")
            
            print("Origins:")
            for j in range(ep.originCount()):
                origin = ep.origin(j)
                print(f"  Origin ID: {origin.publicID()}")
                print(f"    Latitude: {origin.latitude().value():.4f}")
                print(f"    Longitude: {origin.longitude().value():.4f}")
                print(f"    Depth: {origin.depth().value():.2f} km")
                print(f"    Time: {origin.time().value().toString('%Y-%m-%d %H:%M:%S.%f')}")
                print("    Magnitudes:")
                for k in range(origin.magnitudeCount()):
                    magnitude = origin.magnitude(k)
                    print(f"      Type: {magnitude.type()}, Value: {magnitude.magnitude().value():.2f}")
                if origin.publicID() == event.preferredOriginID():
                    print("    (Preferred Origin)")
                print()

            if event.preferredFocalMechanismID():
                preferred_fm = None
                for j in range(ep.focalMechanismCount()):
                    if ep.focalMechanism(j).publicID() == event.preferredFocalMechanismID():
                        preferred_fm = ep.focalMechanism(j)
                        break
                if preferred_fm:
                    print(f"Focal Mechanism:")
                    print(f"  ID: {preferred_fm.publicID()}")
                    print(f"  Triggering Origin ID: {preferred_fm.triggeringOriginID()}")
                    
                    if preferred_fm.momentTensorCount() > 0:
                        mt = preferred_fm.momentTensor(0)
                        print("Moment Tensor:")
                        tensor = mt.tensor()
                        print(f"  Mrr: {tensor.Mrr().value():.2e}")
                        print(f"  Mtt: {tensor.Mtt().value():.2e}")
                        print(f"  Mpp: {tensor.Mpp().value():.2e}")
                        print(f"  Mrt: {tensor.Mrt().value():.2e}")
                        print(f"  Mrp: {tensor.Mrp().value():.2e}")
                        print(f"  Mtp: {tensor.Mtp().value():.2e}")
                        print(f"  Scalar Moment: {mt.scalarMoment().value():.2e}")
                        print(f"  Double Couple: {mt.doubleCouple():.2f}")
                        print(f"  CLVD: {mt.clvd():.2f}")
                        print(f"  ISO: {mt.iso():.2f}")
                        print(f"  Derived Origin ID: {mt.derivedOriginID()}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()