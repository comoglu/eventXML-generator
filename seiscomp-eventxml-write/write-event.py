#!/usr/bin/env seiscomp-python

import sys
from seiscomp import core, datamodel, io


def createArtificialEvent():
    ep = datamodel.EventParameters()

    event = datamodel.Event.Create("test2021abcd")
    ep.add(event)
    event.add(datamodel.EventDescription("Test area", datamodel.REGION_NAME))
    event.setType(datamodel.NOT_EXISTING)

    creationInfo = datamodel.CreationInfo()
    creationInfo.setAuthor("Agent Smith")
    creationInfo.setAgencyID("Test")
    event.setCreationInfo(creationInfo)

    origin = datamodel.Origin.Create()
    ep.add(origin)
    event.setPreferredOriginID(origin.publicID())
    origin.setCreationInfo(creationInfo)
    origin.setEvaluationMode(datamodel.MANUAL)
    origin.setEvaluationStatus(datamodel.PRELIMINARY)
    origin.setLatitude(datamodel.RealQuantity(52.38753))
    origin.setLongitude(datamodel.RealQuantity(13.06883))
    origin.setDepth(datamodel.RealQuantity(1.0))
    origin.setTime(datamodel.TimeQuantity(core.Time.GMT()))

    mag = datamodel.Magnitude.Create()
    origin.add(mag)
    event.setPreferredMagnitudeID(mag.publicID())
    mag.setMagnitude(datamodel.RealQuantity(5.0))
    mag.setType("M")
    mag.setStationCount(1)

    return ep


def writeObject(obj, fileName):
    ar = io.XMLArchive()
    if not ar.create(fileName):
        raise OSError(f"could not open file for writing: {fileName}")

    ar.setFormattedOutput(True)
    ar.writeObject(obj)
    ar.close()


def main():
    fileName = sys.argv[1] if len(sys.argv) > 1 else "-"
    writeObject(createArtificialEvent(), fileName)


if __name__ == "__main__":
    main()
