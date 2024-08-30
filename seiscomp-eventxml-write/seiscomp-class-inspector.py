#!/usr/bin/env seiscomp-python

from seiscomp import datamodel

def inspect_class(cls):
    print(f"Inspecting {cls.__name__}:")
    methods = [method for method in dir(cls) if callable(getattr(cls, method)) and not method.startswith("__")]
    for method in methods:
        print(f"  - {method}")

def main():
    inspect_class(datamodel.FocalMechanism)
    
    # Also inspect MomentTensor class
    print("\n")
    inspect_class(datamodel.MomentTensor)

if __name__ == "__main__":
    main()
