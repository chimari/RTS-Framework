# Dataset Preparation Tools

The RTS Framework does not prescribe how input manifests are created.

Different cameras require different preprocessing tools.

Examples

- build_temperature_index_from_fits.py
- build_temperature_index_imx811.py
- build_temperature_index_qhy.py

Their only responsibility is to generate a CSV conforming to

csv_manifest_specification.md

The internal implementation of these tools is outside the scope of the RTS
Framework.