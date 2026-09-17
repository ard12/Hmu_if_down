# Place compiled ESP32 firmware binaries here for OTA distribution.
#
# The hub serves files in this directory via:
#   GET /firmware/<filename>.bin
#
# Build firmware with:
#   cd firmware/wifi_csi/tracker_node && idf.py build
#   cp build/tracker_node.bin <project_root>/deploy/firmware/tracker_v3.0.0.bin
#
# ESP32 OTA URL (resolved via mDNS):
#   http://falldetect-hub.local/firmware/tracker_v3.0.0.bin
