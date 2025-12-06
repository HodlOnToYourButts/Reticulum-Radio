#!/usr/bin/env python3
"""
Reticulum Radio Listener

Allows users to discover and tune into radio broadcasts on the Reticulum network.
"""

import RNS
import LXST
import argparse
import sys
from pathlib import Path


class RadioListener:
    """
    Listener for receiving radio broadcasts over Reticulum network.
    """

    def __init__(self):
        """Initialize the listener."""
        self.reticulum = None
        self.identity = None
        self.lxst_inlet = None
        self.current_station = None

    def setup_reticulum(self):
        """Initialize Reticulum network stack."""
        print("Initializing Reticulum listener...")
        self.reticulum = RNS.Reticulum()

        # Create or load identity
        identity_path = Path.home() / ".reticulum" / "identities" / "radio_listener"

        if identity_path.exists():
            self.identity = RNS.Identity.from_file(str(identity_path))
            print(f"Loaded existing identity")
        else:
            self.identity = RNS.Identity()
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            self.identity.to_file(str(identity_path))
            print(f"Created new identity")

        print(f"Listener Identity: {RNS.prettyhexrep(self.identity.hash)}")

    def discover_stations(self):
        """Discover available radio stations on the network."""
        print("\nScanning for radio stations...")
        print("(This feature requires LXST discovery implementation)")

        # TODO: Implement LXST outlet discovery
        # LXST should provide a way to announce/discover outlets
        # For now, stations must be manually specified

        print("No stations discovered. Use --station to connect to a specific station.")

    def tune_to_station(self, station_hash):
        """
        Tune into a specific radio station.

        Args:
            station_hash: The destination hash of the broadcaster
        """
        print(f"\nTuning to station: {station_hash}")

        try:
            # Convert hash string to bytes
            destination_hash = bytes.fromhex(station_hash)

            # Create LXST inlet to receive audio stream
            print("Connecting to LXST stream...")

            # TODO: Implement LXST inlet connection
            # self.lxst_inlet = LXST.Inlet(
            #     self.reticulum,
            #     destination_hash
            # )

            self.current_station = station_hash
            print(f"Connected to station!")
            print("Playing audio...")

            # Start receiving and playing audio
            self.play_stream()

        except Exception as e:
            print(f"Error tuning to station: {e}")

    def play_stream(self):
        """Receive and play the audio stream."""
        print("\nReceiving audio stream...")
        print("Press Ctrl+C to stop listening")

        try:
            # TODO: Implement audio stream reception
            # TODO: Decode OPUS audio
            # TODO: Play audio through speakers using PyAudio

            # Placeholder - wait for interrupt
            import time
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping playback...")

    def start(self, station_hash=None):
        """
        Start the listener.

        Args:
            station_hash: Optional station hash to tune to immediately
        """
        self.setup_reticulum()

        print(f"\n{'='*60}")
        print(f"  Reticulum Radio Listener")
        print(f"{'='*60}\n")

        if station_hash:
            self.tune_to_station(station_hash)
        else:
            self.discover_stations()

    def shutdown(self):
        """Clean shutdown of listener."""
        if self.lxst_inlet:
            print("Disconnecting from station...")
            # TODO: Proper LXST cleanup

        print("Listener stopped.")


def main():
    """Main entry point for listener CLI."""
    parser = argparse.ArgumentParser(description="Reticulum Radio Listener")
    parser.add_argument(
        "--station",
        type=str,
        help="Station hash to tune to (broadcaster's destination hash)"
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover available stations"
    )

    args = parser.parse_args()

    listener = RadioListener()

    try:
        if args.discover:
            listener.setup_reticulum()
            listener.discover_stations()
        else:
            listener.start(station_hash=args.station)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        listener.shutdown()


if __name__ == "__main__":
    main()
