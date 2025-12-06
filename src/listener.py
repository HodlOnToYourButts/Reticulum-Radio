#!/usr/bin/env python3
"""
Reticulum Radio Listener

Allows users to discover and tune into radio broadcasts on the Reticulum network.
"""

import RNS
import LXST
import argparse
import sys
import time
from pathlib import Path

from LXST import Pipeline
from LXST.Sinks import LineSink
from LXST.Network import LinkSource
from LXST.Codecs import Opus

APP_NAME = "reticulumradio"
BROADCAST_ASPECT = "broadcast"


class RadioListener:
    """
    Listener for receiving radio broadcasts over Reticulum network.
    """

    def __init__(self):
        """Initialize the listener."""
        self.reticulum = None
        self.identity = None
        self.current_station = None

        # LXST components
        self.link = None
        self.link_source = None
        self.speaker_sink = None
        self.pipeline = None

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
        print("Listening for broadcast announcements...")
        print("(Stations announce every 5 minutes)")
        print("\nPress Ctrl+C to stop scanning\n")

        def announce_handler(destination_hash, announced_identity, app_data):
            """Handle incoming destination announcements."""
            # Check if this is a radio broadcast destination
            aspect = RNS.Destination.hash_from_name_and_identity(
                f"{APP_NAME}.{BROADCAST_ASPECT}",
                announced_identity
            )

            station_hash = RNS.prettyhexrep(destination_hash)
            print(f"Found station: {station_hash}")

        # Register announce handler
        RNS.Transport.register_announce_handler(announce_handler)

        try:
            # Wait for announcements
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped scanning.")

    def tune_to_station(self, station_hash):
        """
        Tune into a specific radio station.

        Args:
            station_hash: The destination hash of the broadcaster
        """
        print(f"\nTuning to station: {station_hash}")

        try:
            # Convert hash string to bytes (remove any spaces or formatting)
            destination_hash = bytes.fromhex(station_hash.replace(" ", "").replace(":", ""))

            # Create destination from hash
            # Note: For broadcasts, we don't establish a link, we just listen for packets
            # However, LXST currently uses Link-based communication
            # So we need to request a link to the broadcaster

            print("Establishing link to broadcaster...")

            # Create link to broadcaster destination
            self.link = RNS.Link(destination_hash)

            # Wait for link to establish
            timeout = time.time() + 10
            while not self.link.status == RNS.Link.ACTIVE:
                time.sleep(0.1)
                if time.time() > timeout:
                    print("Link establishment timed out!")
                    return

            print("Link established!")

            # Set up audio pipeline
            print("Setting up audio playback...")

            # TODO: Create speaker sink
            # self.speaker_sink = LineSink()

            # Create link source to receive audio
            # self.link_source = LinkSource(self.link, None, self.speaker_sink)

            # Start receiving
            # self.link_source.start()

            self.current_station = station_hash
            print(f"Connected to station!")
            print("Playing audio...")

            # Start receiving and playing audio
            self.play_stream()

        except Exception as e:
            print(f"Error tuning to station: {e}")
            import traceback
            traceback.print_exc()

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
        if self.link_source:
            print("Stopping audio stream...")
            self.link_source.stop()

        if self.link and self.link.status == RNS.Link.ACTIVE:
            print("Disconnecting from station...")
            self.link.teardown()

        if self.pipeline:
            print("Stopping audio pipeline...")
            self.pipeline.stop()

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
