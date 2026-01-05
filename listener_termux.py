#!/usr/bin/env python3
"""
Reticulum Radio Listener for Termux/Android

Simplified listener that works without LXST by using sounddevice directly.
Designed for LoRa radio reception on Android devices via Termux.

Usage:
    python listener_termux.py --station <control_dest_hash> [--buffer 60]

Requirements:
    pip install rns encodec numpy sounddevice

Note: Does NOT require LXST (which can't install on Termux due to codec2 dependency)
"""

import argparse
import sys
import time
import threading
from pathlib import Path
from collections import deque

import numpy as np
import RNS
import RNS.vendor.umsgpack as umsgpack

# Audio playback
import sounddevice as sd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our codec (Termux version - no LXST dependency)
from encodec_codec_termux import EnCodecStreamable

# Constants
APP_NAME = "reticulum_radio"


class AudioPlayer:
    """
    Simple audio player using sounddevice.
    Replaces LXST's BufferedLineSink for Termux compatibility.
    """

    def __init__(self, sample_rate=48000, channels=1, buffer_seconds=60):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds

        # Audio buffer (deque of frames)
        self.buffer = deque()
        self.buffer_lock = threading.Lock()

        # Playback state
        self.is_playing = False
        self.stream = None

        # Buffer management
        self.frames_received = 0
        self.frames_played = 0
        self.autostart_threshold = 8  # Start after 8 frames (~40s for 5s frames)

        print(f"[AudioPlayer] Initialized: {sample_rate}Hz, {channels}ch, {buffer_seconds}s buffer")
        print(f"[AudioPlayer] Will autostart after {self.autostart_threshold} frames")

    def add_frame(self, audio_data):
        """Add decoded audio frame to buffer."""
        with self.buffer_lock:
            self.buffer.append(audio_data)
            self.frames_received += 1

            if self.frames_received % 10 == 0:
                print(f"[AudioPlayer] Buffer: {len(self.buffer)} frames ({len(self.buffer) * 5}s)")

            # Autostart playback when buffer is full enough
            if not self.is_playing and len(self.buffer) >= self.autostart_threshold:
                print(f"[AudioPlayer] Buffer ready ({len(self.buffer)} frames), starting playback...")
                self.start_playback()

    def audio_callback(self, outdata, frames, time_info, status):
        """Called by sounddevice to fill audio buffer."""
        if status:
            print(f"[AudioPlayer] Status: {status}")

        with self.buffer_lock:
            if len(self.buffer) > 0:
                # Get next frame from buffer
                frame = self.buffer.popleft()

                # Handle frame size mismatch
                if len(frame) < frames:
                    # Pad with silence if frame too short
                    padding = np.zeros((frames - len(frame), self.channels), dtype=np.float32)
                    frame = np.vstack([frame, padding])
                elif len(frame) > frames:
                    # Split frame if too long, put remainder back
                    remainder = frame[frames:]
                    frame = frame[:frames]
                    self.buffer.appendleft(remainder)

                outdata[:] = frame
                self.frames_played += 1

                if self.frames_played % 10 == 0:
                    print(f"[AudioPlayer] Played {self.frames_played} frames, buffer: {len(self.buffer)}")
            else:
                # Buffer underrun - output silence
                outdata.fill(0)
                if self.frames_played > 0:  # Don't warn during initial buffering
                    print(f"[AudioPlayer] Warning: Buffer underrun!")

    def start_playback(self):
        """Start audio playback stream."""
        if self.is_playing:
            return

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=4096
            )
            self.stream.start()
            self.is_playing = True
            print("[AudioPlayer] Playback started")
        except Exception as e:
            print(f"[AudioPlayer] Error starting playback: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop audio playback."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_playing = False
            print("[AudioPlayer] Playback stopped")


class TermuxListener:
    """
    Reticulum Radio Listener for Termux/Android.
    Receives and decodes EnCodec audio over Reticulum network.
    """

    def __init__(self, buffer_seconds=60):
        self.buffer_seconds = buffer_seconds
        self.reticulum = None
        self.identity = None
        self.decoder = None
        self.player = None
        self.broadcaster_identity = None
        self.broadcast_destination = None
        self.station_name = None  # Full station name from broadcaster

        # Statistics
        self.packets_received = 0
        self.frames_decoded = 0
        self.bytes_received = 0
        self.start_time = None

    def setup_reticulum(self):
        """Initialize Reticulum network stack."""
        print("\n" + "=" * 70)
        print("Initializing Reticulum...")
        print("=" * 70)

        # Start Reticulum
        self.reticulum = RNS.Reticulum()

        # Load or create identity
        identity_path = Path.home() / ".reticulum" / "identities" / "radio_listener"

        if identity_path.exists():
            self.identity = RNS.Identity.from_file(str(identity_path))
            print(f"Loaded identity from {identity_path}")
        else:
            print("Creating new identity...")
            self.identity = RNS.Identity()
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            self.identity.to_file(str(identity_path))
            print(f"Saved identity to {identity_path}")

        print(f"Identity hash: {RNS.prettyhexrep(self.identity.hash)}")
        print()

    def connect_to_broadcaster(self, control_dest_hash):
        """Connect to broadcaster control channel and get station info."""
        print("=" * 70)
        print(f"Connecting to broadcaster: {control_dest_hash}")
        print("=" * 70)

        # Request path to broadcaster
        print("Requesting path to broadcaster...")
        dest_hash_bytes = bytes.fromhex(control_dest_hash)

        if not RNS.Transport.has_path(dest_hash_bytes):
            RNS.Transport.request_path(dest_hash_bytes)
            print("Waiting for path...")

            # Wait for path
            timeout = 30
            start = time.time()
            while not RNS.Transport.has_path(dest_hash_bytes):
                time.sleep(0.1)
                if time.time() - start > timeout:
                    print("ERROR: Path request timeout!")
                    return False

        print("✓ Path found")

        # Get broadcaster identity
        self.broadcaster_identity = RNS.Identity.recall(dest_hash_bytes)
        if not self.broadcaster_identity:
            print("ERROR: Could not recall broadcaster identity")
            return False

        print(f"✓ Broadcaster identity: {RNS.prettyhexrep(self.broadcaster_identity.hash)}")

        # Create destination for control channel
        control_dest = RNS.Destination(
            self.broadcaster_identity,
            RNS.Destination.OUT,
            RNS.Destination.SINGLE,
            APP_NAME,
            "control"
        )

        # Create link to broadcaster
        print("Establishing link...")
        link = RNS.Link(control_dest)

        # Wait for link to establish
        timeout = 30
        start = time.time()
        while link.status != RNS.Link.ACTIVE:
            time.sleep(0.1)
            if time.time() - start > timeout:
                print("ERROR: Link establishment timeout!")
                return False

        print("✓ Link established")

        # Request station info
        print("Requesting station information...")
        request_data = umsgpack.packb({"type": "get_station_info"})
        link.request(
            "/station_info",
            request_data,
            response_callback=self.handle_station_info,
            failed_callback=lambda: print("ERROR: Station info request failed")
        )

        # Wait for response
        time.sleep(2)

        return True

    def handle_station_info(self, request_receipt):
        """Handle station info response."""
        try:
            response = umsgpack.unpackb(request_receipt.response)

            # Save station name for destination creation
            self.station_name = response.get('station_name', 'Unknown')

            print("\n" + "=" * 70)
            print("STATION INFORMATION")
            print("=" * 70)
            print(f"Station: {self.station_name}")
            print(f"Streams available: {len(response.get('streams', []))}")

            for stream in response.get('streams', []):
                print(f"  - {stream['name']}: {stream['codec']}")

            print("=" * 70)
            print()

            # We'll use 'low' quality (EnCodec)
            self.setup_decoder()
            self.setup_broadcast_listener()

        except Exception as e:
            print(f"Error handling station info: {e}")
            import traceback
            traceback.print_exc()

    def setup_decoder(self):
        """Setup EnCodec decoder."""
        print("Setting up EnCodec decoder...")

        self.decoder = EnCodecStreamable(
            bandwidth=1.5,
            force_cpu=True,  # Force CPU on Android
            compression=None  # Must match broadcaster
        )

        # Set expected identity for signature verification
        self.decoder.expected_identity = self.broadcaster_identity

        print("✓ Decoder ready")

    def setup_broadcast_listener(self):
        """Setup broadcast destination to receive audio packets."""
        print("Setting up broadcast listener...")

        # Create PLAIN destination for broadcast reception
        # Must match broadcaster's destination exactly: APP_NAME/broadcast/<station_name>/low
        self.broadcast_destination = RNS.Destination(
            None,
            RNS.Destination.IN,
            RNS.Destination.PLAIN,
            APP_NAME,
            "broadcast",
            self.station_name,  # Full station name from broadcaster
            "low"  # Quality level
        )

        # Set packet callback
        self.broadcast_destination.set_packet_callback(self.packet_callback)

        print(f"✓ Listening on: {APP_NAME}/broadcast/{self.station_name}/low")
        print(f"✓ Destination hash: {RNS.prettyhexrep(self.broadcast_destination.hash)}")

        # Create audio player
        self.player = AudioPlayer(
            sample_rate=48000,
            channels=1,
            buffer_seconds=self.buffer_seconds
        )

        print("✓ Broadcast listener ready")
        print()
        print("=" * 70)
        print("LISTENING FOR AUDIO...")
        print("=" * 70)
        print()

        self.start_time = time.time()

    def packet_callback(self, data, packet):
        """Handle incoming audio packet."""
        try:
            self.packets_received += 1
            self.bytes_received += len(data)

            # Decode audio chunk
            decoded_frame = self.decoder.decode(data)

            # Only send non-empty frames to player
            if decoded_frame.shape[0] > 0:
                self.frames_decoded += 1
                self.player.add_frame(decoded_frame)

                # Statistics
                if self.frames_decoded % 10 == 0:
                    elapsed = time.time() - self.start_time
                    bitrate = (self.bytes_received * 8) / (elapsed * 1000)  # kbps
                    print(f"[Stats] {self.frames_decoded} frames | {self.packets_received} packets | {bitrate:.2f} kbps")

        except Exception as e:
            print(f"[Error] Packet callback: {e}")
            import traceback
            traceback.print_exc()

    def run(self, control_dest_hash):
        """Main listener loop."""
        # Setup Reticulum
        self.setup_reticulum()

        # Connect to broadcaster
        if not self.connect_to_broadcaster(control_dest_hash):
            print("Failed to connect to broadcaster")
            return

        # Keep running
        try:
            print("Listener running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            if self.player:
                self.player.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Reticulum Radio Listener for Termux/Android"
    )
    parser.add_argument(
        "--station",
        required=True,
        help="Control destination hash of broadcaster"
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=60,
        help="Audio buffer size in seconds (default: 60)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Reticulum Radio Listener for Termux/Android")
    print("=" * 70)
    print()

    listener = TermuxListener(buffer_seconds=args.buffer)
    listener.run(args.station)


if __name__ == "__main__":
    main()
