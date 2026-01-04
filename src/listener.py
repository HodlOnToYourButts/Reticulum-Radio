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
from LXST.Codecs import Opus, Codec2
from LXST.Sources import Source
import threading
from threading import Event
from collections import deque
import RNS.vendor.umsgpack as mp

# Import neural codecs for ultra-low bitrate
from encodec_codec_streamable import EnCodecStreamable, ENCODEC
from hilcodec_codec import HILCodecLXST, HILCODEC

# Alias for compatibility
EnCodecLXST = EnCodecStreamable

# Patch LXST's codec_header_byte to support EnCodec and HILCodec
import LXST.Codecs
from LXST.Codecs import Raw, Opus, Codec2, RAW, OPUS, CODEC2

_original_codec_header_byte = LXST.Codecs.codec_header_byte

def patched_codec_header_byte(codec):
    """Extended codec_header_byte that supports EnCodec and HILCodec"""
    if codec == EnCodecStreamable or codec == EnCodecLXST:
        return ENCODEC.to_bytes()
    elif codec == HILCodecLXST:
        return HILCODEC.to_bytes()
    else:
        return _original_codec_header_byte(codec)

LXST.Codecs.codec_header_byte = patched_codec_header_byte
# Also patch in Network module where it's imported
import LXST.Network
LXST.Network.codec_header_byte = patched_codec_header_byte

# Now import codec_header_byte after patching so we get the patched version
from LXST.Network import codec_header_byte

APP_NAME = "reticulumradio"
BROADCAST_ASPECT = "broadcast"
CONTROL_ASPECT = "control"


class PacketSource(Source):
    """
    LXST Source that receives broadcast audio packets from RNS destination.
    Verifies Ed25519 signatures to prevent spoofing/DoS attacks.
    """

    def __init__(self):
        self.should_run = False
        self.codec = None
        self.sink = None
        self.frames_received = 0
        self.bytes_received = 0
        self.broadcaster_identity = None  # Set after control channel handshake (not used for EnCodec)
        self.signature_failures = 0
        self.packets_verified = 0
        self.start_time = None  # Track when first packet is received

        print("PacketSource: Initializing broadcast packet receiver with signature verification")

    def packet_callback(self, data, packet):
        """Called when a broadcast packet is received."""
        if not self.should_run:
            return

        try:
            if data and len(data) > 0:
                # For EnCodec: chunks are sent directly, no packet-level signature
                # Signature verification happens in the codec after reassembly
                audio_data = data

                # Track start time on first packet
                if self.start_time is None:
                    self.start_time = time.time()

                self.packets_verified += 1
                self.bytes_received += len(audio_data)

                # Calculate and display bitrate every 100 packets
                if self.packets_verified % 100 == 0:
                    elapsed_time = time.time() - self.start_time
                    kbps = (self.bytes_received * 8) / (elapsed_time * 1000)
                    print(f"Received {self.packets_verified} packets | {kbps:.2f} kbps")
                # For EnCodec: audio_data is already a chunk, pass directly to codec
                # No framing headers needed
                encoded_frame = audio_data

                # Debug: Check if codec and sink are set
                if self.frames_received == 0:
                    print(f"[PacketSource] codec={self.codec}, sink={self.sink}")

                # Decode the chunk using the codec (handles reassembly internally)
                if self.codec and self.sink:
                    try:
                        # The codec's decode method returns raw audio (numpy array)
                        decoded_frame = self.codec.decode(encoded_frame)

                        # Debug: Check decoded frame before sending to sink
                        if self.frames_received < 5:
                            print(f"[PacketSource] Decoded frame #{self.frames_received + 1}: {decoded_frame.shape}")
                            if decoded_frame.shape[0] > 0:
                                print(f"[PacketSource] Sending to sink: {type(self.sink).__name__}")
                            else:
                                print(f"[PacketSource] Skipping empty frame (waiting for chunk reassembly)")

                        # Only send non-empty frames to the sink
                        # Empty frames are returned while chunks are being reassembled
                        if decoded_frame.shape[0] > 0:
                            self.sink.handle_frame(decoded_frame, self)
                            self.frames_received += 1

                            if self.frames_received < 5:
                                print(f"[PacketSource] Frame sent to sink successfully")

                        # Report bitrate every 10 frames for real-time monitoring
                        if self.frames_received % 10 == 0 and self.start_time:
                            elapsed_time = time.time() - self.start_time
                            kbps = (self.bytes_received * 8) / (elapsed_time * 1000)
                            avg_frame_size = self.bytes_received / self.frames_received if self.frames_received > 0 else 0
                            print(f"[Bitrate] {kbps:.2f} kbps | Avg frame: {avg_frame_size:.0f} bytes | {self.frames_received} frames received")

                        if self.frames_received % 100 == 0:
                            if self.start_time:
                                elapsed_time = time.time() - self.start_time
                                kbps = (self.bytes_received * 8) / (elapsed_time * 1000)
                                print(f"PacketSource: {self.frames_received} frames received, {self.bytes_received} bytes total | {kbps:.2f} kbps")
                            else:
                                print(f"PacketSource: {self.frames_received} frames received, {self.bytes_received} bytes total")
                    except Exception as decode_error:
                        # Don't skip errors silently - print them to debug
                        print(f"[PacketSource] ERROR decoding/playing frame: {decode_error}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            print(f"Error processing packet: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        if not self.should_run:
            self.should_run = True
            print("PacketSource: Started and listening for broadcast packets")

    def stop(self):
        self.should_run = False
        print("PacketSource: Stopped")


class BufferedLineSink(LineSink):
    """Extended LineSink with a much larger buffer for network streaming."""

    def __init__(self, preferred_device=None, autodigest=True, low_latency=False, buffer_seconds=20):
        super().__init__(preferred_device=preferred_device, autodigest=autodigest, low_latency=low_latency)

        # We'll calculate the actual buffer size once we know the frame duration
        # For now, use a conservative estimate (60ms frames = 16.67 fps)
        self.buffer_seconds = buffer_seconds
        self.estimated_fps = 17  # Conservative estimate
        max_frames = int(self.estimated_fps * buffer_seconds)

        # Override the default small buffer with a large one
        self.frame_deque = deque(maxlen=max_frames)
        self.MAX_FRAMES = max_frames
        self.buffer_max_height = max_frames - 10  # Leave some headroom

        # Adjust autostart to wait for a good buffer before playing
        # Start playing when we have at least 2 seconds of audio buffered
        self.AUTOSTART_MIN = int(self.estimated_fps * 2)
        self.autostart_min = self.AUTOSTART_MIN

        # Never timeout - keep waiting for frames indefinitely
        self.frame_timeout = 999999

        self._actual_fps_calculated = False

        print(f"Created buffered sink: {buffer_seconds}s buffer (~{max_frames} frames), "
              f"will start after {self.AUTOSTART_MIN} frames (~2s)")

    def handle_frame(self, frame, source=None):
        """Override to show buffer status periodically and adjust buffer size."""
        # Debug first few frames
        if not hasattr(self, '_frames_handled'):
            self._frames_handled = 0

        if self._frames_handled < 5:
            print(f"[BufferedLineSink] Receiving frame #{self._frames_handled + 1}: {frame.shape}")
            print(f"[BufferedLineSink] Current buffer size: {len(self.frame_deque)}")

        self._frames_handled += 1

        super().handle_frame(frame, source)

        # Once we know the actual frame time, recalculate buffer size
        if not self._actual_fps_calculated and self.frame_time:
            actual_fps = 1.0 / self.frame_time
            max_frames = int(actual_fps * self.buffer_seconds)

            # For very low frame rates (like 0.5 fps for 2-second EnCodec frames),
            # ensure we have at least 8-10 frames buffered before starting
            # This provides a 16-20 second cushion against network jitter
            min_autostart_frames = max(8, int(actual_fps * 2))

            # Update buffer size
            old_deque = list(self.frame_deque)
            self.frame_deque = deque(old_deque, maxlen=max_frames)
            self.MAX_FRAMES = max_frames
            self.buffer_max_height = max_frames - 10
            self.AUTOSTART_MIN = min_autostart_frames
            self.autostart_min = self.AUTOSTART_MIN

            self._actual_fps_calculated = True
            print(f"Detected frame rate: {actual_fps:.1f} fps ({self.frame_time*1000:.1f}ms frames)")
            print(f"Adjusted buffer: {max_frames} frames = {self.buffer_seconds}s, autostart at {self.AUTOSTART_MIN} frames")
            if actual_fps < 1.0:
                buffer_time = self.AUTOSTART_MIN * self.frame_time
                print(f"  NOTE: Low frame rate detected - buffering {self.AUTOSTART_MIN} frames (~{buffer_time:.0f}s) before playback")
                print(f"        This prevents dropouts from network jitter")

        # Show buffer status occasionally
        if hasattr(self, '_frame_count'):
            self._frame_count += 1
            if self._frame_count % 100 == 0:  # Every 100 frames
                buffer_pct = (len(self.frame_deque) / self.MAX_FRAMES) * 100
                buffer_seconds = len(self.frame_deque) * (self.frame_time if self.frame_time else 0.06)
                print(f"Buffer: {len(self.frame_deque)}/{self.MAX_FRAMES} frames ({buffer_pct:.1f}%) = {buffer_seconds:.1f}s")
        else:
            self._frame_count = 0


class RadioListener:
    """
    Listener for receiving radio broadcasts over Reticulum network.
    """

    def __init__(self):
        """Initialize the listener."""
        self.reticulum = None
        self.identity = None
        self.current_station = None
        self.broadcaster_destination = None

        # LXST components
        self.packet_source = None
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
        print("\nDiscovering radio stations on the network...")
        print("Looking for announced control channels...")
        print("\n(This feature coming soon - for now, you must know the station name)")
        print("\nTo tune in to a station:")
        print("  python listener.py --station \"Station Name\"")

    def connect_to_control_channel(self, control_dest_hash):
        """
        Connect to broadcaster's control channel to get station info and public key.

        Args:
            control_dest_hash: Control destination hash (16 bytes) from broadcaster

        Returns:
            Tuple of (full_station_name, public_key, broadcaster_identity) or None on failure
        """
        print(f"Connecting to control channel...")
        print(f"Control destination: {RNS.prettyhexrep(control_dest_hash)}")

        try:
            # Request path to control destination
            if not RNS.Transport.has_path(control_dest_hash):
                print("Requesting path to broadcaster...")
                RNS.Transport.request_path(control_dest_hash)
                print("Waiting for path...")

                # Wait up to 10 seconds for path
                wait_time = 0
                while not RNS.Transport.has_path(control_dest_hash) and wait_time < 10:
                    time.sleep(0.5)
                    wait_time += 0.5

                if not RNS.Transport.has_path(control_dest_hash):
                    print("ERROR: Could not find path to broadcaster")
                    print("  Make sure broadcaster is running and announced")
                    return None

            print("Path found! Recalling broadcaster identity...")

            # Recall the broadcaster's identity from the announce
            # The announce contains the identity, and RNS caches it
            broadcaster_identity = RNS.Identity.recall(control_dest_hash)
            if not broadcaster_identity:
                print("ERROR: Could not recall broadcaster identity from announce")
                print("  The broadcaster may not have announced yet, or announce not received")
                return None

            print(f"Identity recalled: {RNS.prettyhexrep(broadcaster_identity.hash)}")

            # Create destination object for the control channel
            control_destination = RNS.Destination(
                broadcaster_identity,
                RNS.Destination.OUT,  # Outbound destination (we're connecting TO it)
                RNS.Destination.SINGLE,
                APP_NAME,
                CONTROL_ASPECT
            )

            print("Establishing link...")

            # Create link to control channel
            link = RNS.Link(control_destination)

            # Wait for link to establish
            wait_time = 0
            while link.status != RNS.Link.ACTIVE and wait_time < 10:
                time.sleep(0.5)
                wait_time += 0.5

            if link.status != RNS.Link.ACTIVE:
                print("ERROR: Failed to establish link to broadcaster")
                return None

            print("Link established! Requesting station info...")

            # Set up response callback BEFORE sending request
            response_received = Event()
            response_data = {}

            def response_callback(message, packet):
                response = mp.unpackb(message)
                response_data['station_name'] = response.get('station_name')
                response_data['public_key'] = response.get('public_key')
                response_data['streams'] = response.get('streams', {})  # Dict of available streams
                response_received.set()

            link.set_packet_callback(response_callback)

            # Now send the request
            request = {'type': 'get_info'}
            request_data = mp.packb(request)

            packet = RNS.Packet(link, request_data)
            packet.send()

            # Wait for response
            if not response_received.wait(timeout=10):
                print("ERROR: Timeout waiting for station info")
                link.teardown()
                return None

            print(f"Received station info: {response_data['station_name']}")

            # Create broadcaster identity from public key
            broadcaster_identity = RNS.Identity(create_keys=False)
            broadcaster_identity.load_public_key(response_data['public_key'])

            # Close the control link (we only needed it for key exchange)
            link.teardown()

            return (response_data['station_name'], response_data['public_key'], broadcaster_identity, response_data['streams'])

        except Exception as e:
            print(f"Error connecting to control channel: {e}")
            import traceback
            traceback.print_exc()
            return None

    def tune_to_station(self, station_name, quality='high'):
        """
        Tune into a specific radio station.

        Args:
            station_name: Control destination hash (32 hex chars) from broadcaster
            quality: Quality level to listen to ('high' or 'low')
        """
        print(f"\nTuning to station: {station_name} (Quality: {quality})")

        try:
            # Parse control destination hash
            control_dest_hash = None

            # Expect 32 hex chars (16 bytes)
            if len(station_name) == 32:
                try:
                    control_dest_hash = bytes.fromhex(station_name)
                    print(f"Using control destination: {RNS.prettyhexrep(control_dest_hash)}")
                except ValueError as e:
                    print(f"ERROR: Invalid hex string: {e}")
                    return
            else:
                print(f"ERROR: Expected 32 hex characters, got {len(station_name)}")
                print("\nPlease copy the 'Control Destination' hash from broadcaster output")
                print("Example: python listener.py --station a1b2c3d4e5f6g7h8...")
                return

            # Step 1: Connect to control channel to get public key and station info
            control_result = self.connect_to_control_channel(control_dest_hash)

            if not control_result:
                print("ERROR: Failed to connect to control channel")
                return

            verified_station_name, public_key, broadcaster_identity, streams = control_result

            print(f"✓ Verified station: {verified_station_name}")
            print(f"✓ Obtained public key for signature verification")

            # Show available streams
            print(f"\n✓ Available streams:")
            for stream_quality, stream_info in streams.items():
                print(f"  - {stream_quality}: {stream_info['quality']}")

            # Check if requested quality is available
            if quality not in streams:
                print(f"\nERROR: Quality '{quality}' not available")
                print(f"Available qualities: {', '.join(streams.keys())}")
                return

            selected_stream = streams[quality]
            print(f"\n✓ Selected stream: {quality} ({selected_stream['quality']})")

            # Step 2: Create PLAIN destination to receive broadcasts on this channel
            # Use the broadcast hash from the selected stream
            broadcast_hash = selected_stream['hash']

            # We need to create a destination that matches the broadcaster's PLAIN destination
            # Since we have the hash, we can use it directly to set packet callback
            # But we need the actual destination object for the callback
            # Let's recreate the destination using the same parameters
            self.broadcaster_destination = RNS.Destination(
                None,  # No identity for PLAIN destinations
                RNS.Destination.IN,
                RNS.Destination.PLAIN,
                APP_NAME,
                BROADCAST_ASPECT,
                verified_station_name,  # Full channel name from broadcaster
                quality  # Quality level
            )

            print(f"✓ Listening on channel: {verified_station_name} ({quality})")
            print(f"✓ Broadcast destination hash: {RNS.prettyhexrep(self.broadcaster_destination.hash)}")
            print(f"✓ Ready to receive and verify signed broadcasts")

            # Set up audio pipeline
            print("Setting up audio playback...")

            # Create speaker sink with very large buffer for network streaming
            # 60 second buffer to handle network jitter
            self.speaker_sink = BufferedLineSink(buffer_seconds=60)

            # Create appropriate decoder based on stream codec
            codec_type = selected_stream['codec']
            if codec_type == 'opus':
                # High quality Opus
                decoder = Opus(profile=Opus.PROFILE_AUDIO_HIGH)
            elif codec_type == 'encodec_stereo':
                # EnCodec 48kHz stereo at 6 kbps with entropy coding
                # Use GPU but disable torch.compile() to avoid hangs
                # GPU is needed for fast entropy decoding (CPU too slow)
                decoder = EnCodecLXST(bandwidth=6.0, use_compile=False, force_cpu=False, use_48khz=True, stereo=True)
            elif codec_type == 'hilcodec':
                # Legacy: HILCodec neural codec at 3 kbps (~2.9 kbps actual)
                # Uses Python 3.10 subprocess with ONNX models
                decoder = HILCodecLXST(num_quantizers=4)
            elif codec_type == 'encodec':
                # EnCodec 24kHz mono at 1.5 kbps (no post-compression)
                # Streamable mode: 5s frames with signature verification
                # ~1.6 kbps actual (fits well within 3.1 kbps LoRa limit)
                # Android compatible (no PyTorch language model needed)
                decoder = EnCodecStreamable(
                    bandwidth=1.5,
                    force_cpu=False,
                    compression=None  # Must match broadcaster setting
                )
                # Set expected identity for signature verification (after creation)
                decoder.expected_identity = broadcaster_identity
            elif codec_type == 'opus_low':
                # Low quality Opus for LoRa (legacy)
                decoder = Opus(profile=Opus.PROFILE_AUDIO_LOW)
            elif codec_type == 'opus_min':
                # Minimum bitrate Opus for LoRa (legacy)
                decoder = Opus(profile=Opus.PROFILE_AUDIO_MIN)
            elif codec_type == 'codec2':
                # Legacy Codec2 support
                decoder = Codec2(mode=Codec2.CODEC2_3200)
            else:
                print(f"ERROR: Unknown codec type: {codec_type}")
                return

            # Create packet source to receive and verify broadcast packets
            # Broadcaster identity obtained from control channel
            self.packet_source = PacketSource()
            self.packet_source.broadcaster_identity = broadcaster_identity

            # Create pipeline: PacketSource -> Decoder (Opus/EnCodec/Codec2) -> Speaker
            # Pipeline requires source, codec, and sink in constructor
            self.pipeline = Pipeline(self.packet_source, decoder, self.speaker_sink)

            # CRITICAL: Manually set codec and sink on PacketSource
            # PacketSource.packet_callback() needs these to decode and play audio
            self.packet_source.codec = decoder
            self.packet_source.sink = self.speaker_sink

            print(f"✓ PacketSource codec set to: {type(decoder).__name__}")
            print(f"✓ PacketSource sink set to: {type(self.speaker_sink).__name__}")

            # Register packet callback to receive broadcast packets
            self.broadcaster_destination.set_packet_callback(self.packet_source.packet_callback)

            # Start the packet source
            self.packet_source.start()

            # Start the pipeline
            self.pipeline.start()

            self.current_station = verified_station_name
            print(f"\n✓ Tuned to station '{verified_station_name}'")
            print("✓ Listening for broadcast packets...")
            print("✓ Playing audio...")

            # Start receiving and playing audio
            self.play_stream()

        except Exception as e:
            print(f"Error tuning to station: {e}")
            import traceback
            traceback.print_exc()

    def play_stream(self):
        """Receive and play the audio stream."""
        print("\nReceiving broadcast stream...")
        print("Press Ctrl+C to stop listening")

        try:
            # The pipeline is already running, just keep it alive
            # Keep receiving broadcast packets
            while True:
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nStopping playback...")

    def start(self, station_name=None, quality='high'):
        """
        Start the listener.

        Args:
            station_name: Optional station name/channel to tune to immediately
            quality: Quality level ('high' or 'low')
        """
        self.setup_reticulum()

        print(f"\n{'='*60}")
        print(f"  Reticulum Radio Listener")
        print(f"{'='*60}\n")

        if station_name:
            self.tune_to_station(station_name, quality)
        else:
            print("ERROR: No station specified!")
            print("\nUsage: python listener.py --station <control_dest_hash>")
            print("\nThe control destination hash is shown when broadcaster starts")
            print("Look for 'Control Destination: <hash>' in broadcaster output")
            print("\nExample: python listener.py --station a1b2c3d4e5f6a7b8...")
            sys.exit(1)

    def shutdown(self):
        """Clean shutdown of listener."""
        if self.pipeline:
            print("Stopping audio pipeline...")
            self.pipeline.stop()

        if self.packet_source:
            print("Stopping packet source...")
            self.packet_source.stop()

        print("Listener stopped.")


def main():
    """Main entry point for listener CLI."""
    parser = argparse.ArgumentParser(description="Reticulum Radio Listener")
    parser.add_argument(
        "--station",
        type=str,
        help="Control destination hash (32 hex chars, shown by broadcaster at startup)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["high", "low"],
        help="Stream quality: 'high' (Opus stereo music) or 'low' (Codec2 ultra-low bandwidth <3kbps)"
    )
    args = parser.parse_args()

    listener = RadioListener()

    try:
        listener.start(station_name=args.station, quality=args.quality)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        listener.shutdown()


if __name__ == "__main__":
    main()
