#!/usr/bin/env python3
"""
Reticulum Radio Broadcaster

DJ-style broadcaster with always-on microphone and MP3 queue management.
When music is playing, it broadcasts; when paused or queue empty, microphone goes through.
"""

import RNS
import LXST
import argparse
import sys
import os
from pathlib import Path
from queue import Queue
from threading import Thread, Event, Lock
import time
import numpy as np
import math

from LXST import Pipeline, Mixer
from LXST.Sources import LineSource, OpusFileSource, Source
from LXST.Sinks import LineSink, RemoteSink
from LXST.Codecs import Opus, Codec2, Codec
from LXST.Network import Packetizer, FIELD_FRAMES
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

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. MP3 playback will not work.")

APP_NAME = "reticulumradio"
BROADCAST_ASPECT = "broadcast"
CONTROL_ASPECT = "control"


class RawPassthroughCodec(Codec):
    """
    Passthrough codec that doesn't encode - just returns raw audio frames.
    Used to bypass LXST Pipeline's encoding when we want to handle encoding ourselves.
    """
    def __init__(self):
        super().__init__()
        self._debug_count = 0

    def encode(self, frame):
        # Debug first few frames
        if self._debug_count < 3:
            print(f"DEBUG: RawPassthroughCodec.encode() frame #{self._debug_count + 1}")
            print(f"  Shape: {frame.shape if hasattr(frame, 'shape') else 'N/A'}")
            print(f"  Dtype: {frame.dtype if hasattr(frame, 'dtype') else 'N/A'}")
            if hasattr(frame, 'min'):
                print(f"  Range: [{frame.min():.3f}, {frame.max():.3f}]")
            self._debug_count += 1
        return frame  # Return raw audio as-is

    def decode(self, frame):
        return frame  # Return raw audio as-is


class ForkedBroadcastSink(RemoteSink):
    """
    Sink that broadcasts audio in multiple qualities simultaneously.
    Encodes raw audio with multiple codecs and broadcasts to different destinations.
    """

    def __init__(self, destinations, identity, codecs, batch_sizes=None):
        """
        Args:
            destinations: Dict of {quality_name: RNS.Destination}
            identity: RNS.Identity for signing packets
            codecs: Dict of {quality_name: codec_instance}
            batch_sizes: Dict of {quality_name: batch_size} or single int for all
        """
        self.destinations = destinations
        self.identity = identity
        self.codecs = codecs
        self.should_run = False
        self.source = None

        # Set batch sizes per quality
        if batch_sizes is None:
            batch_sizes = 2
        if isinstance(batch_sizes, int):
            # Single value for all qualities
            self.batch_sizes = {quality: batch_sizes for quality in destinations.keys()}
        else:
            # Per-quality batch sizes
            self.batch_sizes = batch_sizes

        # Per-quality stats
        self.quality_stats = {}
        for quality in destinations.keys():
            self.quality_stats[quality] = {
                'frames_sent': 0,
                'packets_sent': 0,
                'bytes_sent': 0,
                'send_failures': 0,
                'frame_batch': [],
                'batch_lock': Lock(),
                'batch_size': self.batch_sizes.get(quality, 2)
            }

        self.last_send_time = 0
        self.min_send_interval = 0.01

        print(f"ForkedBroadcastSink: Broadcasting to {len(destinations)} quality levels")
        for quality, dest in destinations.items():
            batch_size = self.batch_sizes.get(quality, 2)
            print(f"  {quality}: {RNS.prettyhexrep(dest.hash)} ({batch_size} frames/packet)")

    def handle_frame(self, raw_frame, source=None):
        """
        Receives RAW audio frame, encodes with all codecs, broadcasts to all destinations.
        """
        if not self.should_run:
            return

        # Debug: Check raw frame format on first frame
        if not hasattr(self, '_raw_frame_logged'):
            self._raw_frame_logged = True
            print(f"DEBUG: First raw frame received by ForkedBroadcastSink:")
            print(f"  Type: {type(raw_frame)}")
            print(f"  Shape: {raw_frame.shape if hasattr(raw_frame, 'shape') else 'N/A'}")
            print(f"  Dtype: {raw_frame.dtype if hasattr(raw_frame, 'dtype') else 'N/A'}")
            if hasattr(raw_frame, 'min'):
                print(f"  Range: [{raw_frame.min():.3f}, {raw_frame.max():.3f}]")
            print(f"  Source param: {type(source).__name__ if source else 'None'}")

        # Initialize codecs with source info on first frame
        if source and not hasattr(self, '_codecs_initialized'):
            print("DEBUG: Initializing codecs with source info:")
            for quality, codec in self.codecs.items():
                # Use the actual audio source (LineSource), not the pipeline source (RawPassthroughCodec)
                codec.source = self.source
                print(f"  {quality}: {type(codec).__name__} initialized with source {type(self.source).__name__}")
                if hasattr(self.source, 'samplerate'):
                    print(f"    Source sample rate: {self.source.samplerate} Hz")
            self._codecs_initialized = True

        # Encode with each codec and broadcast
        for quality, codec in self.codecs.items():
            stats = self.quality_stats[quality]
            destination = self.destinations[quality]

            with stats['batch_lock']:
                try:
                    # Debug: Show encoding attempts for first few frames
                    if quality == 'low' and stats['frames_sent'] < 3:
                        print(f"DEBUG: Encoding {quality} frame #{stats['frames_sent'] + 1}, shape={raw_frame.shape}")

                    # High quality uses Opus (stereo), low quality uses EnCodec (mono)
                    frame_to_encode = raw_frame

                    # Encode raw audio with this quality's codec
                    encoded_frame = codec.encode(frame_to_encode)

                    # Skip empty frames (EnCodec returns empty when buffering)
                    if len(encoded_frame) == 0:
                        continue

                    # For EnCodec, chunks are already signed internally
                    # Just send them directly without extra framing
                    if isinstance(codec, EnCodecStreamable):
                        # Chunks already have headers and signatures
                        # Send immediately without batching
                        packet_data = encoded_frame

                        # Check MTU
                        if len(packet_data) > RNS.Reticulum.MTU:
                            stats['send_failures'] += 1
                            print(f"WARNING ({quality}): Chunk size {len(packet_data)} exceeds MTU {RNS.Reticulum.MTU}")
                            continue

                        # Send packet
                        packet = RNS.Packet(destination, packet_data, create_receipt=False)
                        packet.send()

                        stats['frames_sent'] += 1
                        stats['packets_sent'] += 1
                        stats['bytes_sent'] += len(packet_data)

                        # Report stats periodically
                        if stats['packets_sent'] % 50 == 0:
                            avg_size = stats['bytes_sent'] // stats['packets_sent']
                            print(f"Broadcast ({quality}): {stats['packets_sent']} packets ({stats['frames_sent']} chunks), avg: {avg_size} bytes/packet")
                    else:
                        # Other codecs: use old batching + signing method
                        # Add codec header
                        frame_with_header = codec_header_byte(type(codec)) + encoded_frame

                        # Add length prefix for proper framing
                        frame_length = len(frame_with_header)
                        length_prefix = frame_length.to_bytes(2, byteorder='big')
                        framed_data = length_prefix + frame_with_header

                        # Add to batch
                        stats['frame_batch'].append(framed_data)
                        stats['frames_sent'] += 1

                        # Send when batch is full
                        if len(stats['frame_batch']) >= stats['batch_size']:
                            self._send_batch(quality, stats, destination)

                except Exception as e:
                    stats['send_failures'] += 1
                    if stats['send_failures'] <= 3 or stats['send_failures'] % 10 == 1:
                        print(f"Error encoding/broadcasting {quality} frame: {e}")
                        print(f"  Frame shape: {raw_frame.shape if hasattr(raw_frame, 'shape') else 'unknown'}")
                        print(f"  Codec: {type(codec).__name__}")
                        import traceback
                        traceback.print_exc()

    def _send_batch(self, quality, stats, destination):
        """Send a batch of frames for a specific quality."""
        # Debug for low quality
        if quality == 'low' and stats['packets_sent'] < 3:
            print(f"DEBUG: _send_batch called for {quality}, batch has {len(stats['frame_batch'])} frames")

        # Concatenate all frames in batch
        batch_data = b''.join(stats['frame_batch'])

        # Sign the batch data
        signature = self.identity.sign(batch_data)

        # Package: signature (64 bytes) + data
        signed_packet = {
            's': signature,
            'd': batch_data
        }
        packet_data = mp.packb(signed_packet)

        # Debug packet size for low quality
        if quality == 'low' and stats['packets_sent'] < 3:
            print(f"DEBUG: {quality} packet size: {len(packet_data)} bytes (MTU: {RNS.Reticulum.MTU})")

        # Check MTU
        if len(packet_data) > RNS.Reticulum.MTU:
            batch_frame_count = len(stats['frame_batch'])
            stats['send_failures'] += batch_frame_count
            # Always warn on MTU violations - they indicate a configuration problem
            print(f"WARNING ({quality}): Packet size {len(packet_data)} exceeds MTU {RNS.Reticulum.MTU}")
            print(f"  Dropped {batch_frame_count} frames. Reduce batch_size (currently {stats['batch_size']})")
            stats['frame_batch'] = []
            return

        # Rate limiting
        current_time = time.time()
        time_since_last_send = current_time - self.last_send_time
        if time_since_last_send < self.min_send_interval:
            time.sleep(self.min_send_interval - time_since_last_send)

        # Broadcast packet
        packet = RNS.Packet(destination, packet_data, create_receipt=False)
        packet.send()
        self.last_send_time = time.time()

        # Debug packet sent confirmation
        if quality == 'low' and stats['packets_sent'] < 3:
            print(f"DEBUG: {quality} packet sent successfully")

        # Update stats
        stats['packets_sent'] += 1
        stats['bytes_sent'] += len(packet_data)

        # Report stats periodically
        if stats['packets_sent'] == 1 or stats['packets_sent'] % 50 == 0:
            avg_size = stats['bytes_sent'] // stats['packets_sent']
            print(f"Broadcast ({quality}): {stats['packets_sent']} packets ({stats['frames_sent']} frames), avg: {avg_size} bytes/packet")

        # Clear batch
        stats['frame_batch'] = []

    def start(self):
        if not self.should_run:
            self.should_run = True

    def stop(self):
        self.should_run = False
        # Flush remaining frames for all qualities
        for quality, stats in self.quality_stats.items():
            with stats['batch_lock']:
                if len(stats['frame_batch']) > 0:
                    self._send_batch(quality, stats, self.destinations[quality])


class BroadcastSink(RemoteSink):
    """
    Sink that broadcasts audio frames to all listeners via RNS destination.
    Uses true Reticulum broadcast - one packet reaches all listeners.
    Batches multiple frames per packet to reduce packet rate.
    Signs packets with Ed25519 for authentication.
    """

    def __init__(self, destination, identity, batch_size=2):
        self.destination = destination
        self.identity = identity  # For signing packets
        self.should_run = False
        self.source = None
        self.frames_sent = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.send_failures = 0
        self.batch_size = batch_size
        self.frame_batch = []
        self.batch_lock = Lock()
        self.last_send_time = 0
        self.min_send_interval = 0.01  # Minimum 10ms between packets

        print(f"BroadcastSink: Broadcasting to destination {RNS.prettyhexrep(destination.hash)}")
        print(f"BroadcastSink: Batching {batch_size} frames per packet, signed with Ed25519")

    def handle_frame(self, frame, source=None):
        if not self.should_run:
            return

        with self.batch_lock:
            try:
                # Add codec header to frame
                frame_with_header = codec_header_byte(type(self.source.codec)) + frame

                # Add length prefix for proper framing
                # Format: [length:2 bytes][codec_byte:1 byte][encoded_data]
                frame_length = len(frame_with_header)
                length_prefix = frame_length.to_bytes(2, byteorder='big')
                framed_data = length_prefix + frame_with_header

                # Add to batch
                self.frame_batch.append(framed_data)
                self.frames_sent += 1

                # Send when batch is full
                if len(self.frame_batch) >= self.batch_size:
                    # Concatenate all frames in batch
                    batch_data = b''.join(self.frame_batch)

                    # Sign the batch data with Ed25519
                    signature = self.identity.sign(batch_data)

                    # Package: signature (64 bytes) + data
                    # Public key is obtained via control channel, not in every packet
                    signed_packet = {
                        's': signature,     # Ed25519 signature (64 bytes)
                        'd': batch_data     # Audio data
                    }
                    packet_data = mp.packb(signed_packet)

                    # Check if packet will fit within MTU
                    if len(packet_data) > RNS.Reticulum.MTU:
                        self.send_failures += len(self.frame_batch)
                        if self.send_failures % 10 == 1:
                            print(f"Warning: Signed packet size {len(packet_data)} exceeds MTU {RNS.Reticulum.MTU}")
                            print(f"Reduce batch_size (currently {self.batch_size}) or frame duration")
                        self.frame_batch = []
                        return

                    # Rate limiting: ensure minimum interval between sends
                    current_time = time.time()
                    time_since_last_send = current_time - self.last_send_time
                    if time_since_last_send < self.min_send_interval:
                        time.sleep(self.min_send_interval - time_since_last_send)

                    # Broadcast signed packet to all listeners
                    packet = RNS.Packet(self.destination, packet_data, create_receipt=False)

                    # For PLAIN destinations, send() returns None (no receipt)
                    # Just call send() and assume success
                    packet.send()
                    self.last_send_time = time.time()

                    # Count packet as sent
                    self.packets_sent += 1
                    self.bytes_sent += len(packet_data)

                    # Report stats periodically
                    if self.packets_sent % 50 == 0:
                        avg_size = self.bytes_sent // self.packets_sent if self.packets_sent > 0 else 0
                        overhead = 64  # Signature only (public key from control channel)
                        print(f"Broadcast: {self.packets_sent} packets ({self.frames_sent} frames), avg: {avg_size} bytes/packet (+{overhead}b sig)")

                    # Clear batch
                    self.frame_batch = []

            except Exception as e:
                self.send_failures += 1
                if self.send_failures % 10 == 1:
                    print(f"Error broadcasting frame: {e}")
                self.frame_batch = []

    def start(self):
        if not self.should_run:
            RNS.log(f"{self} starting", RNS.LOG_DEBUG)
            self.should_run = True

    def stop(self):
        self.should_run = False
        # Flush any remaining frames
        with self.batch_lock:
            if len(self.frame_batch) > 0:
                batch_data = b''.join(self.frame_batch)
                if len(batch_data) <= RNS.Reticulum.MTU:
                    packet = RNS.Packet(self.destination, batch_data, create_receipt=False)
                    packet.send()
                self.frame_batch = []


class MP3Source(LXST.Sources.Source):
    """Custom audio source for playing MP3 files using pydub."""

    def __init__(self, file_path, target_frame_ms=20):
        self.file_path = file_path
        self.should_run = False
        self.playback_thread = None
        self.target_frame_ms = target_frame_ms
        self._codec = None
        self.codec = None  # Will be set by Pipeline
        self.sink = None  # Will be set by Pipeline
        self.pipeline = None  # Will be set by Pipeline
        self.finished = False

        # Load the MP3 file
        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub is required for MP3 playback")

        audio = AudioSegment.from_mp3(file_path)

        # Convert to the format we need
        audio = audio.set_frame_rate(48000)  # LXST standard sample rate
        audio = audio.set_channels(2)  # Stereo for high quality music
        audio = audio.set_sample_width(2)  # 16-bit

        self.samplerate = audio.frame_rate
        self.channels = audio.channels
        self.bitdepth = 16

        # Convert to numpy array normalized to [-1.0, 1.0]
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize int16 to float32
        # Always reshape to (samples, channels) format - even for mono
        self.samples = samples.reshape(-1, self.channels)
        self.sample_count = len(self.samples)
        self.samples_per_frame = int((self.target_frame_ms / 1000) * self.samplerate)
        self.current_sample = 0

        print(f"Loaded MP3: {os.path.basename(file_path)}, duration: {len(audio)/1000:.1f}s")

    def start(self):
        if not self.should_run:
            self.should_run = True
            self.current_sample = 0
            self.finished = False
            self.playback_thread = Thread(target=self._playback_job, daemon=True)
            self.playback_thread.start()

    def stop(self):
        self.should_run = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)

    def _playback_job(self):
        """Feed audio frames to the sink."""
        frame_time = self.samples_per_frame / self.samplerate
        next_frame_time = time.time()
        frames_sent = 0

        while self.should_run and self.current_sample < self.sample_count:
            # Wait until it's time for the next frame
            current_time = time.time()
            if current_time < next_frame_time:
                time.sleep(next_frame_time - current_time)

            # Check if the pipeline is ready (codec and sink should be set by Pipeline)
            if self.codec and self.sink and (not hasattr(self.sink, 'can_receive') or self.sink.can_receive(from_source=self)):
                # Get next frame
                end_sample = min(self.current_sample + self.samples_per_frame, self.sample_count)
                frame = self.samples[self.current_sample:end_sample]

                # Pad if necessary (for the last frame)
                if len(frame) < self.samples_per_frame:
                    padding = np.zeros((self.samples_per_frame - len(frame), self.channels), dtype=np.float32)
                    frame = np.vstack([frame, padding])

                # Ensure frame is float32
                frame = frame.astype(np.float32)

                # Let the codec encode it (works like LineSource)
                encoded_frame = self.codec.encode(frame)
                self.sink.handle_frame(encoded_frame, self)

                self.current_sample = end_sample
                frames_sent += 1

                # Show progress occasionally
                if frames_sent % 100 == 0:
                    elapsed = time.time() - (next_frame_time - (frames_sent * frame_time))
                    expected_time = frames_sent * frame_time
                    print(f"MP3: Sent {frames_sent} frames, elapsed: {elapsed:.1f}s, expected: {expected_time:.1f}s")

                # Schedule next frame - use absolute timing to prevent drift
                next_frame_time += frame_time
            else:
                # Pipeline not ready yet or sink can't receive
                time.sleep(0.001)

        print(f"MP3 playback finished: {frames_sent} frames sent")
        self.finished = True
        self.should_run = False


class MP3Queue:
    """Manages the MP3 playback queue."""

    def __init__(self):
        self.queue = []
        self.current_index = -1
        self.lock = Lock()

    def add(self, mp3_path):
        """Add an MP3 to the queue."""
        with self.lock:
            # Expand ~ and resolve path
            expanded_path = os.path.expanduser(mp3_path)
            expanded_path = os.path.abspath(expanded_path)

            if os.path.exists(expanded_path):
                self.queue.append(expanded_path)
                print(f"Added to queue: {os.path.basename(expanded_path)}")
                return True
            else:
                print(f"Error: File not found: {expanded_path}")
                return False

    def get_current(self):
        """Get the currently playing MP3 path."""
        with self.lock:
            if 0 <= self.current_index < len(self.queue):
                return self.queue[self.current_index]
            return None

    def next(self):
        """Move to next track in queue."""
        with self.lock:
            if self.current_index + 1 < len(self.queue):
                self.current_index += 1
                return self.queue[self.current_index]
            return None

    def has_next(self):
        """Check if there are more tracks."""
        with self.lock:
            return self.current_index + 1 < len(self.queue)

    def get_queue_status(self):
        """Get formatted queue status."""
        with self.lock:
            if not self.queue:
                return "Queue: Empty"

            status = f"Queue ({len(self.queue)} tracks):\n"
            for i, track in enumerate(self.queue):
                marker = "► " if i == self.current_index else "  "
                status += f"{marker}{i+1}. {os.path.basename(track)}\n"
            return status


class RadioBroadcaster:
    """
    DJ-style radio broadcaster with microphone and MP3 queue.
    """

    def __init__(self, station_name="Reticulum Radio"):
        """
        Initialize the broadcaster.

        Args:
            station_name: Name of the radio station
        """
        self.station_name = station_name
        self.reticulum = None
        self.identity = None
        self.destination_high = None  # High quality broadcast (Opus)
        self.destination_low = None   # Low quality broadcast (EnCodec 3.78 kbps)
        self.control_destination = None  # Control destination (SINGLE)

        # LXST components
        self.mic_source = None
        self.mp3_source = None
        self.current_source = None  # Active source (mic or mp3)
        self.mixer = None
        self.pipeline = None
        self.opus_encoder = None
        self.codec2_encoder = None
        self.broadcast_sink_high = None  # High quality broadcast sink (Opus)
        self.broadcast_sink_low = None   # Low quality broadcast sink (EnCodec)
        self.switching_source = False  # Flag to prevent race conditions

        # Audio state
        self.mp3_queue = MP3Queue()
        self.music_playing = False
        self.music_paused = False
        self.mic_enabled = True  # Always on by default

        # Threading
        self.running = Event()
        self.audio_thread = None

    def setup_reticulum(self):
        """Initialize Reticulum network stack."""
        print(f"Initializing Reticulum for {self.station_name}...")
        self.reticulum = RNS.Reticulum()

        # Create or load identity
        identity_path = Path.home() / ".reticulum" / "identities" / "radio_broadcaster"

        if identity_path.exists():
            self.identity = RNS.Identity.from_file(str(identity_path))
            print(f"Loaded existing identity")
        else:
            self.identity = RNS.Identity()
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            self.identity.to_file(str(identity_path))
            print(f"Created new identity")

        print(f"Broadcaster Identity: {RNS.prettyhexrep(self.identity.hash)}")

        # Append truncated identity hash to station name for verification
        # This allows listeners to verify they're tuned to the correct broadcaster
        # Strip angle brackets from prettyhexrep output
        identity_hex = RNS.prettyhexrep(self.identity.hash).strip('<>')
        identity_suffix = identity_hex[:8]  # First 8 hex chars (4 bytes)
        self.full_station_name = f"{self.station_name}-{identity_suffix}"

        # Create broadcast destinations (PLAIN type for public broadcast)
        # Two quality levels: High (Opus) and Low (EnCodec 3.78 kbps)

        # High quality destination - Opus stereo music
        self.destination_high = RNS.Destination(
            None,  # No identity for PLAIN destinations
            RNS.Destination.IN,
            RNS.Destination.PLAIN,
            APP_NAME,
            BROADCAST_ASPECT,
            self.full_station_name,  # Channel name with identity suffix
            "high"  # Quality level
        )

        # Low quality destination - EnCodec neural codec at 3.78 kbps
        self.destination_low = RNS.Destination(
            None,  # No identity for PLAIN destinations
            RNS.Destination.IN,
            RNS.Destination.PLAIN,
            APP_NAME,
            BROADCAST_ASPECT,
            self.full_station_name,  # Channel name with identity suffix
            "low"  # Quality level
        )

        print(f"High Quality Destination (Opus): {RNS.prettyhexrep(self.destination_high.hash)}")
        print(f"Low Quality Destination (EnCodec 6kbps → ~3.78kbps actual): {RNS.prettyhexrep(self.destination_low.hash)}")
        print(f"Broadcasting on channel: {self.full_station_name}")

        # Create control destination (SINGLE type for encrypted key exchange)
        # Listeners connect here first to get public key before tuning to broadcast
        self.control_destination = RNS.Destination(
            self.identity,
            RNS.Destination.IN,
            RNS.Destination.SINGLE,
            APP_NAME,
            CONTROL_ASPECT
        )

        # Register request handler for control channel
        self.control_destination.set_link_established_callback(self.control_link_established)

        print(f"Control Destination: {RNS.prettyhexrep(self.control_destination.hash)}")
        print(f"Identity suffix: {identity_suffix} (use this to verify authenticity)")
        print("Control channel: Encrypted key exchange")
        print("Data channel: Signed broadcasts (Ed25519)")

    def setup_lxst(self):
        """Initialize LXST audio pipeline for streaming."""
        print("Setting up LXST audio pipeline...")

        # Create microphone source with 20ms frames for Opus compatibility
        # EnCodec will buffer frames internally to reach 2000ms for optimal quality
        self.mic_source = LineSource(target_frame_ms=20)

        # Wrap the mic source's codec to track when frames are sent/dropped
        original_mic_codec = None
        frame_stats = {'sent': 0, 'dropped': 0, 'last_report': time.time()}

        def track_encoding():
            # This will be called after codec is set by pipeline
            if self.mic_source.codec and original_mic_codec != self.mic_source.codec:
                original_encode = self.mic_source.codec.encode
                def tracked_encode(frame):
                    result = original_encode(frame)
                    # Check if frame will actually be sent
                    if self.mic_source.sink and self.mic_source.sink.can_receive(from_source=self.mic_source):
                        frame_stats['sent'] += 1
                    else:
                        frame_stats['dropped'] += 1

                    # Report every 4 seconds
                    if time.time() - frame_stats['last_report'] >= 4.0:
                        total = frame_stats['sent'] + frame_stats['dropped']
                        if total > 0:
                            drop_pct = (frame_stats['dropped'] / total) * 100
                            print(f"Frame stats: {frame_stats['sent']} sent, {frame_stats['dropped']} dropped ({drop_pct:.1f}%)")
                        frame_stats['sent'] = 0
                        frame_stats['dropped'] = 0
                        frame_stats['last_report'] = time.time()

                    return result
                self.mic_source.codec.encode = tracked_encode

        # Check periodically if codec is set
        import threading
        def check_codec():
            time.sleep(2)  # Wait for pipeline to initialize
            track_encoding()
        threading.Thread(target=check_codec, daemon=True).start()

        self.current_source = self.mic_source

        # Create encoders for dual-quality broadcasting
        # High quality: Opus stereo music (PROFILE_AUDIO_HIGH)
        self.opus_encoder = Opus(profile=Opus.PROFILE_AUDIO_HIGH)

        # Low quality: EnCodec 24kHz mono at 1.5 kbps (no post-compression)
        # Streamable mode: buffers to 5s frames, signs once per frame
        # 1.5 kbps → ~1.6 kbps actual (fits well within 3.1 kbps LoRa limit)
        # Android compatible (no PyTorch language model needed)
        self.low_quality_encoder = EnCodecStreamable(
            bandwidth=1.5,
            broadcaster_identity=self.identity,
            compression=None  # No post-compression needed at 1.5 kbps
        )

        # Create forked broadcast sink to send both qualities
        destinations = {
            'high': self.destination_high,
            'low': self.destination_low
        }
        codecs = {
            'high': self.opus_encoder,
            'low': self.low_quality_encoder
        }

        # Batch sizes for each stream:
        # High quality: 2 frames/packet (Opus 20ms frames)
        # Low quality: 1 chunk/packet (EnCodec buffers 5s, encodes, chunks to 350 bytes)
        self.broadcast_sink_forked = ForkedBroadcastSink(
            destinations,
            self.identity,
            codecs,
            batch_sizes={'high': 2, 'low': 1}
        )
        self.broadcast_sink_forked.source = self.current_source

        # Use passthrough codec so pipeline doesn't encode
        # ForkedBroadcastSink will handle encoding with both Opus and Codec2
        passthrough_codec = RawPassthroughCodec()

        # Set up pipeline: Source -> Passthrough -> ForkedBroadcastSink
        # The sink receives raw audio and encodes with both codecs
        self.pipeline = Pipeline(self.current_source, passthrough_codec, self.broadcast_sink_forked)

        # Start the broadcast sink
        self.broadcast_sink_forked.start()

        # Start the pipeline
        self.pipeline.start()

        print(f"LXST dual-quality pipeline created and started for: {self.station_name}")
        print(f"  High quality (Opus): {RNS.prettyhexrep(self.destination_high.hash)}")
        print(f"  Low quality (EnCodec 3.78kbps): {RNS.prettyhexrep(self.destination_low.hash)}")

    def switch_source(self, new_source):
        """Switch the audio source to mic or MP3."""
        if self.switching_source:
            return

        self.switching_source = True
        try:
            # Stop the current pipeline
            if self.pipeline and self.pipeline.running:
                self.pipeline.stop()

            # Stop the current source
            if self.current_source and self.current_source.should_run:
                self.current_source.stop()

            # Update to new source
            self.current_source = new_source

            # Update the codec's source
            if self.opus_encoder:
                self.opus_encoder.source = new_source

            # Update the broadcast sink's source reference
            if self.broadcast_sink_forked:
                self.broadcast_sink_forked.source = new_source

            # Use passthrough codec for pipeline
            passthrough_codec = RawPassthroughCodec()

            # Recreate the pipeline with the new source and forked broadcast sink
            self.pipeline = Pipeline(new_source, passthrough_codec, self.broadcast_sink_forked)

            # Update sink references
            if self.current_source:
                self.current_source.sink = self.broadcast_sink_forked

            # Start the new pipeline
            self.pipeline.start()

        finally:
            self.switching_source = False

    def audio_mixer_loop(self):
        """
        Main audio mixing loop.
        Handles switching between microphone and MP3 playback.
        """
        print("Audio mixer started...")

        while self.running.is_set():
            try:
                # Check if we should play music
                if self.music_playing and not self.music_paused:
                    current_track = self.mp3_queue.get_current()

                    if current_track:
                        # Play current MP3 track
                        self.stream_mp3(current_track)

                        # Check if track finished
                        if self.mp3_source and self.mp3_source.finished:
                            # Move to next track when done
                            if self.mp3_queue.has_next():
                                next_track = self.mp3_queue.next()
                                print(f"\nNext track: {os.path.basename(next_track)}")
                            else:
                                # Queue finished
                                self.music_playing = False
                                print("\nQueue finished. Switching to microphone...")
                                self.switch_source(self.mic_source)
                    else:
                        # No current track, switch to mic
                        self.music_playing = False
                        self.switch_source(self.mic_source)

                # Switch to microphone when music paused or not playing
                if (not self.music_playing or self.music_paused) and self.current_source != self.mic_source:
                    print("\nSwitching to microphone...")
                    self.switch_source(self.mic_source)

                time.sleep(0.5)

            except Exception as e:
                print(f"Error in audio mixer: {e}")
                import traceback
                traceback.print_exc()

    def stream_mp3(self, mp3_path):
        """
        Stream an MP3 file through LXST.

        Args:
            mp3_path: Path to the MP3 file
        """
        # If we're already streaming this file, don't reload
        if self.mp3_source and self.mp3_source.file_path == mp3_path and not self.mp3_source.finished:
            return

        # If this is a new file or the previous one finished, load it
        if not self.mp3_source or self.mp3_source.file_path != mp3_path or self.mp3_source.finished:
            try:
                # Create MP3 source (codec and sink will be set by Pipeline)
                self.mp3_source = MP3Source(mp3_path)

                # Switch to MP3 source
                print(f"Now playing: {os.path.basename(mp3_path)}")
                self.switch_source(self.mp3_source)

            except Exception as e:
                print(f"Error loading MP3: {e}")
                self.music_playing = False
                import traceback
                traceback.print_exc()

    def play_music(self):
        """Start playing music from queue."""
        if not self.mp3_queue.queue:
            print("Queue is empty. Add tracks first.")
            return

        if not self.music_playing:
            # Start from beginning if not already playing
            self.mp3_queue.current_index = 0

        self.music_playing = True
        self.music_paused = False
        print("Playing music...")

    def pause_music(self):
        """Pause music playback (switches to mic)."""
        if self.music_playing:
            self.music_paused = True
            print("Music paused. Microphone active.")
        else:
            print("No music playing.")

    def resume_music(self):
        """Resume music playback."""
        if self.music_playing and self.music_paused:
            self.music_paused = False
            print("Music resumed.")
        else:
            print("Use 'play' to start music.")

    def add_to_queue(self, mp3_path):
        """Add MP3 to queue."""
        self.mp3_queue.add(mp3_path)

    def control_link_established(self, link):
        """Called when a listener connects to the control channel."""
        print(f"Control: Listener connected from {RNS.prettyhexrep(link.destination.hash)}")
        link.set_link_closed_callback(self.control_link_closed)
        link.set_packet_callback(self.control_packet_received)

    def control_link_closed(self, link):
        """Called when control link is closed."""
        print(f"Control: Listener disconnected")

    def control_packet_received(self, message, packet):
        """Handle requests on control channel."""
        try:
            request = mp.unpackb(message)
            request_type = request.get('type')

            if request_type == 'get_info':
                # Listener is requesting station info and public key
                response = {
                    'station_name': self.full_station_name,
                    'public_key': self.identity.get_public_key(),
                    'streams': {
                        'high': {
                            'codec': 'opus',
                            'quality': 'High quality stereo music (Opus)',
                            'hash': self.destination_high.hash
                        },
                        'low': {
                            'codec': 'encodec',
                            'quality': 'LoRa-optimized (EnCodec 24kHz mono 6 kbps → ~3.78 kbps with entropy coding)',
                            'hash': self.destination_low.hash
                        }
                    }
                }
                response_data = mp.packb(response)

                # Send response back over the link
                response_packet = RNS.Packet(packet.link, response_data)
                response_packet.send()

                print(f"Control: Sent station info to listener")
            else:
                print(f"Control: Unknown request type: {request_type}")

        except Exception as e:
            print(f"Control: Error handling request: {e}")
            import traceback
            traceback.print_exc()

    def show_status(self):
        """Display current broadcast status."""
        print("\n" + "="*60)
        print(f"  {self.station_name} - Status")
        print("="*60)
        print(f"Microphone: {'LIVE' if (not self.music_playing or self.music_paused) else 'standby'}")
        print(f"Music: {'PLAYING' if (self.music_playing and not self.music_paused) else 'PAUSED' if self.music_paused else 'stopped'}")
        print(self.mp3_queue.get_queue_status())
        print("="*60 + "\n")

    def command_interface(self):
        """Interactive command interface for DJ controls."""
        print("\nDJ Controls:")
        print("  add <path>  - Add MP3 to queue")
        print("  play        - Start playing queue")
        print("  pause       - Pause music (mic goes live)")
        print("  resume      - Resume music")
        print("  status      - Show current status")
        print("  quit        - Stop broadcasting")
        print()

        while self.running.is_set():
            try:
                cmd = input("DJ> ").strip().split(maxsplit=1)

                if not cmd:
                    continue

                command = cmd[0].lower()

                if command == "help":
                    print("\nDJ Controls:")
                    print("  add <path>  - Add MP3 to queue")
                    print("  play        - Start playing queue")
                    print("  pause       - Pause music (mic goes live)")
                    print("  resume      - Resume music")
                    print("  status      - Show current status")
                    print("  quit        - Stop broadcasting")
                    print()
                elif command == "add" and len(cmd) > 1:
                    self.add_to_queue(cmd[1])
                elif command == "play":
                    self.play_music()
                elif command == "pause":
                    self.pause_music()
                elif command == "resume":
                    self.resume_music()
                elif command == "status":
                    self.show_status()
                elif command in ["quit", "exit"]:
                    print("Shutting down...")
                    self.running.clear()
                    break
                else:
                    print("Unknown command. Type 'help' for commands.")

            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.running.clear()
                break

    def start(self):
        """Start the broadcaster."""
        self.setup_reticulum()
        self.setup_lxst()

        print(f"\n{'='*60}")
        print(f"  {self.station_name} - ON AIR")
        print(f"{'='*60}")
        print(f"  Microphone: LIVE (always active)")
        print(f"  Music Queue: Ready")
        print(f"{'='*60}\n")

        self.running.set()

        # Start audio mixer thread
        self.audio_thread = Thread(target=self.audio_mixer_loop, daemon=True)
        self.audio_thread.start()

        # Announce control destination (SINGLE) so listeners can find us
        # Data channel (PLAIN) doesn't need announcements
        self.control_destination.announce()
        print(f"\n{'='*60}")
        print("CONTROL CHANNEL ANNOUNCED")
        print(f"{'='*60}")

        # Show the control destination hash without angle brackets for easy copy-paste
        control_hash_hex = RNS.prettyhexrep(self.control_destination.hash).strip('<>').replace(':', '')
        print(f"\nListeners: Copy this hash to connect:")
        print(f"  {control_hash_hex}")
        print(f"\nCommand:")
        print(f"  python listener.py --station {control_hash_hex}")
        print(f"\n{'='*60}\n")

        # Run command interface on main thread
        self.command_interface()

    def shutdown(self):
        """Clean shutdown of broadcaster."""
        self.running.clear()

        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)

        # Stop the forked broadcast sink
        if hasattr(self, 'broadcast_sink_forked') and self.broadcast_sink_forked:
            self.broadcast_sink_forked.stop()

        if self.pipeline:
            print("Shutting down LXST pipeline...")
            self.pipeline.stop()

        print("Broadcaster stopped.")


def main():
    """Main entry point for broadcaster CLI."""
    parser = argparse.ArgumentParser(description="Reticulum Radio Broadcaster")
    parser.add_argument(
        "--station-name",
        type=str,
        default="Reticulum Radio",
        help="Name of your radio station"
    )

    args = parser.parse_args()

    broadcaster = RadioBroadcaster(station_name=args.station_name)

    try:
        broadcaster.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        broadcaster.shutdown()


if __name__ == "__main__":
    main()
