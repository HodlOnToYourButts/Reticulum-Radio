#!/usr/bin/env python3
"""
EnCodec LXST Codec - Proper Streamable/Causal Implementation

Uses EnCodec's causal model for real-time radio streaming:
- Frame-by-frame encoding/decoding (20ms frames)
- No large buffering (designed for live streaming)
- Low latency
- With entropy coding for better compression

Based on: High Fidelity Neural Audio Compression (arXiv:2210.13438)
"""

import numpy as np
import torch
import RNS
import struct
import time
import zlib
import lzma
from encodec import EncodecModel, compress, decompress
from encodec.utils import convert_audio
from LXST.Codecs import Codec

# Codec identifier for LXST
ENCODEC = 3


class EnCodecStreamable(Codec):
    """
    Streamable EnCodec codec for LXST pipeline.
    Uses causal model with 5-second frames for better compression.
    """

    # Buffer incoming 20ms frames to 5 seconds before encoding
    FRAME_SIZE_MS = 5000  # 5 second frames

    def __init__(self, bandwidth=3.0, use_48khz=False, stereo=False, force_cpu=False, broadcaster_identity=None, compression='zlib'):
        """
        Initialize streamable EnCodec codec.

        Args:
            bandwidth: Target bandwidth in kbps (1.5, 3, 6, 12, 24)
            use_48khz: Use 48kHz model (vs 24kHz)
            stereo: Use stereo (vs mono)
            force_cpu: Force CPU-only mode (disable GPU)
            broadcaster_identity: RNS.Identity for signing (encoder only)
            compression: Post-compression method ('zlib', 'lzma', or None)
                        - 'zlib': Fast, good compression (recommended for LoRa)
                        - 'lzma': Slower but better compression
                        - None: No post-compression (for testing)
        """
        # Initialize parent Codec class
        super().__init__()

        # Compression settings
        self.compression = compression
        if compression not in ('zlib', 'lzma', None):
            raise ValueError(f"compression must be 'zlib', 'lzma', or None, got: {compression}")

        # Identity for signing encoded frames (encoder only)
        self.broadcaster_identity = broadcaster_identity

        # EnCodec model selection
        self.use_48khz = use_48khz
        self.stereo = stereo
        self.bandwidth = bandwidth

        if use_48khz:
            self.preferred_samplerate = 48000
            self.model_samplerate = 48000  # Model's native rate
            self.output_samplerate = 48000  # Output to LXST
        else:
            self.preferred_samplerate = 24000
            self.model_samplerate = 24000  # Model's native rate
            self.output_samplerate = 48000  # Upsample to 48kHz for LXST compatibility

        self.input_samplerate = 48000  # Will be updated from source

        if stereo:
            self.channels = 2
            self.input_channels = 2
            self.output_channels = 2
        else:
            self.channels = 1
            self.input_channels = 1
            self.output_channels = 1

        self.bitdepth = 16

        # Load causal (streamable) EnCodec model
        model_desc = f"{'48kHz' if use_48khz else '24kHz'} {'stereo' if stereo else 'mono'}"
        RNS.log(f"Loading EnCodec {model_desc} causal/streamable model (bandwidth: {bandwidth} kbps)...", RNS.LOG_INFO)

        # Use GPU if available
        if force_cpu:
            self.device = torch.device("cpu")
            RNS.log(f"EnCodec: Forced CPU-only mode", RNS.LOG_INFO)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_48khz:
            self.model = EncodecModel.encodec_model_48khz()
        else:
            self.model = EncodecModel.encodec_model_24khz()  # Returns causal model

        # NOTE: Not using entropy coding (Transformer language model)
        # Instead using zlib/lzma post-compression for Android compatibility
        # This avoids needing PyTorch on the decoder side
        self.model.lm = None  # Explicitly disable language model

        self.model.set_target_bandwidth(bandwidth)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        if torch.cuda.is_available() and not force_cpu:
            gpu_name = torch.cuda.get_device_name(0)
            RNS.log(f"EnCodec: Using GPU acceleration: {gpu_name}", RNS.LOG_INFO)

            # Enable TF32 for faster matmul on Ampere+ GPUs
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                RNS.log(f"EnCodec: Enabled TF32 for faster GPU operations", RNS.LOG_INFO)
            except:
                pass

        # Track statistics
        self.output_bytes = 0
        self.frames_encoded = 0

        # Frame buffering: accumulate 20ms frames to 1 second before encoding
        self.frame_buffer = []
        self.buffer_samples = 0
        self.buffer_target_samples = int((self.FRAME_SIZE_MS / 1000) * self.input_samplerate)

        # Chunking for MTU compliance (500 byte MTU)
        # Chunk format: [seq:2 bytes][chunk_idx:1 byte][total_chunks:1 byte][data:~475 bytes]
        # RNS.Packet adds ~19 bytes overhead, so: 475 data + 4 header + 19 RNS = 498 total
        self.MAX_CHUNK_SIZE = 475  # Maximize MTU usage while accounting for RNS packet overhead
        self.chunk_queue = []  # Queue of chunks ready to send
        self.sequence_number = 0  # Increments for each encoded frame

        # Decoder reassembly with security limits
        self.reassembly_buffer = {}  # {seq_num: {'chunks': {chunk_idx: chunk_data}, 'total': total_chunks, 'timestamp': time}}
        self.last_complete_seq = -1
        self.MAX_INCOMPLETE_SEQUENCES = 10  # Limit memory usage from malicious packets
        self.SEQUENCE_TIMEOUT = 10.0  # Expire incomplete sequences after 10 seconds

        # Security statistics
        self.dropped_malicious = 0
        self.dropped_expired = 0
        self.dropped_overflow = 0

        # Identity for signature verification (decoder only)
        self.expected_identity = None  # Set by listener after control channel handshake

        compression_name = compression.upper() if compression else "None"
        RNS.log(f"EnCodec initialized: {self.output_samplerate}Hz, {self.channels}ch, {bandwidth}kbps + {compression_name}", RNS.LOG_INFO)
        RNS.log(f"EnCodec: Causal model with {self.FRAME_SIZE_MS}ms frames for quality + low latency", RNS.LOG_INFO)
        RNS.log(f"EnCodec: Using {compression_name} post-compression (no entropy coding for Android compatibility)", RNS.LOG_INFO)
        RNS.log(f"EnCodec: Chunking to {self.MAX_CHUNK_SIZE} bytes for MTU compliance", RNS.LOG_INFO)

    def encode(self, frame):
        """
        Encode audio frame using streamable EnCodec.
        Buffers 20ms frames to 1 second, then encodes.

        Args:
            frame: numpy array of shape (samples, channels) with float32 values [-1.0, 1.0]

        Returns:
            bytes: Encoded frame data (chunk from queue or empty)
        """
        try:
            if frame.shape[1] == 0:
                raise ValueError("Cannot encode frame with 0 channels")

            # Get input sample rate from source
            if hasattr(self.source, 'samplerate'):
                self.input_samplerate = self.source.samplerate
                self.buffer_target_samples = int((self.FRAME_SIZE_MS / 1000) * self.input_samplerate)

            # Convert to mono if needed
            if not self.stereo and frame.shape[1] > 1:
                frame = frame[:, 0:1]  # Take first channel
            elif self.stereo and frame.shape[1] == 1:
                # Duplicate mono to stereo if needed
                frame = np.hstack([frame, frame])

            # Add frame to buffer
            self.frame_buffer.append(frame)
            self.buffer_samples += frame.shape[0]

            # Check if we have enough samples to encode
            if self.buffer_samples >= self.buffer_target_samples:
                # Concatenate buffered frames
                audio_np = np.vstack(self.frame_buffer)

                # Clear buffer
                self.frame_buffer = []
                self.buffer_samples = 0

                # Convert numpy to torch tensor (channels, samples)
                audio_tensor = torch.from_numpy(audio_np.T).float()

                # Resample if needed
                if self.input_samplerate != self.model_samplerate:
                    audio_tensor = convert_audio(
                        audio_tensor,
                        self.input_samplerate,
                        self.model_samplerate,
                        self.channels
                    )

                # compress() expects (channels, samples) not (batch, channels, samples)
                audio_tensor = audio_tensor.to(self.device)

                # Encode WITHOUT entropy coding (for Android compatibility)
                # Using zlib/lzma post-compression instead
                with torch.no_grad():
                    compressed_bytes = compress(self.model, audio_tensor, use_lm=False)

                    # Synchronize if using GPU to ensure encoding completes
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                # Apply post-compression (zlib or lzma)
                if self.compression == 'zlib':
                    compressed_bytes = zlib.compress(compressed_bytes, level=6)
                elif self.compression == 'lzma':
                    compressed_bytes = lzma.compress(compressed_bytes, preset=1)
                # else: no post-compression

                # Track statistics
                self.output_bytes += len(compressed_bytes)
                self.frames_encoded += 1

                # Calculate bitrate
                frame_duration_ms = self.FRAME_SIZE_MS
                avg_bitrate = self.output_bytes * 8 / (self.frames_encoded * frame_duration_ms / 1000) / 1000

                # Report every 10 frames
                if self.frames_encoded % 10 == 0:
                    avg_size = self.output_bytes / self.frames_encoded
                    comp_name = self.compression.upper() if self.compression else "none"
                    print(f"[EnCodec] {avg_bitrate:.2f} kbps ({comp_name}) | Avg: {avg_size:.0f} bytes/frame | {self.frames_encoded} frames")

                # Sign the compressed frame if we have an identity (prevents spoofing)
                if self.broadcaster_identity:
                    signature = self.broadcaster_identity.sign(compressed_bytes)
                    # Format: [signature:64 bytes][compressed_data]
                    signed_data = signature + compressed_bytes
                else:
                    # No signing (for testing only - should always sign in production)
                    signed_data = compressed_bytes

                # Split signed data into chunks for MTU compliance
                num_chunks = (len(signed_data) + self.MAX_CHUNK_SIZE - 1) // self.MAX_CHUNK_SIZE
                seq = self.sequence_number
                self.sequence_number = (self.sequence_number + 1) % 65536  # 16-bit sequence number

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * self.MAX_CHUNK_SIZE
                    end = min(start + self.MAX_CHUNK_SIZE, len(signed_data))
                    chunk_data = signed_data[start:end]

                    # Create chunk with header: [seq:2][chunk_idx:1][total_chunks:1][data]
                    chunk = struct.pack('>HBB', seq, chunk_idx, num_chunks) + chunk_data
                    self.chunk_queue.append(chunk)

        except Exception as e:
            print(f"[EnCodec] Encoding error: {e}")
            import traceback
            traceback.print_exc()

        # Always return next chunk from queue (or empty if queue empty)
        # This ensures continuous stream of chunks after encoding
        if self.chunk_queue:
            return self.chunk_queue.pop(0)
        return b''

    def decode(self, encoded_data):
        """
        Decode EnCodec frame with security protections against malicious packets.
        Reassembles chunks, decodes complete frames, returns full 5-second chunks.

        Security features:
        - Buffer size limits to prevent memory exhaustion
        - Sequence expiration to clean up incomplete frames
        - Chunk validation to prevent KeyErrors
        - Consistency checking for total_chunks
        - Graceful handling of malicious/invalid packets

        Args:
            encoded_data: bytes containing chunked frame data

        Returns:
            numpy array of shape (samples, channels) with float32 values [-1.0, 1.0]
            Returns 240000 samples (5s at 48kHz) when frame is complete, or empty array otherwise
        """
        try:
            if len(encoded_data) == 0:
                # Return empty array (no silence) while buffering
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Clean up expired sequences (prevent memory buildup from incomplete frames)
            current_time = time.time()
            expired_seqs = [
                seq for seq, data in self.reassembly_buffer.items()
                if current_time - data['timestamp'] > self.SEQUENCE_TIMEOUT
            ]
            for seq in expired_seqs:
                del self.reassembly_buffer[seq]
                self.dropped_expired += 1
                if self.dropped_expired % 10 == 1:
                    print(f"[EnCodec Security] Expired {self.dropped_expired} incomplete sequences (timeout)")

            # Parse chunk header: [seq:2][chunk_idx:1][total_chunks:1][data]
            if len(encoded_data) < 4:
                self.dropped_malicious += 1
                if self.dropped_malicious <= 3 or self.dropped_malicious % 100 == 1:
                    print(f"[EnCodec Security] Invalid chunk size: {len(encoded_data)} (dropped {self.dropped_malicious} malicious)")
                return np.zeros((0, self.output_channels), dtype=np.float32)

            seq_num, chunk_idx, total_chunks = struct.unpack('>HBB', encoded_data[:4])
            chunk_data = encoded_data[4:]

            # SECURITY: Validate total_chunks is reasonable (prevent excessive memory allocation)
            if total_chunks == 0 or total_chunks > 20:
                self.dropped_malicious += 1
                if self.dropped_malicious <= 3 or self.dropped_malicious % 100 == 1:
                    print(f"[EnCodec Security] Invalid total_chunks: {total_chunks} (dropped {self.dropped_malicious} malicious)")
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Validate chunk_idx is within range
            if chunk_idx >= total_chunks:
                self.dropped_malicious += 1
                if self.dropped_malicious <= 3 or self.dropped_malicious % 100 == 1:
                    print(f"[EnCodec Security] Chunk index {chunk_idx} >= total {total_chunks} (dropped {self.dropped_malicious} malicious)")
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Enforce buffer size limit (prevent memory exhaustion DoS)
            if seq_num not in self.reassembly_buffer:
                if len(self.reassembly_buffer) >= self.MAX_INCOMPLETE_SEQUENCES:
                    # Drop oldest sequence
                    oldest_seq = min(self.reassembly_buffer.keys())
                    del self.reassembly_buffer[oldest_seq]
                    self.dropped_overflow += 1
                    if self.dropped_overflow % 10 == 1:
                        print(f"[EnCodec Security] Buffer overflow, dropped sequence {oldest_seq} (total: {self.dropped_overflow})")

                # Create new sequence entry
                self.reassembly_buffer[seq_num] = {
                    'chunks': {},
                    'total': total_chunks,
                    'timestamp': current_time
                }

            seq_data = self.reassembly_buffer[seq_num]

            # SECURITY: Consistency check - all chunks must agree on total_chunks
            if seq_data['total'] != total_chunks:
                self.dropped_malicious += 1
                if self.dropped_malicious <= 3 or self.dropped_malicious % 100 == 1:
                    print(f"[EnCodec Security] Inconsistent total_chunks for seq {seq_num}: {total_chunks} vs {seq_data['total']} (dropped {self.dropped_malicious} malicious)")
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # Store chunk (allows overwriting - last chunk wins)
            seq_data['chunks'][chunk_idx] = chunk_data

            # Check if we have all chunks for this sequence
            if len(seq_data['chunks']) < total_chunks:
                # Not complete yet, return empty array
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Verify all chunk indices are present (prevent KeyError)
            chunks = seq_data['chunks']
            for i in range(total_chunks):
                if i not in chunks:
                    # Missing chunk - attacker sent wrong total_chunks
                    self.dropped_malicious += 1
                    if self.dropped_malicious <= 3 or self.dropped_malicious % 100 == 1:
                        print(f"[EnCodec Security] Missing chunk {i}/{total_chunks} for seq {seq_num} (dropped {self.dropped_malicious} malicious)")
                    # Delete this sequence and reject
                    del self.reassembly_buffer[seq_num]
                    return np.zeros((0, self.output_channels), dtype=np.float32)

            # We have all chunks - reassemble
            complete_data = b''.join(chunks[i] for i in range(total_chunks))

            # Clean up old sequences
            del self.reassembly_buffer[seq_num]
            self.last_complete_seq = seq_num

            # Verify signature if we have expected identity (prevents spoofing)
            if self.expected_identity:
                if len(complete_data) < 64:
                    print(f"[EnCodec] Frame too short for signature: {len(complete_data)} bytes")
                    return np.zeros((0, self.output_channels), dtype=np.float32)

                # Extract signature and compressed data
                signature = complete_data[:64]
                compressed_data = complete_data[64:]

                # Verify signature
                if not self.expected_identity.validate(signature, compressed_data):
                    print(f"[EnCodec] WARNING: Invalid signature! Possible spoofed frame (rejected)")
                    return np.zeros((0, self.output_channels), dtype=np.float32)
            else:
                # No signature verification (testing only)
                compressed_data = complete_data

            # Decompress post-compression (zlib or lzma)
            if self.compression == 'zlib':
                compressed_data = zlib.decompress(compressed_data)
            elif self.compression == 'lzma':
                compressed_data = lzma.decompress(compressed_data)
            # else: no post-compression to remove

            # Decompress EnCodec using the standalone decompress function
            # Returns tuple: (decoded_tensor, sample_rate)
            with torch.no_grad():
                decoded, sr = decompress(compressed_data, device=self.device)

                # Synchronize if using GPU
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

            # decoded shape for mono: (1, samples) - need to reshape to (1, 1, samples)
            # decoded shape for stereo: (1, 2, samples) - already correct
            # For 24kHz model: need to upsample to 48kHz for LXST playback
            if self.model_samplerate != 48000:
                # Reshape for mono: (1, samples) -> (1, 1, samples)
                if decoded.dim() == 2 and decoded.shape[0] == 1:
                    decoded = decoded.unsqueeze(1)  # Add channel dimension: (1, 1, samples)

                # Move to CPU for resampling (convert_audio uses CPU resampler)
                decoded_cpu = decoded[0].cpu()

                # Resample from model's native rate (24kHz) to 48kHz for LXST
                # decoded_cpu is now (channels, samples) on CPU - correct for convert_audio
                decoded_audio_tensor = convert_audio(
                    decoded_cpu,  # (channels, samples) on CPU
                    self.model_samplerate,  # From 24kHz
                    48000,  # To 48kHz for LXST
                    self.output_channels
                )
                # Convert to numpy: (channels, samples) -> (samples, channels)
                decoded_audio = decoded_audio_tensor.numpy().T
            else:
                # Already at 48kHz
                # Reshape for mono if needed
                if decoded.dim() == 2 and decoded.shape[0] == 1:
                    decoded = decoded.unsqueeze(1)
                decoded_audio = decoded[0].cpu().numpy().T

            # Ensure 2D shape (samples, channels) for LXST
            if decoded_audio.ndim == 1:
                decoded_audio = decoded_audio.reshape(-1, 1)

            # Return the full 1-second decoded frame
            return decoded_audio.astype(np.float32)

        except Exception as e:
            print(f"[EnCodec] Decoding error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty array on error
            return np.zeros((0, self.output_channels), dtype=np.float32)


# Alias for backwards compatibility
EnCodecLXST = EnCodecStreamable
