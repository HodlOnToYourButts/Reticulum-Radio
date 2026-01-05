#!/usr/bin/env python3
"""
EnCodec Codec for Termux/Android - No LXST Dependency

Simplified EnCodec decoder for Termux that doesn't require LXST.
Only implements decoding (listener side) functionality.

Based on: High Fidelity Neural Audio Compression (arXiv:2210.13438)
"""

import numpy as np
import torch
import RNS
import struct
import time
import zlib
import lzma
from encodec import EncodecModel, decompress
from encodec.utils import convert_audio


class EnCodecStreamable:
    """
    Streamable EnCodec codec for Termux (decoder only).
    No LXST dependency - works standalone.
    """

    # Frame size
    FRAME_SIZE_MS = 5000  # 5 second frames

    def __init__(self, bandwidth=1.5, use_48khz=False, stereo=False, force_cpu=False, compression=None):
        """
        Initialize streamable EnCodec codec (decoder only for Termux).

        Args:
            bandwidth: Target bandwidth in kbps (1.5, 3, 6, 12, 24)
            use_48khz: Use 48kHz model (vs 24kHz)
            stereo: Use stereo (vs mono)
            force_cpu: Force CPU-only mode (disable GPU)
            compression: Post-compression method ('zlib', 'lzma', or None)
        """
        # Compression settings
        self.compression = compression
        if compression not in ('zlib', 'lzma', None):
            raise ValueError(f"compression must be 'zlib', 'lzma', or None, got: {compression}")

        # EnCodec model selection
        self.use_48khz = use_48khz
        self.stereo = stereo
        self.bandwidth = bandwidth

        if use_48khz:
            self.model_samplerate = 48000
            self.output_samplerate = 48000
        else:
            self.model_samplerate = 24000
            self.output_samplerate = 48000  # Upsample to 48kHz for playback

        if stereo:
            self.channels = 2
            self.output_channels = 2
        else:
            self.channels = 1
            self.output_channels = 1

        # Load EnCodec model
        model_desc = f"{'48kHz' if use_48khz else '24kHz'} {'stereo' if stereo else 'mono'}"
        print(f"[EnCodec] Loading {model_desc} model (bandwidth: {bandwidth} kbps)...")

        # Force CPU on Android
        self.device = torch.device("cpu")
        print(f"[EnCodec] Using CPU (Android/Termux)")

        if use_48khz:
            self.model = EncodecModel.encodec_model_48khz()
        else:
            self.model = EncodecModel.encodec_model_24khz()

        # No language model (no entropy coding)
        self.model.lm = None

        self.model.set_target_bandwidth(bandwidth)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Decoder reassembly with security limits
        self.reassembly_buffer = {}  # {seq_num: {'chunks': {chunk_idx: chunk_data}, 'total': total_chunks, 'timestamp': time}}
        self.last_complete_seq = -1
        self.MAX_INCOMPLETE_SEQUENCES = 10
        self.SEQUENCE_TIMEOUT = 10.0

        # Security statistics
        self.dropped_malicious = 0
        self.dropped_expired = 0
        self.dropped_overflow = 0

        # Identity for signature verification
        self.expected_identity = None

        compression_name = compression.upper() if compression else "None"
        print(f"[EnCodec] Initialized: {self.output_samplerate}Hz, {self.channels}ch, {bandwidth}kbps + {compression_name}")
        print(f"[EnCodec] Decoder ready for {self.FRAME_SIZE_MS}ms frames")

    def decode(self, encoded_data):
        """
        Decode EnCodec frame with security protections.
        Reassembles chunks, decodes complete frames, returns full 5-second chunks.

        Args:
            encoded_data: bytes containing chunked frame data

        Returns:
            numpy array of shape (samples, channels) with float32 values [-1.0, 1.0]
        """
        try:
            if len(encoded_data) == 0:
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Clean up expired sequences
            current_time = time.time()
            expired_seqs = [
                seq for seq, data in self.reassembly_buffer.items()
                if current_time - data['timestamp'] > self.SEQUENCE_TIMEOUT
            ]
            for seq in expired_seqs:
                del self.reassembly_buffer[seq]
                self.dropped_expired += 1

            # Parse chunk header: [seq:2][chunk_idx:1][total_chunks:1][data]
            if len(encoded_data) < 4:
                self.dropped_malicious += 1
                return np.zeros((0, self.output_channels), dtype=np.float32)

            seq_num, chunk_idx, total_chunks = struct.unpack('>HBB', encoded_data[:4])
            chunk_data = encoded_data[4:]

            # SECURITY: Validate total_chunks
            if total_chunks == 0 or total_chunks > 20:
                self.dropped_malicious += 1
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Validate chunk_idx
            if chunk_idx >= total_chunks:
                self.dropped_malicious += 1
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Enforce buffer size limit
            if seq_num not in self.reassembly_buffer:
                if len(self.reassembly_buffer) >= self.MAX_INCOMPLETE_SEQUENCES:
                    oldest_seq = min(self.reassembly_buffer.keys())
                    del self.reassembly_buffer[oldest_seq]
                    self.dropped_overflow += 1

                self.reassembly_buffer[seq_num] = {
                    'chunks': {},
                    'total': total_chunks,
                    'timestamp': current_time
                }

            seq_data = self.reassembly_buffer[seq_num]

            # SECURITY: Consistency check
            if seq_data['total'] != total_chunks:
                self.dropped_malicious += 1
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # Store chunk
            seq_data['chunks'][chunk_idx] = chunk_data

            # Check if complete
            if len(seq_data['chunks']) < total_chunks:
                return np.zeros((0, self.output_channels), dtype=np.float32)

            # SECURITY: Verify all chunks present
            chunks = seq_data['chunks']
            for i in range(total_chunks):
                if i not in chunks:
                    self.dropped_malicious += 1
                    del self.reassembly_buffer[seq_num]
                    return np.zeros((0, self.output_channels), dtype=np.float32)

            # Reassemble
            complete_data = b''.join(chunks[i] for i in range(total_chunks))

            # Clean up
            del self.reassembly_buffer[seq_num]
            self.last_complete_seq = seq_num

            # Verify signature if we have expected identity
            if self.expected_identity:
                if len(complete_data) < 64:
                    return np.zeros((0, self.output_channels), dtype=np.float32)

                signature = complete_data[:64]
                compressed_data = complete_data[64:]

                if not self.expected_identity.validate(signature, compressed_data):
                    print(f"[EnCodec] WARNING: Invalid signature! Frame rejected")
                    return np.zeros((0, self.output_channels), dtype=np.float32)
            else:
                compressed_data = complete_data

            # Decompress post-compression (zlib or lzma)
            if self.compression == 'zlib':
                compressed_data = zlib.decompress(compressed_data)
            elif self.compression == 'lzma':
                compressed_data = lzma.decompress(compressed_data)

            # Decompress EnCodec
            with torch.no_grad():
                decoded, sr = decompress(compressed_data, device=self.device)

            # Handle shape for upsampling
            if self.model_samplerate != 48000:
                if decoded.dim() == 2 and decoded.shape[0] == 1:
                    decoded = decoded.unsqueeze(1)

                decoded_cpu = decoded[0].cpu()

                # Upsample to 48kHz
                decoded_audio_tensor = convert_audio(
                    decoded_cpu,
                    self.model_samplerate,
                    48000,
                    self.output_channels
                )
                decoded_audio = decoded_audio_tensor.numpy().T
            else:
                if decoded.dim() == 2 and decoded.shape[0] == 1:
                    decoded = decoded.unsqueeze(1)
                decoded_audio = decoded[0].cpu().numpy().T

            # Ensure 2D shape
            if decoded_audio.ndim == 1:
                decoded_audio = decoded_audio.reshape(-1, 1)

            return decoded_audio.astype(np.float32)

        except Exception as e:
            print(f"[EnCodec] Decoding error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((0, self.output_channels), dtype=np.float32)
