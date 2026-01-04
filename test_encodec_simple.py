#!/usr/bin/env python3
"""
Simple EnCodec Test for Termux/Android

Verifies that EnCodec can run on your Android device.
Tests model loading and basic decode performance.

Usage:
    python test_encodec_simple.py
"""

import sys
import time
import os

print("=" * 70)
print("EnCodec Simple Test for Termux/Android")
print("=" * 70)
print()

# Detect environment
IS_TERMUX = os.environ.get('TERMUX_VERSION') is not None
print(f"Environment: {'Termux' if IS_TERMUX else 'Desktop'}")
print()

# Test PyTorch import
print("Testing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except ImportError as e:
    print(f"✗ PyTorch not found: {e}")
    print()
    print("Install with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("Or:")
    print("  pip install torch -f https://torch.kmtea.eu/whl/stable.html")
    sys.exit(1)

# Test NumPy
print()
print("Testing NumPy...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not found")
    print("Install with: pip install numpy")
    sys.exit(1)

# Test EnCodec
print()
print("Testing EnCodec...")
try:
    from encodec import EncodecModel, compress, decompress
    print(f"✓ EnCodec library")
except ImportError:
    print("✗ EnCodec not found")
    print("Install with: pip install encodec")
    sys.exit(1)

print()
print("-" * 70)
print("All imports successful! Loading model...")
print("-" * 70)
print()

# Load model
try:
    print("Loading EnCodec 24kHz model (this may take 10-30 seconds)...")
    start_time = time.time()

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(3.0)
    model = model.to('cpu')  # Force CPU on mobile
    model.eval()

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.1f}s")

except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("-" * 70)
print("Testing encode/decode performance...")
print("-" * 70)
print()

# Create test audio (5 seconds)
sample_rate = 24000
duration = 5
num_samples = sample_rate * duration

print(f"Creating {duration}s test audio...")
freq = 440.0  # A4 note
t = np.linspace(0, duration, num_samples)
test_audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
test_tensor = torch.from_numpy(test_audio).unsqueeze(0)
print(f"✓ Audio: {test_tensor.shape}")
print()

# Test encode
print("Testing encode (without entropy coding)...")
try:
    with torch.no_grad():
        encode_start = time.time()
        compressed_bytes = compress(model, test_tensor, use_lm=False)
        encode_time = time.time() - encode_start

    compressed_size = len(compressed_bytes)
    bitrate = (compressed_size * 8) / (duration * 1000)

    print(f"✓ Encoded in {encode_time:.2f}s")
    print(f"  Size: {compressed_size} bytes")
    print(f"  Bitrate: {bitrate:.2f} kbps")

except Exception as e:
    print(f"✗ Encode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test decode
print("Testing decode...")
try:
    with torch.no_grad():
        decode_start = time.time()
        decoded, sr = decompress(compressed_bytes, device='cpu')
        decode_time = time.time() - decode_start

    realtime_factor = decode_time / duration

    print(f"✓ Decoded in {decode_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.1%}")
    print(f"  Output shape: {decoded.shape}")

    # Performance assessment
    if realtime_factor < 0.2:
        verdict = "✓ EXCELLENT - Very fast"
    elif realtime_factor < 0.5:
        verdict = "✓ GOOD - Should work for real-time"
    elif realtime_factor < 1.0:
        verdict = "⚠ MARGINAL - May have dropouts"
    else:
        verdict = "✗ TOO SLOW - Not usable for real-time"

    print(f"  Performance: {verdict}")

except Exception as e:
    print(f"✗ Decode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with zlib compression
print()
print("-" * 70)
print("Testing with zlib compression...")
print("-" * 70)
print()

try:
    import zlib

    compressed_zlib = zlib.compress(compressed_bytes, level=6)
    bitrate_zlib = (len(compressed_zlib) * 8) / (duration * 1000)
    compression_ratio = len(compressed_zlib) / len(compressed_bytes)

    print(f"✓ zlib compression:")
    print(f"  Original: {compressed_size} bytes ({bitrate:.2f} kbps)")
    print(f"  Compressed: {len(compressed_zlib)} bytes ({bitrate_zlib:.2f} kbps)")
    print(f"  Ratio: {compression_ratio:.1%}")

    # Check if fits in LoRa bandwidth
    if bitrate_zlib <= 3.1:
        margin = 3.1 - bitrate_zlib
        print(f"  ✓ FITS in 3.1 kbps LoRa (margin: {margin:.2f} kbps)")
    else:
        over = bitrate_zlib - 3.1
        print(f"  ✗ TOO HIGH for 3.1 kbps LoRa (over by: {over:.2f} kbps)")

except Exception as e:
    print(f"✗ zlib test failed: {e}")

# Final verdict
print()
print("=" * 70)
print("FINAL VERDICT")
print("=" * 70)
print()

if realtime_factor < 1.0:
    print("✓ EnCodec CAN run on this device!")
    print()
    print("Next steps:")
    print("  1. Install remaining dependencies:")
    print("     pip install sounddevice rns lxst")
    print()
    print("  2. Run the listener:")
    print("     python src/listener.py --station <control_dest_hash> --quality low")
    print()
    if realtime_factor > 0.5:
        print("  Note: Performance is marginal. You may experience:")
        print("    - Longer buffering times")
        print("    - Occasional dropouts")
        print("    - Higher battery usage")
else:
    print("✗ EnCodec is TOO SLOW on this device")
    print()
    print(f"Decode takes {realtime_factor:.1f}x longer than audio playback.")
    print("This device cannot keep up with real-time decoding.")
    print()
    print("Alternatives:")
    print("  1. Use a more powerful Android device")
    print("  2. Use Opus codec instead (higher bandwidth but faster)")
    print("     python src/listener.py --station <hash> --quality high")

print()
print("=" * 70)
