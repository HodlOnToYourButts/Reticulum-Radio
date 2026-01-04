#!/usr/bin/env python3
"""
EnCodec Compression Comparison Test

Tests different EnCodec configurations to find what fits in 3.1 kbps LoRa bandwidth:
1. 3.0 kbps with entropy coding (current - baseline)
2. 3.0 kbps without entropy coding
3. 3.0 kbps without entropy + LZMA compression
4. 3.0 kbps without entropy + zlib compression
5. 1.5 kbps without entropy coding
6. 1.5 kbps without entropy + LZMA compression

Usage:
    python test_encodec_compression.py [audio_file.mp3]

If no audio file provided, uses synthetic test signal.
"""

import sys
import time
import numpy as np
import torch
import lzma
import zlib
from pathlib import Path

print("=" * 80)
print("EnCodec Compression Comparison for LoRa (3.1 kbps limit)")
print("=" * 80)
print()

# Check dependencies
try:
    from encodec import EncodecModel, compress, decompress
    from encodec.utils import convert_audio
    print("✓ EnCodec library loaded")
except ImportError:
    print("✗ EnCodec not found. Install with: pip install encodec")
    sys.exit(1)

# Load test audio
DURATION = 5  # seconds (matching your current 5-second frames)
SAMPLE_RATE = 24000  # EnCodec 24kHz model

if len(sys.argv) > 1:
    audio_file = Path(sys.argv[1])
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    print(f"Loading audio from: {audio_file}")
    try:
        import librosa
        audio, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, duration=DURATION, mono=True)
        print(f"✓ Loaded {len(audio)} samples at {sr}Hz")
    except ImportError:
        print("✗ librosa not found. Install with: pip install librosa")
        print("Falling back to synthetic test signal...")
        audio = None
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        print("Falling back to synthetic test signal...")
        audio = None
else:
    print("No audio file provided, using synthetic test signal")
    audio = None

# Generate synthetic audio if needed
if audio is None:
    print(f"Generating synthetic audio ({DURATION}s, {SAMPLE_RATE}Hz)...")
    # Complex signal: mix of frequencies to be more realistic than pure sine
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION)
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +      # A4
        0.3 * np.sin(2 * np.pi * 554.37 * t) +   # C#5
        0.2 * np.sin(2 * np.pi * 659.25 * t)     # E5
    )
    audio = audio.astype(np.float32)
    print(f"✓ Generated {len(audio)} samples")

print()

# Load model first to determine device
print("Loading EnCodec 24kHz model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncodecModel.encodec_model_24khz()
model = model.to(device)
model.eval()

# Convert to torch tensor and move to device
audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
audio_tensor = audio_tensor.to(device)
print(f"Audio tensor shape: {audio_tensor.shape} on {device}")
print()

# Load language model for entropy coding tests
print("Loading language model for entropy coding...")
model.lm = model.get_lm_model()
if model.lm is not None:
    model.lm = model.lm.to(device)
print(f"✓ Models loaded on {device}")
print()

# Test configurations
tests = [
    {
        'name': '3.0 kbps + Entropy Coding',
        'bandwidth': 3.0,
        'use_entropy': True,
        'post_compress': None,
        'description': 'Current approach (baseline)'
    },
    {
        'name': '3.0 kbps (No Entropy)',
        'bandwidth': 3.0,
        'use_entropy': False,
        'post_compress': None,
        'description': 'Simpler, works on Android'
    },
    {
        'name': '3.0 kbps + LZMA',
        'bandwidth': 3.0,
        'use_entropy': False,
        'post_compress': 'lzma',
        'description': 'No entropy + LZMA compression'
    },
    {
        'name': '3.0 kbps + zlib',
        'bandwidth': 3.0,
        'use_entropy': False,
        'post_compress': 'zlib',
        'description': 'No entropy + zlib compression'
    },
    {
        'name': '1.5 kbps (No Entropy)',
        'bandwidth': 1.5,
        'use_entropy': False,
        'post_compress': None,
        'description': 'Lower bandwidth, safe margin'
    },
    {
        'name': '1.5 kbps + LZMA',
        'bandwidth': 1.5,
        'use_entropy': False,
        'post_compress': 'lzma',
        'description': 'Lowest bandwidth option'
    },
]

results = []

print("=" * 80)
print("RUNNING TESTS")
print("=" * 80)
print()

for test in tests:
    print(f"Test: {test['name']}")
    print(f"  {test['description']}")
    print("-" * 80)

    try:
        # Set bandwidth
        model.set_target_bandwidth(test['bandwidth'])

        # Encode
        encode_start = time.time()
        with torch.no_grad():
            if test['use_entropy']:
                compressed_bytes = compress(model, audio_tensor, use_lm=True)
            else:
                compressed_bytes = compress(model, audio_tensor, use_lm=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

        encode_time = time.time() - encode_start

        # Post-compression if requested
        final_bytes = compressed_bytes
        post_compress_ratio = 1.0

        if test['post_compress'] == 'lzma':
            lzma_start = time.time()
            final_bytes = lzma.compress(compressed_bytes, preset=1)  # Preset 1 = fast
            lzma_time = time.time() - lzma_start
            post_compress_ratio = len(final_bytes) / len(compressed_bytes)
            print(f"  LZMA: {len(compressed_bytes)} → {len(final_bytes)} bytes ({lzma_time:.3f}s)")

        elif test['post_compress'] == 'zlib':
            zlib_start = time.time()
            final_bytes = zlib.compress(compressed_bytes, level=6)  # Level 6 = balanced
            zlib_time = time.time() - zlib_start
            post_compress_ratio = len(final_bytes) / len(compressed_bytes)
            print(f"  zlib: {len(compressed_bytes)} → {len(final_bytes)} bytes ({zlib_time:.3f}s)")

        # Calculate bitrate
        size_bytes = len(final_bytes)
        bitrate_kbps = (size_bytes * 8) / (DURATION * 1000)

        # Test decode
        decode_start = time.time()
        with torch.no_grad():
            # Decompress post-compression if needed
            decode_bytes = final_bytes
            if test['post_compress'] == 'lzma':
                decode_bytes = lzma.decompress(decode_bytes)
            elif test['post_compress'] == 'zlib':
                decode_bytes = zlib.decompress(decode_bytes)

            decoded, sr = decompress(decode_bytes, device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()

        decode_time = time.time() - decode_start

        # Check if fits in LoRa bandwidth
        fits_lora = bitrate_kbps <= 3.1
        margin = 3.1 - bitrate_kbps

        print(f"  Encode time: {encode_time:.3f}s")
        print(f"  Decode time: {decode_time:.3f}s")
        print(f"  Compressed size: {size_bytes} bytes")
        print(f"  Bitrate: {bitrate_kbps:.2f} kbps")

        if fits_lora:
            print(f"  ✓ FITS in 3.1 kbps LoRa (margin: {margin:.2f} kbps)")
        else:
            print(f"  ✗ TOO HIGH for 3.1 kbps LoRa (over by: {-margin:.2f} kbps)")

        results.append({
            'name': test['name'],
            'bandwidth_setting': test['bandwidth'],
            'bitrate_kbps': bitrate_kbps,
            'size_bytes': size_bytes,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'fits_lora': fits_lora,
            'margin': margin,
            'post_compress_ratio': post_compress_ratio,
            'success': True
        })

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': test['name'],
            'success': False,
            'error': str(e)
        })

    print()

# Summary
print("=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)
print()

# Sort by bitrate (lowest first)
successful = [r for r in results if r.get('success', False)]
successful.sort(key=lambda x: x['bitrate_kbps'])

print(f"LoRa Bandwidth Limit: 3.1 kbps")
print(f"Audio Duration: {DURATION} seconds")
print()

print("Results (sorted by bitrate):")
print("-" * 80)
print(f"{'Configuration':<30} {'Bitrate':<12} {'Size':<12} {'Fits?':<8} {'Margin'}")
print("-" * 80)

for r in successful:
    fits_icon = "✓" if r['fits_lora'] else "✗"
    margin_str = f"+{r['margin']:.2f}" if r['margin'] > 0 else f"{r['margin']:.2f}"
    print(f"{r['name']:<30} {r['bitrate_kbps']:>6.2f} kbps  {r['size_bytes']:>6} bytes  {fits_icon:<8} {margin_str} kbps")

print()

# Find best options
fits_options = [r for r in successful if r['fits_lora']]
best_quality = None
best_margin = None

if fits_options:
    # Best quality = highest bitrate that still fits
    best_quality = max(fits_options, key=lambda x: x['bitrate_kbps'])
    # Best margin = lowest bitrate (most headroom)
    best_margin = min(fits_options, key=lambda x: x['bitrate_kbps'])

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if best_quality == best_margin:
        print("✓ RECOMMENDED: Use this single option that fits")
        print()
        print(f"  {best_quality['name']}")
        print(f"    Bitrate: {best_quality['bitrate_kbps']:.2f} kbps")
        print(f"    Margin: {best_quality['margin']:.2f} kbps")
        print(f"    Frame size: {best_quality['size_bytes']} bytes")
        print()

        if not best_quality['name'].startswith('3.0 kbps + Entropy'):
            print("  Benefits over current approach:")
            current = next((r for r in successful if r['name'] == '3.0 kbps + Entropy Coding'), None)
            if current and current['success']:
                if 'No Entropy' in best_quality['name']:
                    print("    - Simpler decoding (works on Android without PyTorch)")
                    print("    - Faster decode time")
                if 'LZMA' in best_quality['name'] or 'zlib' in best_quality['name']:
                    savings = current['bitrate_kbps'] - best_quality['bitrate_kbps']
                    print(f"    - Lower bandwidth (saves {savings:.2f} kbps vs entropy coding)")

    else:
        print("Options that fit in 3.1 kbps LoRa:")
        print()

        print(f"1. BEST QUALITY: {best_quality['name']}")
        print(f"     Bitrate: {best_quality['bitrate_kbps']:.2f} kbps")
        print(f"     Margin: {best_quality['margin']:.2f} kbps")
        print()

        print(f"2. MOST HEADROOM: {best_margin['name']}")
        print(f"     Bitrate: {best_margin['bitrate_kbps']:.2f} kbps")
        print(f"     Margin: {best_margin['margin']:.2f} kbps")
        print()

    # Android compatibility note
    android_compatible = [r for r in fits_options if 'No Entropy' in r['name'] or 'LZMA' in r['name'] or 'zlib' in r['name']]
    if android_compatible:
        print("Android-Compatible Options (no PyTorch language model needed):")
        for r in android_compatible:
            print(f"  • {r['name']}: {r['bitrate_kbps']:.2f} kbps")
        print()

    # Implementation notes
    print("Implementation Notes:")
    print()
    if 'LZMA' in best_quality['name']:
        print("  To use LZMA compression:")
        print("    Encoder: compressed_bytes = compress(model, audio, use_lm=False)")
        print("             final_bytes = lzma.compress(compressed_bytes, preset=1)")
        print("    Decoder: compressed_bytes = lzma.decompress(final_bytes)")
        print("             decoded, sr = decompress(compressed_bytes, device)")
        print()
    elif 'zlib' in best_quality['name']:
        print("  To use zlib compression:")
        print("    Encoder: compressed_bytes = compress(model, audio, use_lm=False)")
        print("             final_bytes = zlib.compress(compressed_bytes, level=6)")
        print("    Decoder: compressed_bytes = zlib.decompress(final_bytes)")
        print("             decoded, sr = decompress(compressed_bytes, device)")
        print()
    elif 'No Entropy' in best_quality['name']:
        print("  To disable entropy coding:")
        print("    Change: compress(model, audio, use_lm=True)")
        print("    To:     compress(model, audio, use_lm=False)")
        print()

else:
    print("✗ WARNING: No configurations fit within 3.1 kbps LoRa limit!")
    print()
    print("Closest option:")
    closest = min(successful, key=lambda x: abs(x['bitrate_kbps'] - 3.1))
    print(f"  {closest['name']}: {closest['bitrate_kbps']:.2f} kbps (over by {-closest['margin']:.2f} kbps)")
    print()
    print("Recommendations:")
    print("  1. Test if LoRa can handle slight overrun")
    print("  2. Reduce frame size (currently 5s)")
    print("  3. Use lower bandwidth setting")

print()
print("=" * 80)
print("Test complete!")
print("=" * 80)
