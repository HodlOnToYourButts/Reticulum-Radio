#!/usr/bin/env python3
"""
EnCodec GPU Feasibility Test for Android/Termux

Tests whether PyTorch GPU acceleration (Vulkan/NNAPI) can efficiently
run EnCodec decoding on Android devices. This determines if EnCodec is
viable for mobile, or if we should fall back to Opus codec.

Usage:
    python test_android_gpu.py

Requirements:
    pip install torch encodec numpy
"""

import sys
import time
import os
import platform
import traceback
from pathlib import Path

print("=" * 70)
print("EnCodec GPU Feasibility Test for Android/Termux")
print("=" * 70)
print()

# Detect if running in Termux
IS_TERMUX = os.environ.get('TERMUX_VERSION') is not None
print(f"Environment: {'Termux' if IS_TERMUX else 'Desktop/Server'}")
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python: {sys.version}")
print()

# Try importing dependencies
print("Checking dependencies...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not found. Install with: pip install torch")
    sys.exit(1)

try:
    from encodec import EncodecModel, compress, decompress
    print(f"✓ EnCodec library")
except ImportError:
    print("✗ EnCodec not found. Install with: pip install encodec")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError:
    print("✗ NumPy not found. Install with: pip install numpy")
    sys.exit(1)

print()

# Detect available backends
print("Detecting PyTorch backends...")
print("-" * 70)

backends = {}

# CPU (always available)
backends['cpu'] = {
    'available': True,
    'name': 'CPU',
    'device': torch.device('cpu')
}
print(f"✓ CPU: Always available")

# CUDA (NVIDIA GPUs - not on Android)
backends['cuda'] = {
    'available': torch.cuda.is_available(),
    'name': 'CUDA (NVIDIA GPU)',
    'device': torch.device('cuda') if torch.cuda.is_available() else None
}
if backends['cuda']['available']:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ CUDA: {gpu_name}")
else:
    print(f"✗ CUDA: Not available (expected on Android)")

# Vulkan (cross-platform GPU - potential Android support)
try:
    vulkan_available = hasattr(torch.backends, 'vulkan') and torch.backends.vulkan.is_available()
    backends['vulkan'] = {
        'available': vulkan_available,
        'name': 'Vulkan GPU',
        'device': 'vulkan' if vulkan_available else None
    }
    if vulkan_available:
        print(f"✓ Vulkan: Available (Android GPU acceleration)")
    else:
        print(f"✗ Vulkan: Not available")
except AttributeError:
    backends['vulkan'] = {'available': False, 'name': 'Vulkan GPU', 'device': None}
    print(f"✗ Vulkan: Not supported in this PyTorch build")

# MPS (Apple Silicon - not on Android)
try:
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    backends['mps'] = {
        'available': mps_available,
        'name': 'Apple MPS',
        'device': torch.device('mps') if mps_available else None
    }
    if mps_available:
        print(f"✓ MPS: Available (Apple Silicon)")
    else:
        print(f"✗ MPS: Not available")
except AttributeError:
    backends['mps'] = {'available': False, 'name': 'Apple MPS', 'device': None}

print()

# Select backends to test
test_backends = {k: v for k, v in backends.items() if v['available']}

if len(test_backends) == 0:
    print("ERROR: No backends available!")
    sys.exit(1)

print(f"Will test {len(test_backends)} backend(s): {', '.join(test_backends.keys())}")
print()

# Load EnCodec model
print("Loading EnCodec 24kHz model...")
print("-" * 70)

try:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(3.0)  # 3 kbps target
    print(f"✓ Model loaded (bandwidth: 3.0 kbps)")

    # Load language model for entropy coding
    print("Loading language model for entropy coding...")
    model.lm = model.get_lm_model()
    print(f"✓ Language model loaded")

except Exception as e:
    print(f"✗ Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# Create test audio (5 seconds at 24kHz mono)
print("Creating test audio (5 seconds, 24kHz mono)...")
sample_rate = 24000
duration = 5  # seconds
num_samples = sample_rate * duration

# Generate sine wave test signal
freq = 440.0  # A4 note
t = np.linspace(0, duration, num_samples)
test_audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

# Convert to torch tensor (channels, samples)
test_tensor = torch.from_numpy(test_audio).unsqueeze(0)  # Add channel dimension
print(f"✓ Test audio created: {test_tensor.shape} (1 channel, {num_samples} samples)")
print()

# Encode once on CPU (to get compressed data for decode tests)
print("Encoding test audio with entropy coding (CPU)...")
print("-" * 70)

try:
    model_cpu = model.to('cpu')
    if hasattr(model_cpu, 'lm') and model_cpu.lm is not None:
        model_cpu.lm = model_cpu.lm.to('cpu')

    model_cpu.eval()

    with torch.no_grad():
        start_time = time.time()
        compressed_bytes = compress(model_cpu, test_tensor, use_lm=True)
        encode_time = time.time() - start_time

    compressed_size = len(compressed_bytes)
    bitrate_kbps = (compressed_size * 8) / (duration * 1000)

    print(f"✓ Encoded in {encode_time:.2f}s")
    print(f"  Compressed size: {compressed_size} bytes")
    print(f"  Effective bitrate: {bitrate_kbps:.2f} kbps")
    print()

except Exception as e:
    print(f"✗ Encoding failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test decoding on each available backend
print("Testing decode performance on each backend...")
print("=" * 70)

results = {}

for backend_name, backend_info in test_backends.items():
    print(f"\nTesting: {backend_info['name']} ({backend_name})")
    print("-" * 70)

    try:
        device = backend_info['device']

        # Note: Vulkan may not support all PyTorch operations
        # For now, we'll test CPU and CUDA (if available)
        if backend_name == 'vulkan':
            print("⚠ Vulkan backend detected but may not support all EnCodec operations")
            print("  EnCodec uses complex Transformer models that may not be Vulkan-compatible")
            print("  Skipping for now - would need model conversion to TorchScript/NNAPI")
            results[backend_name] = {
                'success': False,
                'error': 'Model conversion required for Vulkan',
                'recommendation': 'Needs TorchScript optimization'
            }
            continue

        # Move model to device
        model_test = model.to(device)
        if hasattr(model_test, 'lm') and model_test.lm is not None:
            model_test.lm = model_test.lm.to(device)

        model_test.eval()

        # Measure decode time (3 iterations for average)
        decode_times = []
        memory_usage = []

        for i in range(3):
            # Clear cache if GPU
            if backend_name == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                start_time = time.time()
                decoded, sr = decompress(compressed_bytes, device=device)

                # Synchronize if GPU
                if backend_name == 'cuda':
                    torch.cuda.synchronize()

                decode_time = time.time() - start_time

            decode_times.append(decode_time)

            # Memory usage
            if backend_name == 'cuda':
                mem_bytes = torch.cuda.max_memory_allocated()
                memory_usage.append(mem_bytes / (1024**2))  # Convert to MB

            print(f"  Iteration {i+1}: {decode_time:.3f}s", end='')
            if memory_usage:
                print(f" (GPU mem: {memory_usage[-1]:.1f} MB)")
            else:
                print()

        avg_decode_time = np.mean(decode_times)
        min_decode_time = np.min(decode_times)
        max_decode_time = np.max(decode_times)

        # Calculate realtime factor (decode time / audio duration)
        realtime_factor = avg_decode_time / duration

        print()
        print(f"Results:")
        print(f"  Average decode time: {avg_decode_time:.3f}s")
        print(f"  Min/Max: {min_decode_time:.3f}s / {max_decode_time:.3f}s")
        print(f"  Realtime factor: {realtime_factor:.2f}x")
        if memory_usage:
            print(f"  Peak GPU memory: {np.mean(memory_usage):.1f} MB")

        # Determine if acceptable
        is_acceptable = realtime_factor < 0.1  # Decode should be <10% of audio duration
        is_good = realtime_factor < 0.05  # Great performance: <5% of audio duration

        if is_good:
            verdict = "✓ EXCELLENT - Very fast, suitable for mobile"
        elif is_acceptable:
            verdict = "✓ ACCEPTABLE - Fast enough for real-time"
        else:
            verdict = "✗ TOO SLOW - Not suitable for real-time streaming"

        print(f"  Verdict: {verdict}")

        results[backend_name] = {
            'success': True,
            'avg_time': avg_decode_time,
            'min_time': min_decode_time,
            'max_time': max_decode_time,
            'realtime_factor': realtime_factor,
            'memory_mb': np.mean(memory_usage) if memory_usage else None,
            'acceptable': is_acceptable,
            'excellent': is_good
        }

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        results[backend_name] = {
            'success': False,
            'error': str(e)
        }

print()
print("=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)
print()

# Check battery status if in Termux
if IS_TERMUX:
    try:
        import subprocess
        battery_result = subprocess.run(['termux-battery-status'], capture_output=True, text=True)
        if battery_result.returncode == 0:
            print("Battery Status:")
            print(battery_result.stdout)
    except:
        print("(Install termux-api for battery monitoring)")
    print()

# Analyze results
successful_backends = {k: v for k, v in results.items() if v.get('success', False)}
acceptable_backends = {k: v for k, v in successful_backends.items() if v.get('acceptable', False)}
excellent_backends = {k: v for k, v in successful_backends.items() if v.get('excellent', False)}

print(f"Tested backends: {len(results)}")
print(f"Successful: {len(successful_backends)}")
print(f"Acceptable performance: {len(acceptable_backends)}")
print(f"Excellent performance: {len(excellent_backends)}")
print()

if len(excellent_backends) > 0:
    print("✓ RECOMMENDATION: EnCodec is VIABLE for Android")
    print()
    print("Best backend(s):")
    for name, result in sorted(excellent_backends.items(), key=lambda x: x[1]['realtime_factor']):
        rt_factor = result['realtime_factor']
        decode_ms = result['avg_time'] * 1000
        print(f"  - {backends[name]['name']}: {decode_ms:.1f}ms avg ({rt_factor:.1%} realtime)")
    print()
    print("Next steps:")
    print("  1. Proceed with Phase 2: Full Termux listener implementation")
    print("  2. Use the fastest backend for production")
    print("  3. Test on actual device with battery monitoring")

elif len(acceptable_backends) > 0:
    print("⚠ RECOMMENDATION: EnCodec MAY WORK but performance is marginal")
    print()
    print("Acceptable backend(s):")
    for name, result in acceptable_backends.items():
        rt_factor = result['realtime_factor']
        decode_ms = result['avg_time'] * 1000
        print(f"  - {backends[name]['name']}: {decode_ms:.1f}ms avg ({rt_factor:.1%} realtime)")
    print()
    print("Concerns:")
    print("  - Performance may degrade on lower-end devices")
    print("  - Battery impact could be significant")
    print()
    print("Recommendations:")
    print("  1. Test on target Android device before committing")
    print("  2. Consider Opus codec as fallback (24-48 kbps, hardware accelerated)")
    print("  3. Implement power-saving mode that switches codecs")

else:
    print("✗ RECOMMENDATION: EnCodec NOT VIABLE for Android")
    print()
    print("Reasons:")
    if len(successful_backends) == 0:
        print("  - All backends failed to decode")
    else:
        print("  - Decode performance too slow for real-time streaming")
        for name, result in successful_backends.items():
            rt_factor = result['realtime_factor']
            print(f"    {backends[name]['name']}: {rt_factor:.1%} realtime (need <10%)")
    print()
    print("Alternative: Use Opus codec instead")
    print("  - Already supported in listener.py")
    print("  - Bandwidth: 24-48 kbps (vs 2 kbps for EnCodec)")
    print("  - Hardware acceleration often available on Android")
    print("  - Much better battery life")
    print("  - Command: python src/listener.py --station <hash> --quality high")

print()
print("=" * 70)
print("Test complete!")
print("=" * 70)
