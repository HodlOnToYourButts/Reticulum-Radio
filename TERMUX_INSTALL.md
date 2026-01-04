# Termux Listener Installation Guide

## Prerequisites
- Android device (Android 7+)
- Termux from **F-Droid** (not Google Play - outdated version)
- WiFi connection for initial setup (large downloads)

## Step 1: Install Termux
Download from: https://f-droid.org/en/packages/com.termux/

## Step 2: Grant Storage Permission
```bash
termux-setup-storage
```
Tap "Allow" when prompted.

## Step 3: Update Packages
```bash
pkg update && pkg upgrade -y
```

## Step 4: Install System Dependencies
```bash
# Core development tools
pkg install python python-pip git clang cmake ninja -y

# Audio libraries (for sounddevice)
pkg install portaudio pulseaudio -y

# Math libraries (for NumPy/PyTorch)
pkg install openblas -y
```

## Step 5: Install Python Dependencies

### Option A: Try PyTorch from pip (Recommended - Try First)
```bash
# Upgrade pip
pip install --upgrade pip

# Try installing PyTorch (may work with newer pip versions)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# If that fails, try:
pip install torch torchaudio
```

### Option B: Use Community ARM64 Wheels
If pip fails, try KumaTea's ARM64 wheels:
```bash
pip install torch -f https://torch.kmtea.eu/whl/stable.html
pip install torchaudio -f https://torch.kmtea.eu/whl/stable.html
```

### Install Other Dependencies
```bash
# Audio codec
pip install encodec

# Audio playback (alternative to PyAudio)
pip install sounddevice

# Reticulum networking
pip install rns

# LXST audio framework
pip install lxst

# Other requirements
pip install numpy
```

## Step 6: Clone Repository
```bash
cd ~
git clone <your-repo-url> Reticulum-Radio
cd Reticulum-Radio
```

## Step 7: Test EnCodec
Before running the full listener, test if EnCodec works:
```bash
python test_encodec_simple.py
```

This will test:
- PyTorch import
- EnCodec model loading
- Decode performance on ARM64 CPU

## Step 8: Run Listener
```bash
# Get broadcaster control destination hash from desktop
python src/listener.py --station <control_dest_hash> --quality low
```

## Troubleshooting

### PyTorch Installation Issues

**Error: "Could not find a version that satisfies the requirement torch"**
- ARM64 wheels may not be available for your Python version
- Try: `python --version` (should be 3.8-3.11)
- Downgrade if needed: `pkg install python=3.10 -y`

**Error: "No module named 'torch'"**
- Installation failed silently
- Check: `pip list | grep torch`
- Try Option B (KumaTea wheels)

**Last Resort: Build from Source (Very Slow)**
- Only if both options above fail
- Can take 1+ week on phone
- Not recommended

### Audio Playback Issues

**Error: "No module named 'sounddevice'"**
```bash
pip install sounddevice
```

**Error: "PortAudio not found"**
```bash
pkg install portaudio -y
```

**Error: "ALSA/PulseAudio errors"**
```bash
# Start PulseAudio
pulseaudio --start

# Or try with environment variable
AUDIODRIVER=pulseaudio python src/listener.py --station <hash>
```

### Performance Issues

**Decoding too slow (choppy audio)**
- EnCodec on ARM64 CPU is slower than desktop
- Try reducing buffer: modify listener.py `buffer_seconds=120`
- Alternative: Use a more powerful device

**High battery drain**
- Normal for CPU-intensive neural codec
- Consider using power saving mode between tests
- For production: would need GPU acceleration (difficult on Android)

### Memory Issues

**Error: "Killed" or process dies**
- Phone ran out of RAM
- Close other apps
- Reboot Termux
- EnCodec models need ~200MB RAM minimum

## Testing Checklist

- [ ] PyTorch imports without errors
- [ ] EnCodec model loads successfully
- [ ] Test decode runs without crashing
- [ ] Listener connects to broadcaster
- [ ] Audio plays through phone speaker
- [ ] No excessive battery drain during playback

## Performance Expectations

**On mid-range ARM64 device:**
- Model load time: 10-30 seconds
- Decode time per 5s frame: 1-3 seconds (slower than desktop)
- Initial buffering: 30-60 seconds
- Battery usage: ~5-15% per hour (CPU-intensive)

**On low-end device:**
- May be too slow for real-time playback
- Consider using Opus codec instead (lower quality, higher bandwidth)

## Next Steps

Once working in Termux:
1. Test LoRa range with portable setup
2. Monitor battery life during extended use
3. Consider building native Android app for better efficiency
4. Optimize buffer sizes for mobile network conditions

## Getting Help

If stuck, provide these details:
- Android version: `getprop ro.build.version.release`
- Device model: `getprop ro.product.model`
- Python version: `python --version`
- PyTorch status: `python -c "import torch; print(torch.__version__)"`
- Error messages (full traceback)
