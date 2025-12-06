# Reticulum Radio

A decentralized radio broadcasting system built on the Reticulum network using LXST (Lightweight Extensible Signal Transport).

## Overview

Reticulum Radio allows users to:
- Create radio broadcasts over the Reticulum network
- Broadcast voice from a microphone in real-time
- Stream MP3 music files
- Tune into broadcasts from other users

## Features

- **Live Voice Broadcasting**: Stream your voice using microphone input
- **Music Streaming**: Play and broadcast MP3 files on the fly
- **Decentralized**: Built on Reticulum's cryptography-based networking stack
- **Low Latency**: Uses LXST for real-time audio with <10ms end-to-end latency
- **End-to-End Encryption**: All streams are encrypted via Reticulum

## Requirements

- Python 3.7+
- Reticulum Network Stack
- LXST (Lightweight Extensible Signal Transport)
- PyAudio for microphone input
- Additional dependencies listed in requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Starting a Broadcast
```bash
python broadcaster.py
```

### Tuning into a Broadcast
```bash
python listener.py
```

## Project Status

This project is in early development/prototype stage.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the LICENSE file for details.

## Technology Stack

- **Reticulum**: Cryptography-based networking stack
- **LXST**: Real-time signal transport framework with broadcast radio primitives
- **Codec Support**: OPUS for high-quality audio streaming

## Sources

- [LXST GitHub Repository](https://github.com/markqvist/lxst)
- [Reticulum GitHub Repository](https://github.com/markqvist/Reticulum)
- [Reticulum Documentation](https://markqvist.github.io/Reticulum/manual/gettingstartedfast.html)
