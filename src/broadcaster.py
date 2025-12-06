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


class MP3Queue:
    """Manages the MP3 playback queue."""

    def __init__(self):
        self.queue = []
        self.current_index = -1
        self.lock = Lock()

    def add(self, mp3_path):
        """Add an MP3 to the queue."""
        with self.lock:
            if os.path.exists(mp3_path):
                self.queue.append(mp3_path)
                print(f"Added to queue: {os.path.basename(mp3_path)}")
                return True
            else:
                print(f"Error: File not found: {mp3_path}")
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
        self.destination = None
        self.lxst_outlet = None

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

    def setup_lxst(self):
        """Initialize LXST for audio streaming."""
        print("Setting up LXST audio outlet...")

        # Create LXST outlet for broadcasting
        # Using OPUS codec for good quality audio streaming
        self.lxst_outlet = LXST.Outlet(
            self.reticulum,
            self.identity,
            name=self.station_name
        )

        print(f"LXST outlet created for: {self.station_name}")
        print(f"Outlet address: {RNS.prettyhexrep(self.lxst_outlet.destination.hash)}")

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

                        # Move to next track when done
                        if self.mp3_queue.has_next():
                            self.mp3_queue.next()
                        else:
                            # Queue finished
                            self.music_playing = False
                            print("\nQueue finished. Switching to microphone...")
                    else:
                        # No current track, switch to mic
                        self.music_playing = False

                # Stream microphone when no music or music paused
                if not self.music_playing or self.music_paused:
                    if self.mic_enabled:
                        self.stream_microphone()

            except Exception as e:
                print(f"Error in audio mixer: {e}")

    def stream_microphone(self):
        """Stream microphone audio through LXST."""
        # TODO: Implement microphone capture using PyAudio
        # TODO: Encode audio stream and send via LXST
        pass

    def stream_mp3(self, mp3_path):
        """
        Stream an MP3 file through LXST.

        Args:
            mp3_path: Path to the MP3 file
        """
        # TODO: Implement MP3 decoding with pydub
        # TODO: Encode to OPUS and stream via LXST
        # TODO: Handle play/pause state during playback
        pass

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

                if command == "add" and len(cmd) > 1:
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

        # Run command interface on main thread
        self.command_interface()

    def shutdown(self):
        """Clean shutdown of broadcaster."""
        self.running.clear()

        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)

        if self.lxst_outlet:
            print("Shutting down LXST outlet...")
            # TODO: Proper LXST cleanup

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
