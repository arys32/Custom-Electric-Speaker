import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.io import wavfile
from scipy.signal import find_peaks
import os
import shutil
import subprocess
import tempfile

def get_magnitude_spectrum_data(segment, samplerate):
    # Normalize to [-1, 1] for proper dBFS measurements
    segment_normalized = segment.astype(float) / 32768.0
    fig = plt.figure()
    spectrum, freqs, line = plt.magnitude_spectrum(segment_normalized, Fs=samplerate, scale='dB')
    plt.close(fig)
    # Explicitly convert to numpy arrays to ensure we have pure data, not matplotlib objects
    return np.asarray(spectrum, dtype=float), np.asarray(freqs, dtype=float)

def plot_spectrum(data, start_sec, samplerate, duration=0.1):
    end_sec = start_sec + duration
    start_idx = int(start_sec * samplerate)
    end_idx = int(end_sec * samplerate)
    segment = data[start_idx:end_idx]
    # Normalize to [-1, 1] for proper dBFS measurements
    segment_normalized = segment.astype(float) / 32768.0

    plt.figure()
    plt.xscale('log', base=2)
    plt.grid(True, which="both", ls="-") 

    spectrum, freqs = get_magnitude_spectrum_data(segment, samplerate)
    plt.magnitude_spectrum(segment_normalized, Fs=samplerate, scale='dB', label=rf"$t \in $[{np.round(start_sec,2)}, {np.round(end_sec,2)}]")
    return spectrum, freqs

def get_freq_from_sweep(data, samplerate, time, duration=0.1):
    spectrum, freqs = plot_spectrum(data, time, samplerate, duration=duration)
    peak_indices, _ = find_peaks(spectrum)
    if peak_indices.size == 0:
        plt.close()
        raise ValueError("No spectral peak found for the selected time window.")

    dominant_peak_idx = peak_indices[np.argmax(spectrum[peak_indices])]
    peak_freq = freqs[dominant_peak_idx]
    plt.close()
    return peak_freq

def get_amplitude(data, samplerate, time, duration=0.1):
    if duration <= 0:
        raise ValueError("duration must be > 0")

    start_idx = int(time * samplerate)
    end_idx = int((time + duration) * samplerate)
    segment = data[start_idx:end_idx]

    if segment.size == 0:
        raise ValueError("Selected time window is empty.")

    return float(np.max(np.abs(segment)))

def get_spectral_magnitude_at_freq(data, samplerate, time, target_freq, duration=0.1):
    if duration <= 0:
        raise ValueError("duration must be > 0")
    
    start_idx = int(time * samplerate)
    end_idx = int((time + duration) * samplerate)
    segment = data[start_idx:end_idx]
    
    if segment.size == 0:
        raise ValueError("Selected time window is empty.")
    
    # Get spectrum using same method as plot_spectrum
    spectrum, freqs = get_magnitude_spectrum_data(segment, samplerate)
    # Convert to dB using the correct conversion
    spectrum_dB = 20.0 * np.log10(spectrum)

    # Use the maximum magnitude in a small band just below the target frequency.
    band_lo = target_freq * 0.92
    band_mask = (freqs >= band_lo) & (freqs <= target_freq)
    if np.any(band_mask):
        magnitude_db = np.max(spectrum_dB[band_mask])
    else:
        magnitude_db = np.interp(target_freq, freqs, spectrum_dB)
    return float(magnitude_db)

def plot_sweep_freq_vs_time(data, samplerate, duration=0.1, n_points=200):
    total_duration = len(data) / samplerate
    max_time = max(0.0, total_duration - duration)
    times = np.linspace(0.0, max_time, n_points)
    freqs = np.array([get_freq_from_sweep(data, samplerate, t, duration=duration) for t in times])

    plt.figure()
    plt.plot(times, freqs)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-")
    plt.title("Sweep Frequency vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

def get_freq_response_curve_data(sweep_raw, samplerate, duration, sweep_response, start_time=0.0, analysis_duration=None):
    if duration <= 0:
        raise ValueError("duration must be > 0")

    total_duration = min(len(sweep_raw), len(sweep_response)) / samplerate
    max_time = total_duration - duration
    if max_time < 0:
        raise ValueError("Audio is shorter than one analysis window.")

    analysis_start = float(np.clip(start_time, 0.0, max_time))
    if analysis_duration is not None and analysis_duration > 0:
        analysis_end = min(max_time, analysis_start + float(analysis_duration))
    else:
        analysis_end = max_time

    if analysis_end < analysis_start:
        raise ValueError("Invalid analysis time range.")

    times = np.arange(analysis_start, analysis_end + 1e-12, duration)
    freqs = []
    amplitudes = []
    spectral_magnitudes = []

    total_bins = len(times)
    report_step = max(1, total_bins // 20)
    print(f"[curve] Computing frequency/amplitude bins: 0/{total_bins}", flush=True)

    for idx, t in enumerate(times, start=1):
        freq = get_freq_from_sweep(sweep_raw, samplerate, t, duration=duration)
        amp = get_amplitude(sweep_response, samplerate, t, duration=duration)
        spec_mag = get_spectral_magnitude_at_freq(sweep_response, samplerate, t, freq, duration=duration)
        freqs.append(freq)
        amplitudes.append(amp)
        spectral_magnitudes.append(spec_mag)

        if idx % report_step == 0 or idx == total_bins:
            pct = int(round(100.0 * idx / total_bins))
            print(f"[curve] Progress: {idx}/{total_bins} ({pct}%)", flush=True)

    if len(freqs) == 0:
        raise ValueError("No valid analysis bins were found.")

    freqs = np.array(freqs, dtype=float)
    amplitudes = np.array(amplitudes, dtype=float)
    spectral_magnitudes = np.array(spectral_magnitudes, dtype=float)
    
    # Convert to dBFS (normalize to [-1, 1] range, then to dB)
    amplitudes_normalized = amplitudes / 32768.0
    amplitudes_dbfs = 20.0 * np.log10(np.maximum(amplitudes_normalized, 1e-12))
    
    return times, freqs, amplitudes_dbfs, spectral_magnitudes, total_duration

def plot_freq_vs_response_amplitude(sweep_raw, samplerate, duration, sweep_response, label=""):
    _, freqs, amplitudes_db, spectral_magnitudes, _ = get_freq_response_curve_data(sweep_raw, samplerate, duration, sweep_response)

    plt.figure()
    plt.plot(freqs, amplitudes_db, '-', linewidth=1.0, alpha=0.8, label='Peak amplitude')
    plt.scatter(freqs, amplitudes_db, s=10, alpha=0.7)
    plt.plot(freqs, spectral_magnitudes, '-', color='purple', linewidth=2.0, alpha=0.8, label='Spectral magnitude at sweep frequency')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-")
    plt.title(f"Response Amplitude vs Sweep Frequency {label}")
    plt.xlabel("Frequency from Raw Sweep (Hz)")
    plt.ylabel("Response Level (dBFS)")
    plt.legend(loc='lower left')

def plot_static_freq_response(
    sweep_raw,
    samplerate,
    duration,
    sweep_response,
    output_path="static_freq_response.png",
    font_size=11,
):
    _, freqs, amplitudes_db, spectral_magnitudes, _ = get_freq_response_curve_data(
        sweep_raw, samplerate, duration, sweep_response
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(freqs, amplitudes_db, '-', linewidth=1.0, alpha=0.8, label='Peak amplitude')
    ax.scatter(freqs, amplitudes_db, s=10, alpha=0.7)
    ax.plot(freqs, spectral_magnitudes, '-', color='purple', linewidth=2.0, alpha=0.8, label='Spectral magnitude at sweep frequency')
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-")
    # ax.set_title("Response Amplitude vs Sweep Frequency", fontsize=font_size)
    ax.set_xlabel("Frequency (Hz)", fontsize=font_size)
    ax.set_ylabel("Response Level (dBFS)", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)
    ax.legend(loc='lower left', fontsize=font_size)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax

def plot_multi_peak_amplitude(
    sweep_raw,
    samplerate,
    duration,
    responses,
    output_path="multi_peak_amplitude.png",
    font_size=11,
):
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-")
    ax.set_xlabel("Frequency (Hz)", fontsize=font_size)
    ax.set_ylabel("Response Level (dBFS)", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    for wav_path, label in responses:
        print(f"[multi] Processing {wav_path}...", flush=True)
        sr, sweep_response = wavfile.read(wav_path)
        if sr != samplerate:
            raise ValueError(f"Sample rate mismatch: {wav_path} is {sr} Hz, expected {samplerate} Hz")
        _, freqs, amplitudes_dbfs, _, _ = get_freq_response_curve_data(
            sweep_raw, samplerate, duration, sweep_response
        )
        ax.plot(freqs, amplitudes_dbfs, '-', linewidth=1.5, alpha=0.8, label=label)

    ax.legend(loc='lower left', fontsize=font_size)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    return fig, ax


def make_freq_response_animation_mp4(
    sweep_raw,
    samplerate,
    duration,
    sweep_response,
    output_path="response_vs_frequency.mp4",
    fps=30,
    video_length=None,
    start_time=0.0,
):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to create MP4 output.")

    total_duration = min(len(sweep_raw), len(sweep_response)) / samplerate
    max_time = total_duration - duration
    if max_time < 0:
        raise ValueError("Audio is shorter than one analysis window.")

    start_time = float(np.clip(start_time, 0.0, max_time))
    available_duration = max(0.0, max_time - start_time)
    if video_length is not None and video_length > 0:
        anim_duration = min(float(video_length), available_duration)
        print(
            f"[anim] Limiting animation to {anim_duration:.2f} seconds from t={start_time:.2f}s "
            f"(requested length: {video_length:.2f}s)",
            flush=True,
        )
    else:
        anim_duration = available_duration
        print(f"[anim] Rendering from t={start_time:.2f}s for {anim_duration:.2f} seconds", flush=True)

    print("[anim] Preparing curve data...", flush=True)
    times, freqs, amplitudes_db, spectral_magnitudes, _ = get_freq_response_curve_data(
        sweep_raw,
        samplerate,
        duration,
        sweep_response,
        start_time=start_time,
        analysis_duration=anim_duration,
    )
    print("[anim] Curve data ready.", flush=True)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax_left.plot(freqs, amplitudes_db, '-', linewidth=1.0, alpha=0.8, label='Peak amplitude')
    ax_left.scatter(freqs, amplitudes_db, s=10, alpha=0.7)
    ax_left.plot(freqs, spectral_magnitudes, '-', color='purple', linewidth=2.0, alpha=0.8, label='Spectral magnitude at sweep frequency')
    ax_left.set_xscale('log')
    ax_left.grid(True, which="both", ls="-")
    ax_left.set_title("Response Amplitude vs Sweep Frequency")
    ax_left.set_xlabel("Frequency from Raw Sweep (Hz)")
    ax_left.set_ylabel("Response Level (dBFS)")

    y_min = min(float(np.min(amplitudes_db)), float(np.min(spectral_magnitudes))) - 3.0
    y_max = 3.0  # Slightly above 0 dBFS for headroom
    ax_left.set_ylim(y_min, y_max)

    tracker = ax_left.axvline(freqs[0], color='red', linewidth=2.0, linestyle='--', alpha=0.7, label='Current frequency')
    time_text = ax_left.text(0.02, 0.98, "t = 0.00 s", transform=ax_left.transAxes, va='top')
    ax_left.legend(loc='lower left')

    ax_right.set_xscale('log')
    ax_right.grid(True, which="both", ls="-")
    ax_right.set_title("Current Response Spectrum")
    ax_right.set_xlabel("Frequency (Hz)")
    ax_right.set_ylabel("Magnitude (dBFS)")
    ax_right.set_ylim(-160, 0)
    spec_time_text = ax_right.text(0.02, 0.02, f"Window = {duration:.3f} s", transform=ax_right.transAxes, va='bottom')

    n_frames = max(1, int(np.ceil(anim_duration * fps)))
    render_report_step = max(1, n_frames // 20)
    print(f"[anim] Rendering frames: 0/{n_frames}", flush=True)

    last_reported_frame = 0
    def update(frame_idx):
        nonlocal last_reported_frame
        t = min(frame_idx / fps, anim_duration)
        absolute_t = start_time + t
        t_for_interp = np.clip(absolute_t, times[0], times[-1])
        current_freq = np.interp(t_for_interp, times, freqs)
        tracker.set_xdata([current_freq, current_freq])
        time_text.set_text(f"t = {absolute_t:.2f} s")
        
        # Get current spectral magnitude from pre-computed data
        current_spectral_magnitude = np.interp(current_freq, freqs, spectral_magnitudes)

        segment_len = max(2, int(duration * samplerate))
        start_idx = int(t_for_interp * samplerate)
        end_idx = start_idx + segment_len
        if end_idx > len(sweep_response):
            end_idx = len(sweep_response)
            start_idx = max(0, end_idx - segment_len)

        segment = sweep_response[start_idx:end_idx]
        # Normalize to [-1, 1] for proper dBFS measurements
        segment_normalized = segment.astype(float) / 32768.0
        if segment_normalized.size < segment_len:
            padded = np.zeros(segment_len, dtype=float)
            padded[:segment_normalized.size] = segment_normalized
            segment_normalized = padded

        # Clear and redraw spectrum on right axis
        ax_right.clear()
        ax_right.magnitude_spectrum(segment_normalized, Fs=samplerate, scale='dB', color='tab:orange', label='Response spectrum')
        
        # Add purple cross at the current frequency using the pre-computed spectral magnitude
        # (This matches the static purple line on the left)
        ax_right.plot(current_freq, current_spectral_magnitude, marker='x', color='purple', markersize=12, markeredgewidth=3, label='Magnitude at sweep frequency')
        
        ax_right.axvline(current_freq, color='red', linewidth=2.0, linestyle='--', alpha=0.7, label='Current frequency')
        ax_right.set_xscale('log')
        ax_right.set_xlim(20.0, samplerate / 2.0)
        ax_right.set_ylim(-160, 0)
        ax_right.grid(True, which="both", ls="-")
        ax_right.set_title("Current Response Spectrum")
        ax_right.set_xlabel("Frequency (Hz)")
        ax_right.set_ylabel("Magnitude (dBFS)")
        ax_right.legend(loc='upper right')
        spec_time_text = ax_right.text(0.02, 0.02, f"Window = {duration:.3f} s, t = {absolute_t:.2f} s", transform=ax_right.transAxes, va='bottom')

        current_frame = frame_idx + 1
        if current_frame - last_reported_frame >= render_report_step or current_frame == n_frames:
            pct = int(round(100.0 * current_frame / n_frames))
            print(f"[anim] Render progress: {current_frame}/{n_frames} ({pct}%)", flush=True)
            last_reported_frame = current_frame

        return tracker, time_text

    animation = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)

    with tempfile.TemporaryDirectory(prefix="sweep_anim_") as tmpdir:
        silent_video_path = os.path.join(tmpdir, "video_no_audio.mp4")
        temp_audio_path = os.path.join(tmpdir, "response_audio.wav")

        print("[anim] Writing silent video...", flush=True)
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=5000, 
                             extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.0', '-preset', 'medium'])
        animation.save(silent_video_path, writer=writer)
        print("[anim] Silent video written.", flush=True)

        print("[anim] Writing temporary WAV audio...", flush=True)
        start_sample = int(start_time * samplerate)
        audio_samples = sweep_response[start_sample:]
        if video_length is not None and video_length > 0:
            max_samples = int(anim_duration * samplerate)
            audio_samples = audio_samples[:max_samples]
        wavfile.write(temp_audio_path, samplerate, audio_samples)
        print("[anim] Temporary WAV written.", flush=True)

        mux_cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            silent_video_path,
            "-i",
            temp_audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-shortest",
            output_path,
        ]
        print("[anim] Muxing video + audio with ffmpeg...", flush=True)
        subprocess.run(mux_cmd, check=True, capture_output=True, text=True)
        print(f"[anim] Mux complete: {output_path}", flush=True)

    plt.close(fig)
    return output_path

def plot_spectrogram(data, samplerate):

    plt.figure()
    plt.specgram(data, Fs=samplerate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.yscale('log')
    plt.ylim(20, samplerate / 2)

    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz, log scale)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Intensity (dB)")

samplerate, sweep_raw = wavfile.read("recordings/rawsweep.wav")

# plot_sweep_freq_vs_time(sweep_raw, samplerate, duration=0.5)

samplerate, sweep_response1 = wavfile.read("recordings/sweep1lowgain.50-20k.wav")

# spectrum, freqs = plot_spectrum(sweep_response1, 0, samplerate, duration=0.5)
# spectrum_dB = 20*np.log10(spectrum)
# plt.figure()
# plt.xscale("log")
# # plt.yscale("log")
# plt.plot(freqs, spectrum_dB)
# plt.show()

# plt.show()

# plot_freq_vs_response_amplitude(sweep_raw, samplerate, 0.5, sweep_response1)
# plt.show()

response_files = [
    ("recordings/sweep1lowgain.50-20k.wav", "Center"),
    ("recordings/1point1.50-20k.wav",       "Point 1"),
    ("recordings/1point2.50-20k.wav",       "Point 2"),
    ("recordings/1point3.50-20k.wav",       "Point 3"),
    ("recordings/1point4.50-20k.wav",       "Point 4"),
]
plot_multi_peak_amplitude(sweep_raw, samplerate, 0.5, response_files)

plt.show()
# plot_static_freq_response(
#     sweep_raw,
#     samplerate,
#     0.5,
#     sweep_response1,
#     output_path="static_freq_response.png",
# )

# plt.show()


# print(make_freq_response_animation_mp4(sweep_raw, samplerate, 0.5, sweep_response1, start_time=0.0, video_length=60.0))
