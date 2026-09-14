"""Unit tests for the Multimodal Dataset Recorder."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from hub.recorder import SessionRecorder, record_simulated_session


def test_session_recorder_buffers():
    """Verify in-memory buffer aggregation and save output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        recorder = SessionRecorder(label="fall_forward", subject_id="test_sub", notes="test notes")

        npz_file, json_file = recorder.save(out_path)

        assert npz_file.exists()
        assert json_file.exists()

        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["label"] == "fall_forward"
        assert meta["subject_id"] == "test_sub"
        assert meta["notes"] == "test notes"
        assert meta["csi_samples"] == 0
        assert meta["radar_samples"] == 0


def test_simulated_dataset_recording():
    """Verify synthetic multimodal recording produces valid NPZ arrays."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)

        npz_file, json_file = record_simulated_session(
            label="fall_backward",
            subject_id="sub_05",
            duration=2.0,  # 2 seconds @ 100 Hz
            notes="carpet test",
            output_dir=out_path,
        )

        assert npz_file.exists()
        assert json_file.exists()

        # Check metadata
        with open(json_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["label"] == "fall_backward"
        assert meta["subject_id"] == "sub_05"
        assert meta["csi_samples"] == 600  # 3 nodes * 200 samples
        assert meta["radar_samples"] == 20  # 2s @ 10 Hz

        # Check NPZ arrays
        with np.load(npz_file) as data:
            assert "csi_node_ids" in data
            assert "csi_timestamps" in data
            assert "csi_amplitudes" in data
            assert "radar_heights" in data
            assert "radar_postures" in data

            assert len(data["csi_node_ids"]) == 600
            assert data["csi_amplitudes"].shape == (600, 64)
            assert len(data["radar_heights"]) == 20
