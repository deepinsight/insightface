# PersonAnalysis examples

These examples target InsightFace 2.1. Install or upgrade the library with
`python -m pip install -U insightface`. A missing Cheetah model package downloads
on first use. For offline installation, place the
complete package in `~/.insightface/models/cheetah_s/` or
`~/.insightface/models/cheetah_l/`.
See the [PersonAnalysis guide](../../python-package/docs/person_analysis.md)
for installation, model files and result fields.

The example scripts are in this repository and are not installed by pip.
Download or clone the repository, then run the commands below from its root.
Replace the example filenames with your own images and videos:

```bash
# One image. Without --reference, detections are returned as unmatched.
python examples/person_analysis/demo.py --image scene.jpg --reference "Alice=alice.jpg"

# Local video with boxes. Omit --display when running without a desktop.
python examples/person_analysis/demo.py --video clip.mp4 --reference "Alice=alice.jpg" --display

# Local camera and RTSP. Add --reference to match registered people.
python examples/person_analysis/demo.py --camera 0 --display
python examples/person_analysis/demo.py --rtsp "rtsp://camera/stream" --display

# Two images already cropped to individual bodies: print cosine similarity.
python examples/person_analysis/reid.py body_a.jpg body_b.jpg
```

The algorithm in `demo.py` has three steps: `get(frame)`, `match(observations)`
and optional `update(matches)`. The example handles input and drawing. Use
`--no-update` to disable automatic body-reference updates and `--max-frames 20`
to limit processing to 20 frames. Press Q or Esc to stop the preview, or Ctrl+C
in the terminal. Both scripts default to `cheetah_s` on CPU. Add
`--model cheetah_l` to use the larger package. Add `--gpu` only after installing
the GPU runtime and configuring CUDA as described in the guide. `demo.py` also
accepts `--root` to choose a different model directory.

OpenCV reads frames sequentially. This example does not manage stream reconnects,
guarantee frame dropping or maintain history. OpenCV text drawing has limited
Unicode support; console output preserves the original names, and the desktop
GUI can draw Unicode labels.

References are released when the instance closes. There is no database, tracking,
first-seen time or event history. A qualified face match with clear body ownership
can add body references; a body-only match does not expand the references.

`person_config.json` shows the SDK's JSON configuration format. Load it with
`PersonAnalysis.from_config()` in your own script; the command-line examples do
not read it automatically. Model packages are distributed as complete archives,
with `manifest.json` alongside the model files. Keep those files together when
installing a package.
Both packages use the same ReID model; their body detectors default to 320×320 and 640×640 inputs
respectively. Face detection defaults to 640×640 for both packages.
`config.body_det_size=0` follows the selected package's body input size.
`config.face_det_size=0` uses the SDK's 640×640 default, not a manifest value.
When creating the instance, a positive multiple of 64
overrides the square body input, and a positive multiple of 32 overrides the
square face input. The package and feature extraction models stay unchanged.
The chosen face size applies to both reference registration and scene analysis.
See the [body input-size guide](../../python-package/docs/person_analysis.md#body-detection-input-size)
and [face input-size guide](../../python-package/docs/person_analysis.md#face-detection-input-size).
