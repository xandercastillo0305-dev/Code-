# Demo samples — read before any demonstration

**Use only non-explicit, redacted or synthetic face images in demos.**
Never use real case material, explicit content, or images of minors in a demo,
a panel presentation, screenshots, or the manuscript (Appendix D).

`make_demo_samples.py` creates three safe samples from the public-domain NASA
portrait bundled with scikit-image (`skimage.data.astronaut()`, portrait of
astronaut Eileen Collins, a US Government work):

| file | purpose |
|---|---|
| `samples/reference.png` | reference image (original portrait) |
| `samples/suspect_authentic.jpg` | same photo, rescaled + JPEG re-compressed (should look "Real") |
| `samples/suspect_manipulated.png` | face region synthetically warped + re-toned (a stand-in for a face manipulation) |
| `samples/no_face.png` | an image without a face (shows the "No face detected" error) |

    python demo/make_demo_samples.py

The manipulated sample is a simple synthetic edit, not a real deepfake; it is
only for showing the workflow. With an untrained model the AI result is not
meaningful, and the UI and reports show the UNTRAINED banner.
