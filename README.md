# পাতা ডাক্তার — Leaf Doctor (deploy bundle)

Built by `tools/make_deploy.py` in the thesis project; do not edit here. One address serves the web app (`/`),
the API (`/api/v1/...`, `/health`), the expert desk (`/expert`) and the Android APK (`/leaf-doctor.apk`).

* Model: **Section 28 field model**
* Advice: no pesticide names or doses; spray decisions go to the Sub-Assistant Agriculture Officer.
* Privacy: no account; GPS/EXIF is removed on the phone before upload; cases live in `/tmp` and are
  wiped whenever the server restarts. A thesis demo, not an agricultural service.
