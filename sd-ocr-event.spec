import os
import rapidocr
from pathlib import Path

rapidocr_path = Path(rapidocr.__file__).parent

datas = [
    (rapidocr_path / "default_models.yaml", "rapidocr"),
    (rapidocr_path / "models", "rapidocr/models"),
    (rapidocr_path / "config.yaml", "rapidocr"),
]

binaries = [(Path.cwd().parent.parent / "activitywatch/scripts/dylib/libsqlcipher.0.dylib", '.'), ]

hiddenimports = ["rapidocr", "onnxruntime", "torch", "openvino", "shapely", "shapely.geometry",]

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

block_cipher = None

a = Analysis(["sd_ocr_event/__main__.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

# ---------------------------------------------------------
# PYZ
# ---------------------------------------------------------

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)


# ---------------------------------------------------------
# EXE
# ---------------------------------------------------------

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="sd-ocr-event",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    contents_directory=".",
)

# ---------------------------------------------------------
# COLLECT
# ---------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="sd-ocr-event",
)