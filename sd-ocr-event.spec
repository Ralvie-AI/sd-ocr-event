import os

import rapidocr

from PyInstaller.utils.hooks import collect_all

# ---------------------------------------------------------
# Collect RapidOCR
# ---------------------------------------------------------

rapidocr_datas, rapidocr_binaries, rapidocr_hiddenimports = collect_all("rapidocr")

# ---------------------------------------------------------
# Collect OpenVINO
# ---------------------------------------------------------

openvino_datas, openvino_binaries, openvino_hiddenimports = collect_all("openvino")

# ---------------------------------------------------------
# Merge package data
# ---------------------------------------------------------

datas = []
datas.extend(rapidocr_datas)
datas.extend(openvino_datas)

binaries = []
binaries.extend(rapidocr_binaries)
binaries.extend(openvino_binaries)

hiddenimports = []
hiddenimports.extend(rapidocr_hiddenimports)
hiddenimports.extend(openvino_hiddenimports)

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