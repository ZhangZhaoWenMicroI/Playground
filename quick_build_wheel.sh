#!/bin/bash
pip3 uninstall playground
python3 -m pip install --upgrade build
python3 -m build --wheel
cd dist/
pip3 install --no-deps playground-*.*.*-py3-none-any.whl
cd ..
