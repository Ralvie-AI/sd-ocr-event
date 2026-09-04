build:
	poetry install 

package: 
	pyinstaller sd-ocr-event.spec --clean --noconfirm

clean:
	rm -rf build dist
	rm -rf sd_ocr_event/__pycache__