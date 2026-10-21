@echo off
rem Train the phone-use model after office hours, at low priority so the
rem camera app keeps running smoothly. It goes into use by itself only if
rem it passes its test (tools\train_phone.py).
rem   tools\train_phone_tonight.bat "2026-10-21 20:15"
rem Log: data\phone\train.log   Result: data\phone\report.json
cd /d "%~dp0.."
start "" /belownormal /b /wait py -3.12 tools\train_phone.py --start "%~1" >> data\phone\train_out.log 2>&1
