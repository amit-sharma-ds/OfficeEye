@echo off
rem Collect training frames from the running app during office hours, then
rem pre-label them for review.
rem   tools\collect_day.bat "2026-10-20 09:00" "2026-10-20 19:30"
rem Logs: data\train_collect.log and data\train_prelabel.log
cd /d "%~dp0.."
echo === %date% %time% collecting %~1 to %~2 >> data\train_collect.log
py -3.12 tools\collect.py --office-only --start "%~1" --until "%~2" >> data\train_collect.log 2>&1
echo === %date% %time% pre-labelling >> data\train_prelabel.log
py -3.12 tools\prelabel.py >> data\train_prelabel.log 2>&1
echo === %date% %time% done >> data\train_prelabel.log
