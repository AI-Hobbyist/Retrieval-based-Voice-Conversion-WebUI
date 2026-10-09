@echo off
cd /d "%~dp0"
runtime\python.exe -m rvc_api %*
