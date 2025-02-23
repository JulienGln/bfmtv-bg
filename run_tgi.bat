@echo off
setlocal

REM Define variables
set "model=Gor-bepis/fact-checker-bfmtg-v2"
set "volume=%cd%\data"

REM Attempt to run with GPU support
echo Attempting to run with GPU support...
docker run --gpus all --shm-size 1g -p 8083:80 -v "%volume%:/data" ^
    ghcr.io/huggingface/text-generation-inference:3.1.0 ^
    --model-id %model%

REM Check if the container is running
if %errorlevel% neq 0 (
    echo Failed to run with GPU. Falling back to CPU...
    REM Run without GPU support
    docker run --shm-size 1g -p 8083:80 -v "%volume%:/data" ^
        ghcr.io/huggingface/text-generation-inference:3.1.0 ^
        --model-id %model%
) else (
    echo Container is running with GPU support.
)

endlocal
