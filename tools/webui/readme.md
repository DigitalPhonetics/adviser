# How to install webui

1. install nodejs and npm (for example using nvm: https://github.com/nvm-sh/nvm)
2. go to webui directiory: `cd tools/webui`
3. install dependencies: `npm install`

# How to run webui

1. go to webui directiory: `cd tools/webui`
2. start server (development mode): `python run_webui_server.py&` (if this doesn't work, start the server in a new shell (with the same virtualenv being active) without the ampersand: `python run_webui_server.py`)
3. start client (development mode): `npm run dev`

# Notes:
Flag images taken from https://flaglane.com/