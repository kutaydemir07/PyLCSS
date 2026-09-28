# PyLCSS Privacy Information

Last updated: 2026-09-28

PyLCSS is a desktop application. The project does not operate a cloud service
or add product analytics or advertising telemetry. Network traffic occurs
when installing packages or optional solvers and checking for updates.

## Update checks

Installed Windows copies request the latest stable release metadata from the
official PyLCSS repository on GitHub at most once per day during startup. The
request identifies the launcher as `PyLCSS-Windows-Updater`; it does not include
project contents, a persistent identifier, or usage telemetry.
GitHub receives normal connection metadata such as the public IP address. If
you accept an update, the launcher downloads the release installer and checksum
from GitHub. Developer checkouts do not check automatically. Set
`PYLCSS_DISABLE_UPDATE_CHECK=1` or start `PyLCSS.exe --skip-update-check` to
disable the request.

## Local storage

PyLCSS writes project files, exported models, simulation results, application
logs, and solver logs to the working directory or locations selected by you.
Remove files you no longer need using your operating system's file manager.

Python package indexes, GitHub, and optional solver publishers have their own
privacy policies for network requests you choose to make.
