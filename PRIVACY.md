# PyLCSS Privacy Information

Last updated: 2026-08-01

PyLCSS is a desktop application. The project does not operate a PyLCSS cloud
service and does not add product analytics or advertising telemetry. Network
traffic is created when you configure a third-party AI provider, install
packages or optional solvers, check provider model lists, use features that
explicitly contact those services, or when an installed Windows copy performs
its daily update check.

## Update checks

Installed Windows copies request the latest stable release metadata from the
official PyLCSS repository on GitHub at most once per day during startup. The
request identifies the launcher as `PyLCSS-Windows-Updater`; it does not include
project contents, assistant data, a persistent identifier, or usage telemetry.
GitHub receives normal connection metadata such as the public IP address. If
you accept an update, the launcher downloads the release installer and checksum
from GitHub. Developer checkouts do not check automatically. Set
`PYLCSS_DISABLE_UPDATE_CHECK=1` or start `PyLCSS.exe --skip-update-check` to
disable the request.

## AI assistant data

The `Local` provider sends requests only to the local endpoint you configure.
When OpenAI, Anthropic, or Google is selected, PyLCSS sends data directly to
that provider. A request can contain:

- your prompt and the provider/model selection;
- recent prompts and responses, but only when conversation memory is enabled;
- a snapshot of the active Modeling or Design Studio graph, including node
  names, types, properties, and connections; and
- assistant tool definitions, tool calls, and tool results needed to complete
  the request.

The selected provider's terms, retention rules, and privacy policy govern data
received by that provider. Do not submit personal, confidential, export-
controlled, or employer/client data unless you are authorized to disclose it.
PyLCSS asks for confirmation before the first cloud request. Loading a cloud
provider's model list is also an intentional network request and occurs only
after you press `Load Models` or `Test Connection`.

## Local storage

On Windows, assistant files are normally stored below
`%LOCALAPPDATA%\PyLCSS\assistant\`:

- `settings.json` contains assistant preferences and encrypted API-key values;
- `.llm_key` contains a random salt used for machine/user-bound Fernet
  encryption; and
- `llm_memory.json` contains prompts and responses in plaintext when
  conversation memory is enabled.

Conversation memory is off by default for new installations. The Memory tab
can enable it or delete all retained conversations. API-key encryption reduces
accidental disclosure at rest but is not an operating-system credential vault;
any process running as the same user may be able to access the files and the
machine identity used to derive the encryption key.

PyLCSS may also write `pylcss.log`, solver logs, project files, exported models,
and simulation results to locations selected by the user or the current
working/project directory. Assistant request text is not written to the PyLCSS
application log by the request manager.

## Your controls

You can choose `Local`, leave cloud API keys empty, keep memory disabled, clear
stored memory, delete the per-user assistant directory, and remove project or
solver output files. Deleting an API key in Assistant Settings removes its
stored encrypted value after settings are saved.

This document describes PyLCSS behavior; it is not a substitute for the privacy
information of Python package indexes, AI providers, or optional solver
publishers that you choose to contact.
