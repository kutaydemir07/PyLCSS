# PyLCSS Security

## Treat projects and models as executable content

Design Studio `.cad` projects can contain CadQuery Code nodes whose Python code
runs when the graph is previewed or solved. System-modeling workflows can also
compile and execute generated Python. Serialized `joblib`, pickle, dill, and
PyTorch model files can execute code while loading.

Only open, run, or import projects, scripts, plugins, model files, and solver
decks from people and locations you trust. Inspect Code nodes before approving
execution. PyLCSS warns before executing code embedded in a newly opened Design
Studio project and requires explicit trust for the public `TorchRegressor.load`
API.

## AI assistant

Cloud-assistant requests can disclose graph properties and can invoke
application tools. Keep automatic tool execution disabled unless the provider,
model, prompt, and project are trusted. Review changes and engineering results
before relying on them. The assistant no longer has a system-wide mouse or
keyboard control capability.

## Installers and releases

Official public Windows releases should be Authenticode-signed and published
with their SHA-256 checksum. A valid signature identifies the signer; a
checksum detects a changed download only when the checksum itself comes from a
trusted channel. Do not bypass Windows warnings for an installer whose origin
you cannot verify. The Windows launcher only offers stable releases from the
official GitHub Releases endpoint, requires an exact versioned setup/checksum
pair, verifies the checksum, and rejects installers without a trusted
Authenticode signature. When the installed launcher is signed, the update must
also have the same publisher certificate subject.

## Reporting a vulnerability

Do not publish exploitable details in a public issue. Use the repository's
private GitHub Security Advisory reporting channel when available and include
the affected version, reproduction steps, impact, and any proposed mitigation.
