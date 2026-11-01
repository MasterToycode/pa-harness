"""PAH — Physics-Aware Harness for assemblable 3D parts.

The package is split the way `docs/pah-dev-doc-v1.md` §7.6 describes:

    core/       pure domain: no I/O, no third-party geometry imports
    app/        SQLite store, durable effect queue, composition root
    executors/  serial / pool / dsh bridge
    policy/     triage -> candidates -> rank -> adjudicate
    tools/      verifier + operator + generator adapters
    cli/        the `pah` command

Import direction is one-way: cli -> app -> core, tools -> core. `core` imports
nothing outside the standard library.
"""

__version__ = "0.1.0"
