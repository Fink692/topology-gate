# Security policy

Topology Gate is research software and is not designed to receive untrusted
production traffic. Do not place API keys, broker credentials, private vendor
data, or personal information in issues, pull requests, logs, checkpoints, or
reports.

If you find a security issue, do not disclose a working exploit publicly. Once
the repository is published, use a private GitHub Security Advisory from the
repository's **Security** tab. Include the affected commit, a minimal
reproduction, and the impact. If private advisories are unavailable, open a
minimal issue asking for a private contact channel without including secrets or
exploit details.

Reports containing local paths or sensitive metadata should be treated as
disclosable artifacts: redact them before sharing. The project makes no claim
that its checkpoint, audit, or numerical boundaries are suitable for an
untrusted service without additional process-level controls.
