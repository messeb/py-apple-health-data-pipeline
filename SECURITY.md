# Security Policy

## Supported Versions

This is a young project. Security fixes go into the latest release and `main`.

| Version | Supported |
| --- | --- |
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a Vulnerability

**Please do not open a public issue for a security problem.**

- Preferred: open a [private security advisory](https://github.com/messeb/py-apple-health-data-pipeline/security/advisories/new)
  on GitHub (*Security* tab, then *Report a vulnerability*).
- Or e-mail `<security-contact>` (placeholder: maintainers replace this before publishing).

Please include what is affected (the CLI, a parser, the Streamlit app, the Databricks bundle, or
the ML code), the steps to reproduce with **synthetic** data, and the impact you expect. Never send
us your real export.

We aim to acknowledge a report within 5 working days and to agree on a disclosure date with you.
Coordinated disclosure is credited in the changelog unless you prefer otherwise.

Examples of relevant problems:

- Parser issues triggered by a crafted export, such as XML entity expansion, path traversal in ZIP or
  side-file handling, or unbounded memory use.
- SQL injection through app inputs or bundle variables.
- Personal data leaking into logs, table comments, MLflow runs or artifacts where the docs say it is
  dropped.
- Workflow or dependency problems that could expose secrets in CI.

## Data Privacy

The project sends no data anywhere, and git and pre-commit guards keep health data out of the
repository. [docs/privacy.md](docs/privacy.md) lists what is in an export, what the pipeline drops and
how to share safely. Use the synthetic demo data (`make demo`) for screenshots, bug reports and pull
requests.
