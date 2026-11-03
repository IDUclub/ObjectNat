# Security Policy

## Supported versions

Security fixes go into the latest release of ObjectNat. Please upgrade before reporting:

```bash
pip install -U objectnat
```

## Reporting a vulnerability

Do not open a public issue. Report it privately through
[GitHub private vulnerability reporting](https://github.com/IDUclub/ObjectNat/security/advisories/new)
or by email to ddonnyy@itmo.ru.

Include the affected version, a description, and steps or code to reproduce. We aim to reply within
7 days and to publish a fix and an advisory once the issue is confirmed.

ObjectNat processes geospatial data locally and does not run network services. Relevant reports include,
for example, crashes or memory errors in the compiled Rust extension (`objectnat._rust`) triggered by
crafted input geometries, and problems in the published wheels or the release pipeline.
