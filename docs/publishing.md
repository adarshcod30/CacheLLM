# Publishing a release

Releases go to PyPI through [Trusted Publishing][tp], so there is no API token
in the repository, in GitHub secrets, or on anyone's laptop. GitHub proves the
workflow's identity to PyPI directly over OIDC, and a leaked token cannot exist
because there is no token.

[tp]: https://docs.pypi.org/trusted-publishers/

## One-time setup

You need a PyPI account, then one form.

1. Create an account at [pypi.org/account/register](https://pypi.org/account/register/)
   and turn on two-factor authentication, which PyPI requires for publishing.
2. Go to [Publishing settings](https://pypi.org/manage/account/publishing/) and
   add a **pending publisher** with exactly these values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `cachellm` |
   | Owner | `adarshcod30` |
   | Repository name | `CacheLLM` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   "Pending" is the right choice: the project does not exist on PyPI yet, and
   the first successful publish creates it.

3. In the GitHub repository, under Settings, Environments, create an
   environment named `pypi`. Adding yourself as a required reviewer there is
   worth the extra click: it means no tag can reach PyPI without you approving
   that specific run.

## Cutting a release

```bash
# bump the version in pyproject.toml and src/cachellm/__init__.py first
git tag v0.1.0
git push origin v0.1.0
```

The workflow then runs lint, type checks and the full test suite against a real
Redis, refuses to continue if the tag does not match `cachellm.__version__`,
builds both distributions, validates them with `twine check`, publishes to
PyPI, and opens a GitHub release with the artefacts attached.

## Publishing by hand instead

If you would rather not use the workflow, the same artefacts publish with an
API token from your account:

```bash
uv build
uvx twine check dist/*
uvx twine upload dist/*     # prompts for __token__ and your pypi-... token
```

Test it against [TestPyPI](https://test.pypi.org) first if you want a dry run:

```bash
uvx twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ cachellm
```
