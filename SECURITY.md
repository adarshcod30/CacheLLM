# Security policy

## Supported versions

Security fixes go into the latest release on PyPI. Please upgrade before
reporting, in case it is already fixed:

```bash
pip install --upgrade cachellm-proxy
```

## Reporting a vulnerability

Please do not open a public issue for a security problem. Email
**adarshdwivedi256@gmail.com** with:

- what the problem is and what an attacker could do with it,
- the version, from `cachellm version`,
- steps or a request that reproduces it, ideally with the `fake/echo` model so
  no real provider is involved.

Never include real API keys. You will get a reply as soon as possible, and
credit in the release notes if you would like it.

## What is in scope

- The proxy itself: authentication, routing, caching, and the admin endpoints.
- Anything that could leak a provider key, a prompt or an answer to the wrong
  place, or serve one user's answer to another.

## Things operators should know

- The cache stores prompts and answers. With the default settings they live in
  the proxy's memory. With Redis or a snapshot file, protect those like any
  other store of user data.
- Client authentication is off by default and the proxy listens on
  `127.0.0.1`. Set `CACHELLM_API_KEYS` before letting anything else reach it,
  and put it behind TLS.
- The personal-data guard is a safety net, not a guarantee. Send
  `X-Cache-Control: no-store` for anything your app knows is private.
