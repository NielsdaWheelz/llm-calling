# Services

## Scope

This document covers backend service ownership and the shared infrastructure
that services compose.

## Product Service

- Product services own user-facing functionality and product-facing admin
  surfaces.
- Service entrypoints live in explicit entrypoint directories.
- API bind and origin environment variables belong to the service that serves
  the API.
- Web origin and development bind environment variables belong to the service
  that serves or owns the web surface.

## Infrastructure Service

- Infrastructure services own infrastructure management such as tenants,
  provisioning, endpoint/domain management, tunnels, relays, and
  infrastructure-facing admin surfaces.
- Feature subsystem setup environment variables belong in the subsystem doc that
  owns the setup flow.
- Infrastructure service entrypoints follow the same entrypoint, API
  environment, and web-origin ownership rules as product services.

## Shared Infrastructure

- Shared reusable modules live in the repository's shared module area.
- The lowest-level reusable substrate owns framework concerns such as
  coordination, database families, retry, transport, serialization, setup,
  process helpers, and web substrate.
- Shared persistent tables have one shared storage owner. Semantic behavior
  still lives in the module that owns the concept.
- Shared semantic modules remain outside the lowest-level framework substrate
  and are composed where a service needs them.
- Transient coordination storage stays with coordination because it is a backend
  detail for the transient coordination runtime, not a shared primary schema
  owner.
