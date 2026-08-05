# Tunnels

## Scope

This document covers tunnel subsystem rules: the shared authorize route, HTTP
endpoint tunnels, private TCP tunnels, public TCP tunnels, relay tunnels, and
operator setup entrypoints.

## Shared Module

- Keep a shared tunnel module that mounts one authorize route plus feature-local
  route surfaces.
- Feature-local tenant/account services own tunnel read and mutation semantics
  for each tunnel type. Transport handlers adapt current tenant/account context,
  execute operation handles with replay namespaces, and do not own tunnel joins
  or lifecycle transitions.
- The shared authorize route validates tunnel-daemon lifecycle callbacks such
  as login, proxy creation, heartbeat, and work-connection requests.
- HTTP endpoint authorize requests parse the endpoint tunnel token, load the
  published endpoint resource, require the claimed domain to match the stored
  domain, and require the client to be connected to the assigned tunnel server.
- Private TCP tunnel authorize requests parse the sealed tunnel handle, require
  the referenced tunnel to exist on the connected tunnel server, branch by
  tunnel role, forbid consumer-side proxy creation, and require provider proxy
  creation to present the expected proxy type, proxy name, and shared secret.
- Public TCP tunnel authorize requests parse the sealed tunnel handle, require
  the published tunnel resource to exist on the connected tunnel server, require
  the provider token to resolve to that resource, and require provider proxy
  creation to bind the assigned public port.
- Relay authorize requests parse the sealed relay-resource handle and branch by
  tunnel role. Provider authentication should be allowed against the underlying
  resource before tenant/account publication; consumer authentication should
  require the published tenant/account row. Consumer-side proxy creation is
  forbidden.
- The authorize route is tenant/account agnostic because tunnel daemons may not
  carry user session context. Treat outward handles, provider tokens, and shared
  secrets as the authorization boundary.
- Put shared script-serving, setup-script compilation, daemon-version, and
  install-path helpers in the shared tunnel module.

## HTTP Endpoint

- HTTP endpoint tunnels give managed workloads public HTTP access through their
  assigned tunnel server for managed direct subdomains under a configured base
  domain.
- Validate the base domain at startup before accepting tenant/account
  registrations.
- Centralize canonical lowercase endpoint-domain parsing and direct-subdomain
  checks.
- Centralize managed-domain policy rules, including reserved labels and
  reserved brand substrings.
- Store endpoint tunnel credentials as an outward raw token for setup/get/list
  responses and as a verifier hash for shared authorization.
- Tenant/account endpoint APIs check domain availability against endpoint
  resources, derive returned info from published state, reject registrations
  outside the managed-domain policy, and expose the base domain for UI flows
  when needed.
- Registration and deletion run as durable operations that pair database
  visibility changes with a full edge-proxy projection deploy to all tunnel
  servers. Registration first writes the durable resource plus a reservation
  state so the domain is included in uniqueness checks and deploy projection;
  after deploy succeeds, publish the endpoint and remove the reservation.
  Deletion removes publication before redeploying, then deletes the resource so
  a failed undeploy leaves an allocation hold instead of allowing reuse.
- Client setup and teardown scripts are served from feature-local routes.
- Compile generic setup scripts once during route construction and serve them
  with a shell-script content type.
- Setup scripts accept tunnel server, endpoint domain, tunnel token, and local
  bind target values; install the tunnel client into an owned shared binary
  path; write config, wrapper, and manifest files under an owned install
  directory; and register one managed service through the shared service-manager
  layer.
- Managed services choose the host's native service manager, record the chosen
  backend in a manifest, and point the service manager at the owned wrapper
  script in the install directory.
- Teardown scripts derive the same install id, remove the managed service
  through the shared service-manager layer, and delete the owned install
  directory.
- Maintenance deletes stale leaked DNS challenge records under the managed base
  domain after a grace period.

## Private TCP Tunnel

- Private TCP tunnels give managed workloads private TCP connectivity through
  their assigned tunnel server.
- Persist tenant/account-owned private TCP tunnel rows.
- Derive outward tunnel handles from durable row identity. Store generated
  provider-token and shared-secret credentials for setup/get/list responses, and
  store verifier hashes for shared authorization.
- Tenant/account APIs derive and return only the outward tunnel info.
- Provide four script endpoints: provider setup, provider teardown, consumer
  setup, and consumer teardown.
- Setup routes compile generic scripts once during route construction and serve
  them with a shell-script content type.
- Provider setup scripts accept tunnel server, tunnel handle, provider token,
  shared secret, and local target values; install the tunnel client; write
  config and wrapper files under the owned install directory; and register one
  managed service.
- Consumer setup scripts accept tunnel server, tunnel handle, shared secret, and
  bind target values; install the tunnel client; write config and wrapper files
  under the owned install directory; and register one managed service.
- Provider and consumer install ids are short stable hashes derived from the
  tunnel handle, with the consumer id additionally incorporating the bind port.
- Provider and consumer teardown scripts derive the same install ids, remove
  the managed service through the shared service-manager layer, and delete the
  owned install directory.
- Private TCP tunnels reuse the shared authorize route and have no separate
  tunnel-server provisioning entrypoint.

## Public TCP Tunnel

- Public TCP tunnels give managed workloads a public raw TCP endpoint on the
  assigned tunnel server that forwards to a TCP service reachable from the
  provider machine.
- Persist a tenant/account-owned public TCP tunnel resource with assigned
  tunnel server, public port, and provider token. Use narrow reservation,
  publication, and release states that point to the resource for edge-proxy
  reservation, tenant/account-visible publication, and post-delete port drain.
  Publishing inserts the visible row and deletes the reservation row in the
  same transaction.
- Derive the tenant/account-facing public tunnel handle from the published row.
  Store generated provider-token credentials on the resource for setup/get/list
  responses and store a verifier hash for shared authorization.
- Public TCP tunnel APIs create, delete, get, and list public TCP tunnels.
  Creating a tunnel reserves the port, deploys the edge-proxy projection with a
  reservation marker, then publishes the tenant/account-visible marker pointing
  to the same resource. Deleting a tunnel removes publication before
  redeploying so fresh operations stop observing the tunnel immediately, then
  inserts a release state that holds the resource and port until the old
  provider proxy has drained. Future allocation lazily reclaims expired release
  rows before selecting an available port.
- Provide provider setup and teardown script endpoints.
- Provider setup scripts accept tunnel server, public tunnel handle, provider
  token, public port, and local target values; install the tunnel client; write
  config and wrapper files under the owned install directory; and register one
  managed service whose config fixes the remote port to the assigned public
  port.
- Provider teardown scripts derive the same install id, remove the managed
  service through the shared service-manager layer, and delete the owned install
  directory.
- Public TCP tunnels reuse the shared authorize route and have no consumer setup
  command; outside users connect directly to the returned host/port endpoint.

## Relay

- Relays give tenant/account workloads proxy egress through operator-configured
  relay nodes.
- Persist operator-owned relay nodes with structured specs that tenant/account
  workflows can discover and select. Persist tenant/account-owned
  provider-facing relay resources with requested spec, assigned tunnel server,
  relay node, provider token, shared secret, and verifier hashes. Keep
  tenant/account-visible publication as a narrow state pointing to the resource.
- Derive provider/consumer setup handles from the relay resource identity and
  tenant/account-facing handles from the published relay identity. Store the
  generated provider token for provider setup and generated shared secret for
  setup/get/list responses.
- Relay APIs list available relay-node specs and create, delete, get, and list
  relays. Creating a relay chooses one tunnel server and one relay node matching
  the requested spec, creates the underlying relay resource, installs a managed
  provider on the relay node, then publishes the tenant/account-visible relay
  row. Get/list only read published relays. Deleting a relay removes publication
  first, tears down the relay-node provider, then deletes the underlying
  resource.
- Provide four script endpoints: provider setup, provider teardown, consumer
  setup, and consumer teardown.
- Setup routes compile generic scripts once during route construction and serve
  them with a shell-script content type.
- Provider setup scripts accept tunnel server, relay resource handle, provider
  token, and shared secret; install the tunnel client; write config and wrapper
  files under the owned install directory; and register one managed service.
- Consumer setup scripts accept tunnel server, relay resource handle, shared
  secret, and bind target values; install the tunnel client; write config and
  wrapper files under the owned install directory; register one managed service;
  and print the local proxy URL.
- Provider and consumer install ids are short stable hashes derived from the
  relay resource handle, with the consumer id additionally incorporating the
  bind port.
- Provider and consumer teardown scripts derive the same install ids, remove
  the managed service through the shared service-manager layer, and delete the
  owned install directory.
- Relays reuse the shared authorize route and require separate relay-node
  provisioning before tenant/account relay creation can succeed.

## Tunnel Server

- The server-side tunnel module owns shared tunnel-server setup and projection
  deployment.
- Setup renders one bootstrap script that installs the tunnel server binary,
  installs or builds the edge proxy with required DNS and layer-4 support,
  writes managed service files, validates the edge-proxy config, and restarts
  the tunnel server and edge proxy.
- Generated edge-proxy config should separate external TLS routing from private
  peer termination. External routes send known endpoint and tunnel-server
  hostnames to the owning server's private address with client-address
  forwarding. The private peer terminator binds only to the private address and
  requires forwarded client metadata before TLS so application handling sees the
  original client IP.
- Generated edge-proxy config also renders public TCP listeners for the local
  tunnel server's assigned public TCP tunnel ports. Each listener binds on the
  tunnel server public hostname and proxies raw TCP to the matching local tunnel
  server proxy port.
- Tunnel server config uses the shared authorize path plus the local tunnel
  server public hostname as the authorization callback context for lifecycle
  requests.
- Keep proxy listeners on loopback where possible and restrict client-requested
  public TCP proxy ports to the configured public TCP tunnel port range.
- The setup flow reads stable deployment-wide config from environment or
  config, while the specific tunnel server being provisioned is supplied through
  explicit command arguments.
- Expose setup through a durable operation catalog shared by setup and
  edge-proxy deploys.

## Operations

- Provide an operator maintenance command for HTTP endpoint stale DNS challenge
  cleanup.
- Provide an operator entrypoint for provisioning tunnel servers.
- Tunnel-server setup generates a fresh replay key, executes one durable setup
  attempt under a replay-aware tunnel-config lock, prepares a replay-stable
  tunnel-server id when the row does not exist yet, defects if an existing row
  has a different private address or remote setup command fails, runs setup
  through the target executor/worker, then inserts or refreshes the
  tunnel-server row after setup succeeds.
- Run setup from a control-plane host with the target tunnel-server public
  hostname, private address, and executor/worker credential.
- The background worker must already be running for tunnel-server setup.
- Each tunnel-server setup invocation starts a fresh durable setup attempt.
- If the tunnel-server setup process dies after the durable operation is
  persisted, the background worker can finish or dead-letter that attempt.
  Rerunning the command later starts a new attempt for the same public-hostname
  row.
- Provide an operator entrypoint for registering relay nodes.
- Relay-node setup waits for executor/worker readiness, generates a fresh
  replay key, then inserts a relay-node row for a new worker credential or
  refreshes the spec for the same worker credential.
- Run relay-node setup from a control-plane host with the relay-node spec and
  executor/worker credential.
- Development wrappers around setup should reuse the same setup effect with
  development config sources.
- Development infrastructure commands should wait for a running dev stack and
  rerun tunnel-server and relay-node setup without resetting the database.
- Development reset-and-serve commands should reset the database, seed app
  state, start the dev stack, wait for readiness, then rerun infrastructure
  provisioning so reset databases recreate durable relay catalog entries.
