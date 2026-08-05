# Project Context Boundary

## Scope

This document covers project-context web/server boundaries: context resolution,
project creation, profile fields, summary projection, onboarding state, and
pre-activation plan selection.

## Boundaries

- The project-context module owns project context resolution, access checks, and
  materialized project info.
- The project service owns core project mutation semantics, including ensuring a
  project for a principal, setting pre-activation plan state, and
  completing onboarding state.
- The project-summary owner owns generated project summary projections from
  notes and agent or worker summaries.
- Session and project handlers adapt current session/project context and API
  replay keys into the project service. They do not own core project
  table writes.
- Billing, grants, contact-number verification, navigation, surfaces, and
  project resources remain owned by their feature-local services.
