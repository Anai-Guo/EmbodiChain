# Deformable API Simplification Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep one deformable data implementation and a small, consistently named configuration API.

**Architecture:** Surface and volume objects share `DeformableObjectData`. They expose state through `.data` and keep distinct render and volume-boundary topology. Physical configs use `attrs.surface_props`; volume mesh construction uses `meshing`.

**Tech Stack:** Python, configclass, PyTorch, DexSim Newton, pytest.

**Spec:** The user's final instructions retain `youngs` and `poissons`, name the shared group `SurfaceElementPropertiesCfg`, and require removing compatibility code.

## Constraints

- Work on the user's current branch and leave changes uncommitted.
- Preserve numerical defaults, Spawn ownership, binding rollback, and reset behavior.
- Do not add caching or new public state-write APIs.
- Keep only the current API; remove old imports, forwarding methods, config field migration, and pickle migration.

## Implementation

- [x] Merge particle data into one concrete `DeformableObjectData`.
- [x] Remove redundant object data factories and legacy data/object interfaces.
- [x] Keep `VolumeDeformableObjectCfg` and `SurfaceDeformableObjectCfg` as the concrete object configs.
- [x] Use `VolumeDeformablePhysicsCfg`, `SurfaceDeformablePhysicsCfg`, `VolumeDeformableMeshingCfg`, and shared `SurfaceElementPropertiesCfg`.
- [x] Retain short elasticity field names `youngs` and `poissons`.
- [x] Decode current nested dictionaries directly through config post-init hooks and deformable-specific `from_dict()`.
- [x] Remove legacy configuration decorators, aliases, wrapper modules, manager wrappers, descriptor wrappers, and unsupported construction stubs.
- [x] Migrate source callers, examples, tutorials, tests, API documentation, and project context.
- [x] Verify typed round trips, defaults, descriptor values, invalid keyword rejection, independent snapshots, lifecycle integration, and real GPU behavior.

## Validation

- Expanded related tests: 265 passed; 11 failures also reproduce on the unmodified HEAD baseline. These concern rigid collision/restitution defaults and a missing Articulation mock attribute.
- Four real CUDA checks passed: cloth partial pose update, cloth and volume data contracts, and volume boundary geometry.
- API documentation coverage: 1804/1804 exports.
- Repository Black check: 905 files unchanged.
- Sphinx dummy build succeeded with 160 warnings across the documentation tree; none of the warning lines reference deformable APIs.
