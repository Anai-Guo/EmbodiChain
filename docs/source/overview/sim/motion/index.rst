Robot Motion
============

The :mod:`embodichain.lab.sim.motion` package groups four related capabilities:
kinematic solvers, trajectory planners, workspace analysis, and expert trajectory
augmentation. Simulation objects and atomic actions use these capabilities to
translate robot goals into executable motion.

Choose the owning subpackage when importing an API. The ``motion`` parent loads
its children on access and does not re-export their classes and functions;
public imports still use the normal ``lab`` and ``sim`` initialization path.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Subpackage
     - Responsibility
   * - ``motion.solvers``
     - Forward, inverse, and differential kinematics.
   * - ``motion.planners``
     - Joint-space and Cartesian paths, collision-aware planning, and time
       parameterization.
   * - ``motion.workspace``
     - Offline reachability analysis, workspace caches, and runtime sampling.
   * - ``motion.trajectory_augmentation``
     - Explicit qpos templates and candidates, constrained geometric and timing
       variation, measured coverage, and bounded generation accounting.

The augmentation package currently provides the core contracts and operators.
The :doc:`generation host layer </overview/trajectory_generation>` supplies
full-batch initial-state preparation, restoration, qpos execution, and a
synchronous episode sink. Its handwritten qpos runner collects explicitly free
motion using measured collision/task evidence and persistence confirmations.
Contact, held-object, and atomic-source collection remain outside this path. See the
:doc:`augmentation API </api_reference/embodichain/embodichain.lab.sim.motion.trajectory_augmentation>`
for the implemented boundaries.

.. toctree::
   :maxdepth: 1

   solvers/index
   planners/index

Workspace workflows are documented in
:doc:`/features/workspace_analyzer/index`. The complete public package surface is
listed in the :doc:`motion API </api_reference/embodichain/embodichain.lab.sim.motion>`.
