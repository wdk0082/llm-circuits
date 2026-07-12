===========
Transcoders
===========

circuit-tracer is used **only** to load transcoders — everything downstream is
implemented in this package.
:mod:`~llm_circuits.transcoders.circuit_tracer_loader` is the single module that
imports from ``circuit_tracer``.

Registry
============================================================

.. automodule:: llm_circuits.transcoders.registry
   :members:

Loading
============================================================

.. automodule:: llm_circuits.transcoders.circuit_tracer_loader
   :members:

Feature labels
============================================================

.. automodule:: llm_circuits.transcoders.feature_labels
   :members:
