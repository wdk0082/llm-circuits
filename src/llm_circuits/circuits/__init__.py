"""Circuit analysis: attribution graphs and interventions (WIP)."""

from llm_circuits.circuits.local_replacement_model import (
    LocalReplacementContext as LocalReplacementContext,
)
from llm_circuits.circuits.local_replacement_model import (
    run_local_replacement as run_local_replacement,
)
from llm_circuits.circuits.replacement_model import (
    ComparisonResult as ComparisonResult,
)
from llm_circuits.circuits.replacement_model import (
    ReplacementContext as ReplacementContext,
)
from llm_circuits.circuits.replacement_model import (
    compare_models as compare_models,
)
from llm_circuits.circuits.replacement_model import (
    replace_mlps_with_transcoders as replace_mlps_with_transcoders,
)
