## dataframe umbrella — lazy relational DataFrame values and expression DSL.
##
## This is the public module for structured analytics. It exposes
## backend-neutral DataFrame values, column expressions, chainable relational
## verbs, and explicit DuckDB-backed materialization.

import ./dataframe/[core, materialized, materialize, artifacts]
export core, materialized, materialize, artifacts
