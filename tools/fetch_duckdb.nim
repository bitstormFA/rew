## Fetch the pinned DuckDB C artifact used by rew DataFrames.

import ../src/rew/dataframe

let path = ensureDuckDbArtifact()
echo "DuckDB ", DuckDbVersionTag, " library: ", path
