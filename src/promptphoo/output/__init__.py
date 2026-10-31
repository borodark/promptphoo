"""Eval output: result files (`-o`), the console table and the end-of-run summary.

Ports src/util/output.ts, src/util/outputFormats.ts, src/util/junit.ts, the CSV export in
src/util/eval/evalTableUtils.ts, src/presentation/evalResults.ts (convertResultsToTable),
src/table.ts and src/util/eval/summary.ts.
"""

from promptphoo.output.results_table import convert_results_to_table
from promptphoo.output.summary import format_duration, generate_eval_summary
from promptphoo.output.table import generate_table, wrap_table
from promptphoo.output.writer import (
    SUPPORTED_OUTPUT_FILE_FORMATS,
    OutputWriteError,
    create_output_data,
    create_output_metadata,
    get_output_file_format,
    get_table,
    write_multiple_outputs,
    write_output,
)

__all__ = [
    "SUPPORTED_OUTPUT_FILE_FORMATS",
    "OutputWriteError",
    "convert_results_to_table",
    "create_output_data",
    "create_output_metadata",
    "format_duration",
    "generate_eval_summary",
    "generate_table",
    "get_output_file_format",
    "get_table",
    "wrap_table",
    "write_multiple_outputs",
    "write_output",
]
