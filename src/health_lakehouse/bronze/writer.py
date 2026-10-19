"""Buffered, schema-stable Parquet writer used by all bronze parsers."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class BufferedParquetWriter:
    """Accumulate dict rows and flush them to a single Parquet file in batches.

    Every column is written as a string (bronze = raw, untyped) unless an
    explicit schema is given, so the file schema never drifts between batches.
    """

    def __init__(self, path: Path, schema: pa.Schema, batch_size: int = 200_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.batch_size = batch_size
        self._columns = {name: [] for name in schema.names}
        self._n = 0
        self.rows_written = 0
        self._writer = pq.ParquetWriter(self.path, schema, compression="zstd")

    def write(self, row: dict) -> None:
        for name, col in self._columns.items():
            col.append(row.get(name))
        self._n += 1
        if self._n >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._n:
            return
        table = pa.Table.from_pydict(self._columns, schema=self.schema)
        self._writer.write_table(table)
        self.rows_written += self._n
        self._columns = {name: [] for name in self.schema.names}
        self._n = 0

    def close(self) -> int:
        self.flush()
        self._writer.close()
        return self.rows_written


def string_schema(*names: str, extra: dict[str, pa.DataType] | None = None) -> pa.Schema:
    fields = [pa.field(n, pa.string()) for n in names]
    for n, t in (extra or {}).items():
        fields.append(pa.field(n, t))
    return pa.schema(fields)
