#!/usr/bin/env python3
"""
Build the NCBI Gene + UniProt index for Phaseolus vulgaris (taxid 3885),
used by gene_validator.py for verifying gene/locus mentions in generated
answers (Change 5).

Downloads:
  1. NCBI gene_info.gz (all organisms, ~1.5GB compressed as of writing —
     verified via a live HEAD request, not the older ~5GB estimate) and
     streams through it filtering for tax_id == 3885. Never loads the whole
     file into memory.
  2. UniProt's P. vulgaris subset via the REST /stream endpoint
     (organism_id:3885 — verified at ~33K entries as of writing, small
     enough to fetch in one call, unlike the ~440K UniProt total).

Produces a pickle (default data/gene_index.pkl) containing:
  gene_id_map: {gene_id: {symbol, synonyms, locus_tag, description,
                           type_of_gene, chromosome, map_location, source}}
  symbol_map:  {lowercased symbol/synonym/locus_tag: [gene_id, ...]}

IMPORTANT: NCBI's P. vulgaris gene_info records use PHAVU_-prefixed locus
tags (e.g. "PHAVU_010G111100"), NOT the dotted Phytozome format
("Phvul.010G111100") — verified directly against live NCBI esummary
records. This script indexes NCBI's native PHAVU_ format as-is;
gene_validator.py converts a Phvul.-formatted mention to the matching
PHAVU_ key at lookup time (see _normalize_phytozome_id there), so both
forms resolve without needing to store both here.

Classical bean genetics symbols (Co-1, Phg-1, bc-1/2/3, fin, Ppd, ...) are
NOT produced by this script at all — verified directly that neither NCBI
nor UniProt catalogs them for P. vulgaris. Those are handled separately by
the hand-curated CLASSICAL_GENE_SYMBOLS allowlist in gene_validator.py.

HPCC network note: compute nodes may not have outbound HTTPS egress. If
this script can't reach ftp.ncbi.nlm.nih.gov or rest.uniprot.org, download
gene_info.gz manually on a login node first:
    wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
then re-run with --gene-info-path pointing at the downloaded file (this
skips the NCBI download but still needs network for the UniProt fetch,
unless you also pass --skip-uniprot).

Usage:
    python3 build_gene_index.py --output data/gene_index.pkl \\
        --log-file logs/gene_index.log

    # If gene_info.gz was already downloaded on a login node:
    python3 build_gene_index.py --gene-info-path /path/to/gene_info.gz \\
        --output data/gene_index.pkl --log-file logs/gene_index.log

    # NCBI only, skip UniProt:
    python3 build_gene_index.py --skip-uniprot ...
"""

import gzip
import json
import pickle
import logging
import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen, Request

TAXID = "3885"
NCBI_GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
UNIPROT_STREAM_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=organism_id:3885&format=json&fields=accession,gene_names,protein_name"
)


def setup_logging(log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _download(url: str, dest: Path, logger: logging.Logger, chunk_size: int = 1 << 20) -> None:
    logger.info(f"Downloading {url} -> {dest}")
    req = Request(url, headers={"User-Agent": "bean-lab-gene-index/1.0"})
    with urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        last_logged_pct = -1
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = int(downloaded * 100 / total)
                if pct != last_logged_pct and pct % 5 == 0:
                    logger.info(f"  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct}%)")
                    last_logged_pct = pct
    logger.info(f"✓ Downloaded {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def parse_ncbi_gene_info(gene_info_path: Path, logger: logging.Logger) -> Dict[str, Dict]:
    """
    Stream gene_info.gz, keeping only tax_id == 3885 rows. Column positions
    are read from the file's own header line (starts with '#tax_id'), not
    hardcoded, so this doesn't silently break if NCBI reorders columns.
    """
    genes: Dict[str, Dict] = {}
    col_index: Dict[str, int] = {}

    logger.info(f"Parsing {gene_info_path} (streaming, filtering tax_id={TAXID})...")
    opener = gzip.open if str(gene_info_path).endswith(".gz") else open
    with opener(gene_info_path, "rt", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                header = line.lstrip("#").split("\t")
                col_index = {name: i for i, name in enumerate(header)}
                logger.info(f"Header columns: {header}")
                continue

            fields = line.split("\t")

            def get(col: str) -> str:
                idx = col_index.get(col)
                if idx is None or idx >= len(fields):
                    return "-"
                return fields[idx]

            if get("tax_id") != TAXID:
                continue

            gene_id = get("GeneID")
            if gene_id == "-" or not gene_id:
                continue

            symbol = get("Symbol")
            synonyms_raw = get("Synonyms")
            synonyms = [] if synonyms_raw == "-" else synonyms_raw.split("|")
            locus_tag = get("LocusTag")

            genes[gene_id] = {
                "gene_id": gene_id,
                "symbol": symbol if symbol != "-" else None,
                "synonyms": synonyms,
                "locus_tag": locus_tag if locus_tag != "-" else None,
                "description": get("description") if get("description") != "-" else None,
                "type_of_gene": get("type_of_gene") if get("type_of_gene") != "-" else None,
                "chromosome": get("chromosome") if get("chromosome") != "-" else None,
                "map_location": get("map_location") if get("map_location") != "-" else None,
                "source": "ncbi",
            }

            if line_num % 5_000_000 == 0:
                logger.info(f"  ...scanned {line_num:,} lines, {len(genes)} P. vulgaris genes so far")

    logger.info(f"✓ Found {len(genes)} P. vulgaris (taxid {TAXID}) gene records in NCBI Gene")
    return genes


def fetch_uniprot(logger: logging.Logger) -> Dict[str, Dict]:
    """
    Fetch the P. vulgaris subset of UniProt via the /stream endpoint (no
    pagination needed — ~33K entries as of writing, verified small enough
    for a single request).
    """
    logger.info(f"Fetching UniProt P. vulgaris entries from {UNIPROT_STREAM_URL}")
    req = Request(UNIPROT_STREAM_URL, headers={"User-Agent": "bean-lab-gene-index/1.0"})
    with urlopen(req, timeout=300) as resp:
        data = json.load(resp)

    entries: Dict[str, Dict] = {}
    for item in data.get("results", []):
        accession = item.get("primaryAccession")
        if not accession:
            continue
        genes_field = item.get("genes") or []
        symbol = None
        synonyms: List[str] = []
        if genes_field:
            gene_name = genes_field[0].get("geneName", {}) or {}
            symbol = gene_name.get("value")
            synonyms = [s.get("value") for s in genes_field[0].get("synonyms", []) if s.get("value")]

        protein_desc = item.get("proteinDescription", {}) or {}
        recommended = protein_desc.get("recommendedName", {}) or {}
        full_name = (recommended.get("fullName") or {}).get("value")

        entries[accession] = {
            "gene_id": accession,
            "symbol": symbol,
            "synonyms": synonyms,
            "locus_tag": None,
            "description": full_name,
            "type_of_gene": None,
            "chromosome": None,
            "map_location": None,
            "source": "uniprot",
        }

    logger.info(f"✓ Fetched {len(entries)} UniProt entries for P. vulgaris")
    return entries


def build_symbol_map(gene_id_map: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Lowercased symbol/synonym/locus_tag -> list of gene_ids that use it."""
    symbol_map: Dict[str, List[str]] = {}

    def _index(key: Optional[str], gene_id: str) -> None:
        if not key:
            return
        k = key.lower()
        symbol_map.setdefault(k, [])
        if gene_id not in symbol_map[k]:
            symbol_map[k].append(gene_id)

    for gene_id, meta in gene_id_map.items():
        _index(meta.get("symbol"), gene_id)
        _index(meta.get("locus_tag"), gene_id)
        for syn in meta.get("synonyms") or []:
            _index(syn, gene_id)

    return symbol_map


def main():
    parser = argparse.ArgumentParser(description="Build NCBI+UniProt gene index for P. vulgaris")
    parser.add_argument("--output", default="data/gene_index.pkl")
    parser.add_argument("--log-file", default="logs/gene_index.log")
    parser.add_argument("--gene-info-path",
                        help="Path to an already-downloaded gene_info.gz (skips the NCBI "
                             "download — use this if compute nodes lack network egress)")
    parser.add_argument("--skip-uniprot", action="store_true", help="NCBI Gene only")
    parser.add_argument("--keep-gene-info", action="store_true",
                        help="Don't delete the downloaded gene_info.gz afterward")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=" * 70)
    logger.info("Building Bean Lab Gene Index (NCBI Gene + UniProt, taxid 3885)")
    logger.info("=" * 70)

    tmp_download = None
    if args.gene_info_path:
        gene_info_path = Path(args.gene_info_path)
        if not gene_info_path.exists():
            raise SystemExit(f"--gene-info-path not found: {gene_info_path}")
        logger.info(f"Using existing gene_info file: {gene_info_path}")
    else:
        tmp_download = Path(tempfile.mkdtemp()) / "gene_info.gz"
        try:
            _download(NCBI_GENE_INFO_URL, tmp_download, logger)
        except Exception as e:
            raise SystemExit(
                f"Failed to download {NCBI_GENE_INFO_URL}: {e}\n\n"
                "If HPCC compute nodes block outbound HTTPS, download it on a login "
                "node instead:\n"
                "  wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz\n"
                "then re-run this script with --gene-info-path pointing at the "
                "downloaded file."
            )
        gene_info_path = tmp_download

    gene_id_map = parse_ncbi_gene_info(gene_info_path, logger)

    if tmp_download and not args.keep_gene_info:
        try:
            shutil.rmtree(tmp_download.parent)
            logger.info("Cleaned up temporary gene_info.gz download")
        except Exception as e:
            logger.warning(f"Could not remove temp download: {e}")

    if not args.skip_uniprot:
        try:
            uniprot_entries = fetch_uniprot(logger)
            gene_id_map.update(uniprot_entries)
        except Exception as e:
            logger.warning(f"UniProt fetch failed, continuing with NCBI data only: {e}")

    symbol_map = build_symbol_map(gene_id_map)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({
            "format": "bean-gene-index-v1",
            "taxid": TAXID,
            "gene_id_map": gene_id_map,
            "symbol_map": symbol_map,
        }, f)

    size_mb = output_path.stat().st_size / 1e6
    ncbi_count = sum(1 for v in gene_id_map.values() if v["source"] == "ncbi")
    uniprot_count = sum(1 for v in gene_id_map.values() if v["source"] == "uniprot")

    logger.info("=" * 70)
    logger.info(f"✓ Gene index saved to {output_path} ({size_mb:.1f} MB)")
    logger.info(f"  Total gene/protein records: {len(gene_id_map)}")
    logger.info(f"  Total indexed symbols/synonyms: {len(symbol_map)}")
    logger.info(f"  From NCBI: {ncbi_count}  |  From UniProt: {uniprot_count}")
    logger.info("=" * 70)

    print(f"\n✓ Gene index built: {len(gene_id_map)} records, {len(symbol_map)} symbols "
          f"({ncbi_count} NCBI + {uniprot_count} UniProt) -> {output_path}")


if __name__ == "__main__":
    main()
