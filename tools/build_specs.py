"""
Best-effort extraction of dimensional specs + technical drawings from the
GALLA/UNICOM inspection-report XLS sheets in ../target-objects, one per SKU.

    python tools/build_specs.py

Writes, for every SKU in PART_TO_CLASS below:

    assets/references/<class>/spec.json      nominal + tolerance per named
                                              dimension, parsed from the sheet
    assets/references/<class>/drawing.png    the sheet's embedded technical
                                              drawing (kept for humans; the
                                              classifier never reads it)

The old .xls format stores embedded pictures as Escher BLIP records inside
the Workbook BIFF stream, and any record over 8224 bytes gets split across
CONTINUE records with their own 4-byte headers spliced into the raw bytes.
That's why xlrd/PIL alone can't see the picture: it has to be reassembled
BIFF-record-by-BIFF-record first, THEN searched for a PNG/JPEG signature.

This is a "draft" generator, not a source of truth: every spec.json is
written with "review_status": "draft" precisely because sheet layout,
column usage (see _split_label) and tolerance notation are not fully
uniform across files. Open each spec.json next to its drawing.png, check
the parsed dimensions against the drawing, fix anything wrong, and only
then flip review_status to "reviewed".
"""

import json
import re
import struct
from pathlib import Path

import olefile
import xlrd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TARGET_OBJECTS_DIR = PROJECT_DIR.parent / "target-objects"
REFERENCES_DIR = PROJECT_DIR / "assets" / "references"

# Which XLS becomes which classifier class label. See README for how this
# mapping was derived (from photo content + the sheet's own material/item
# fields) -- it is a best-effort draft, not verified against physically
# marked parts.
PART_TO_CLASS = {
    "98003P.XLS": "98003P",
    "98475.XLS": "98475",
    "98661BBS-1.XLS": "98661BBS-1",
    "98715PS-0.XLS": "98715PS-0",
    "UH-004.XLS": "UH-004",
    "UH-8715P.XLS": "UH-8715P",
}

HEADER_LABELS = {
    "客        戶": "customer", "客戶": "customer",
    "機        種": "type", "機種": "type",
    "  供應商": "supplier", "供應商": "supplier",
    "品        名": "item_name", "品名": "item_name",
    "料        號": "part_no", "料號": "part_no",
    "材    质": "material", "材质": "material",
}


# ----------------------------------------------------------------------
# BIFF / Escher drawing extraction
# ----------------------------------------------------------------------

def _biff_records(data):
    """Walk the raw Workbook stream, splicing CONTINUE records back into
    the record they continue. Returns a list of (type, data) tuples."""
    offset, records, n = 0, [], len(data)
    while offset + 4 <= n:
        rtype, rlen = struct.unpack("<HH", data[offset:offset + 4])
        offset += 4
        rdata = data[offset:offset + rlen]
        offset += rlen
        if rtype == 0x3C and records:  # CONTINUE
            records[-1][1] += rdata
        else:
            records.append([rtype, rdata])
    return records


def _find_images(blob):
    """Return every (signature_end_included_length, start, ext) PNG/JPEG
    found in blob, biggest first."""
    found = []
    for start in (m.start() for m in re.finditer(rb"\x89PNG\r\n\x1a\n", blob)):
        end = blob.find(b"IEND", start)
        if end != -1:
            found.append((end + 8 - start, start, end + 8, "png"))
    for start in (m.start() for m in re.finditer(rb"\xff\xd8\xff", blob)):
        end = blob.find(b"\xff\xd9", start)
        if end != -1:
            found.append((end + 2 - start, start, end + 2, "jpg"))
    found.sort(reverse=True)
    return found


def extract_drawing(xls_path, out_path):
    """
    Best-effort: pull the embedded technical-drawing PNG/JPEG out of an
    old-format .xls file and save it to out_path. Returns the path written,
    or None if no raster image could be found (the sheet may have drawn the
    picture with native Excel shapes instead of pasting a bitmap -- there is
    nothing to extract in that case).

    Tries two levels of BIFF reassembly, cheapest first:
      1. per-record: merge each record with the CONTINUEs that immediately
         follow it (correct whenever one MSODRAWING record holds the whole
         picture).
      2. cross-record: concatenate every MSODRAWINGGROUP (0xEB) and
         MSODRAWING (0xEC) record's reassembled bytes, in file order, before
         searching (needed when a picture is split across more than one of
         those top-level records).
    """
    ole = olefile.OleFileIO(str(xls_path))
    workbook = ole.openstream("Workbook").read()
    records = _biff_records(workbook)

    for _rtype, rdata in records:
        images = _find_images(rdata)
        if images:
            _length, start, end, ext = images[0]
            out_path = out_path.with_suffix("." + ext)
            out_path.write_bytes(rdata[start:end])
            return out_path

    combined = b"".join(rdata for rtype, rdata in records if rtype in (0xEB, 0xEC))
    images = _find_images(combined)
    if images:
        _length, start, end, ext = images[0]
        out_path = out_path.with_suffix("." + ext)
        out_path.write_bytes(combined[start:end])
        return out_path

    return None


# ----------------------------------------------------------------------
# Dimension parsing
# ----------------------------------------------------------------------

def _is_code(s):
    """A short drawing-style label such as 'D', 'D1', 'T', not a phrase."""
    s = s.strip()
    return bool(s) and len(s) <= 4 and not any("一" <= ch <= "鿿" for ch in s)


def split_label(col1, col2):
    """
    The item-name column is inconsistent across sheets: sometimes col1 is
    the drawing code ('D', 'L'), sometimes it's a Chinese description
    ('板厚') and the actual drawing code sits in col2 ('T'). Prefer
    whichever column looks like a short code; keep the other as `name`.
    """
    c1, c2 = str(col1).strip(), str(col2).strip()
    if _is_code(c2):
        return c2, (c1 if c1 and c1 != c2 else None)
    if _is_code(c1):
        return c1, (c2 if c2 and c2 != c1 else None)
    return (c1 or c2 or "?"), None


def parse_tolerance(raw):
    """
    Parse one "規格值 / Description Value" cell into nominal + tolerance.
    Returns None when the text is not a numeric dimension at all (e.g.
    "無变型、缺胶等缺陷") -- those rows become qc_notes instead.
    """
    s = raw.strip()
    if not s:
        return None

    diameter = s[:1] in ("ψ", "φ", "Ø", "ø")  # ψ φ Ø ø
    if diameter:
        s = s[1:].strip()
    # "mm" can sit before a trailing "MAX."/"MIN." ("3.5mm MAX."), not just at
    # the very end, so strip it wherever it appears rather than anchoring to $.
    s = re.sub(r"mm", "", s, flags=re.IGNORECASE).strip()

    m = re.match(r"^([\d.]+)\s*MAX\.?$", s, re.IGNORECASE)
    if m:
        nominal = float(m.group(1))
        return {"nominal": nominal, "upper_tol": None, "lower_tol": None,
                "min": None, "max": nominal, "tolerance_type": "max_only",
                "diameter": diameter}

    m = re.match(r"^([\d.]+)\s*MIN\.?$", s, re.IGNORECASE)
    if m:
        nominal = float(m.group(1))
        return {"nominal": nominal, "upper_tol": None, "lower_tol": None,
                "min": nominal, "max": None, "tolerance_type": "min_only",
                "diameter": diameter}

    m = re.match(r"^([\d.]+)\s*±\s*([\d.]+)$", s)
    if m:
        nominal, tol = float(m.group(1)), float(m.group(2))
        return {"nominal": nominal, "upper_tol": tol, "lower_tol": -tol,
                "min": round(nominal - tol, 6), "max": round(nominal + tol, 6),
                "tolerance_type": "symmetric", "diameter": diameter}

    m = re.match(r"^([\d.]+)\s*\+\s*([\d.]+)\s*/\s*-\s*([\d.]+)$", s)
    if m:
        nominal, up, lo = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return {"nominal": nominal, "upper_tol": up, "lower_tol": -lo,
                "min": round(nominal - lo, 6), "max": round(nominal + up, 6),
                "tolerance_type": "asymmetric", "diameter": diameter}

    m = re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return {"nominal": round((lo + hi) / 2, 6), "upper_tol": None, "lower_tol": None,
                "min": lo, "max": hi, "tolerance_type": "range", "diameter": diameter}

    return None


def parse_xls(xls_path):
    book = xlrd.open_workbook(str(xls_path))
    sheet = book.sheets()[0]
    rows = [[sheet.cell_value(r, c) for c in range(sheet.ncols)]
            for r in range(sheet.nrows)]

    header = {}
    for row in rows:
        for i, cell in enumerate(row):
            key = HEADER_LABELS.get(str(cell).strip())
            if key and i + 1 < len(row) and str(row[i + 1]).strip():
                header.setdefault(key, str(row[i + 1]).strip())

    table_start = None
    for i, row in enumerate(rows):
        if any("Inspection Item" in str(c) for c in row):
            table_start = i + 1
            break
    if table_start is None:
        raise ValueError(f"could not find the inspection-item table in {xls_path.name}")

    dimensions, qc_notes = [], []
    for row in rows[table_start:]:
        no = row[0]
        if not isinstance(no, float):
            break  # end of the numbered table (signature block etc.)
        label, name = split_label(row[1], row[2])
        raw_spec = str(row[3]).strip()
        parsed = parse_tolerance(raw_spec)
        if parsed is None:
            qc_notes.append({"item": str(row[1]).strip() or label, "requirement": raw_spec})
        else:
            entry = {"label": label, "raw_spec": raw_spec, **parsed}
            if name:
                entry["name"] = name
            dimensions.append(entry)

    return header, dimensions, qc_notes


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def build_one(xls_name, class_label):
    xls_path = TARGET_OBJECTS_DIR / xls_name
    class_dir = REFERENCES_DIR / class_label
    class_dir.mkdir(parents=True, exist_ok=True)

    header, dimensions, qc_notes = parse_xls(xls_path)

    drawing_path = extract_drawing(xls_path, class_dir / "drawing")
    drawing_name = drawing_path.name if drawing_path else None

    spec = {
        "sku": class_label,
        "source_file": f"target-objects/{xls_name}",
        "item_name": header.get("item_name"),
        "part_no": header.get("part_no") or class_label,
        "material": header.get("material"),
        "unit": "mm",
        "drawing": drawing_name,
        "review_status": "draft",
        "dimensions": dimensions,
        "qc_notes": qc_notes,
    }

    spec_path = class_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{class_label:<12} {len(dimensions):>2} dimensions, {len(qc_notes):>2} qc notes, "
          f"drawing={'yes' if drawing_name else 'MISSING'} -> {spec_path.relative_to(PROJECT_DIR)}")


def main():
    if not TARGET_OBJECTS_DIR.exists():
        raise SystemExit(f"target-objects folder not found at {TARGET_OBJECTS_DIR}")
    for xls_name, class_label in PART_TO_CLASS.items():
        build_one(xls_name, class_label)


if __name__ == "__main__":
    main()
