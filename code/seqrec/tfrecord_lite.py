"""Minimal reader/writer for gzip-compressed TFRecord files of tf.train.Example (the GRID/P5 splits used by TGH),
without TensorFlow. Supports the int64_list, float_list and bytes_list features those files use."""
import gzip
import struct

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)


def _crc32c(data):
    c = 0xFFFFFFFF
    for b in data:
        c = _CRC_TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _masked_crc(data):
    c = _crc32c(data)
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(buf, i):
    out = shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return out, i


def _fields(buf):
    """(field number, wire type, value) of a protobuf message; length-delimited values as bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield num, wt, v


def _feature(buf):
    for num, _, v in _fields(buf):
        if num == 1:  # BytesList
            return [x for n, _, x in _fields(v) if n == 1]
        if num == 2:  # FloatList (packed or not)
            out = []
            for n, wt, x in _fields(v):
                out += list(struct.unpack(f"<{len(x) // 4}f", x)) if wt == 2 else [struct.unpack("<f", x)[0]]
            return out
        if num == 3:  # Int64List (packed or not)
            out = []
            for n, wt, x in _fields(v):
                if wt == 2:
                    j = 0
                    while j < len(x):
                        y, j = _varint(x, j)
                        out.append(y - (1 << 64) if y >= 1 << 63 else y)
                else:
                    out.append(x - (1 << 64) if x >= 1 << 63 else x)
            return out
    return []


def parse_example(buf):
    """tf.train.Example bytes -> {feature name: list of values}."""
    out = {}
    for num, _, features in _fields(buf):
        if num != 1:
            continue
        for n, _, entry in _fields(features):
            if n != 1:
                continue
            key = val = None
            for m, _, x in _fields(entry):
                if m == 1:
                    key = x.decode()
                elif m == 2:
                    val = _feature(x)
            out[key] = val
    return out


def read(path):
    """Yield the parsed examples of one .tfrecord.gz file (record CRCs are not checked)."""
    with gzip.open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            n = struct.unpack("<Q", head[:8])[0]
            data = f.read(n)
            f.read(4)
            yield parse_example(data)


def _encode_varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(num, payload):
    return _encode_varint((num << 3) | 2) + _encode_varint(len(payload)) + payload


def encode_example(features):
    """{name: list of int | list of bytes} -> tf.train.Example bytes (int64_list or bytes_list)."""
    entries = b""
    for key, vals in features.items():
        if vals and isinstance(vals[0], (bytes, bytearray)):
            feat = _ld(1, b"".join(_ld(1, bytes(v)) for v in vals))
        else:
            feat = _ld(3, _ld(1, b"".join(_encode_varint(int(v) & ((1 << 64) - 1)) for v in vals)))
        entries += _ld(1, _ld(1, key.encode()) + _ld(2, feat))
    return _ld(1, entries)


def write(path, examples):
    """Write encoded examples as a gzip-compressed TFRecord file."""
    with gzip.open(path, "wb") as f:
        for data in examples:
            n = struct.pack("<Q", len(data))
            f.write(n + struct.pack("<I", _masked_crc(n)) + data + struct.pack("<I", _masked_crc(data)))
