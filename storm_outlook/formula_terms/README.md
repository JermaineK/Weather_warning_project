# Formula term code (vendored, unchanged)

These scripts are byte-identical copies of the research term code used in every formula test (GEFS_PORT, RESCALED_SPIN, LEAD_TIME, ENSEMBLE).
`formula.py` runs only their definitions (everything before `def days():`) to compute one day's terms from GEFS; nothing here is run as a script.
They read the ERA5 ARCO bucket's metadata and land-sea mask once at start-up (public, read-only); all weather fields come from GEFS.

SHA-256:
- `gefs_terms_m.py` 9ad57592a09acb6bd4f29e72acd7d065b53673cb563f33de323d243af212c47a
- `feed_stream.py` 7eb94e4090d0cb44faf30a57592fa78eb8a87829d9d99e0ed1cad64bbbde99b1
- `gpi_stream.py` 348a6b8ac8c13b62bead98c75eab17a6aaef20bff0f0644a91b8b1119f1aaaae
- `basin_stream.py` 021960e64a055a730b8523bef6e30e838c66b24ea59013af2ebfa3906e3e782e
- `trough_stream.py` aa5a463028865fa2865638e30aac444e951faab16c2761937f32e07c171c6e34
- `size_stream.py` 723e1d970da8bab7863478e77735811f5f78fb7c0a1ee23ac314a92b502b3350
- `gpi6_stream.py` 13a8bc0a3412440cc861586b1c73e83feac538f3ba3ed84ceda4d19a5a1e11bb
- `gpi8_stream.py` 4cf66e2646ec75150bd44c8a94e4c16eaade96f25b89a87ad862aec49bd79b86
- `gpi10_stream.py` ffa32606aab5ede08cb44d9eba76b4fdb6dbef02d70ee8880db5cbb87747fa5f

Licence: PolyForm Noncommercial 1.0.0 (see ../LICENSE.md).
