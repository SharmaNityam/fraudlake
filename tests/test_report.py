from fraudlake.report import parse_sql_file


def test_sql_header_and_feature_annotations_are_parsed(settings):
    parsed = parse_sql_file(settings.sql_dir / "110_fct_velocity.sql")
    assert parsed["name"] == "feat.fct_velocity"
    assert "1 second PRECEDING" in parsed["leakage"]
    names = [n for n, _ in parsed["features"]]
    assert "vel_cnt_24h" in names and "card_txn_idx" in names
