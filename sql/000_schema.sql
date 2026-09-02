/*
name: schemas
purpose: Namespaces for the layered warehouse. raw = as loaded from silver,
         stg = typed/renamed staging, feat = feature tables (one grain each),
         mart = model-ready wide tables.
*/
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS feat;
CREATE SCHEMA IF NOT EXISTS mart;
