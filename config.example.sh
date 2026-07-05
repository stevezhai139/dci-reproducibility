#!/usr/bin/env bash
# ── Local connection config for the end_to_end/ runners ──
# Copy this file to  config.local.sh , set PGUSER for your machine, then run:
#     source config.local.sh
# before the end_to_end scripts.  (config.local.sh is git-ignored, so your
# credentials never leave your machine.)  Defaults to  postgres@localhost:5432 .
export PGUSER="${PGUSER:-postgres}"          # your PostgreSQL role
# export DEXTER_BIN="/opt/homebrew/bin/dexter"  # if Dexter is not on PATH
