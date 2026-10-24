"""Shared helpers for streaming analysis of sweet.tv files."""
import datetime as dt

DATA_DIR_CANDIDATES = []


def iso_week(d: dt.date) -> int:
    return d.isocalendar()[1]


def parse_view_log_ts(s: str) -> dt.datetime:
    # "2020-03-09 00:00:02"
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def parse_schedule_ts(s: str) -> dt.datetime:
    # "09.03.2020 06:00:00"
    return dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")
