import sys
import pandas as pd
import re
import io
from datetime import datetime
from datetime import timedelta


def shift_time(date: str, timezone_offset: int = 0) -> str:
    date_format = "%Y-%m-%d %H:%M:%S"
    date_obj = datetime.strptime(date, date_format)

    # print(f"before: {date_obj.strftime(date_format)}", end=" ")
    date_obj = date_obj - timedelta(minutes=timezone_offset * 60)
    # print(f"after: {date_obj.strftime(date_format)}")

    return date_obj.strftime(date_format)


def to_unix(date: str, timezone_offset: int = 0) -> int:
    """
    Convert date in 2024-03-05 12:23:34 format to UNIX timestamp format.
    And handle time zone issue.
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    date_obj = datetime.strptime(date, date_format)

    date_obj = date_obj - timedelta(minutes=timezone_offset * 60)

    return date_obj.timestamp()


class AnomalyMarker:
    def __init__(self) -> None:
        self.patterns = [
            (
                re.compile(r"^fault_IO_begin_(\d)\s*(.+)$"),
                re.compile(r"^fault_IO_end_(\d)\s*(.+)$"),
            ),
            (
                re.compile(r"^fault_knob_\d_(\d)_begin\s*(.+)$"),
                re.compile(r"^fault_knob_\d_(\d)_end\s*(.+)$"),
            ),
        ]

    def get_anomaly_time_slices(self, ls: list):
        time_slices = []
        for reg_begin, reg_end in self.patterns:
            active_anomalies = {}
            for line in ls:
                begin_match = reg_begin.match(line)
                if begin_match:
                    id, time = begin_match.groups()
                    active_anomalies[id] = (time, None)
                    continue

                end_match = reg_end.match(line)
                if end_match:
                    id, time = end_match.groups()
                    active_anomalies[id] = (active_anomalies[id][0], time)

                    time_slices.append(active_anomalies[id])
                    active_anomalies.pop(id)

        return time_slices


def mark_anomalies(df: pd.DataFrame, anomaly_time_slices: list):
    df["label"] = 0
    global timezone_shift
    for begin, end in anomaly_time_slices:
        shifted_begin = shift_time(begin, timezone_shift)
        shifted_end = shift_time(end, timezone_shift)
        ts_a = int(to_unix(shifted_begin))
        ts_b = int(to_unix(shifted_end))
        df.loc[(df["time"] >= ts_a) & (df["time"] < ts_b), "label"] = 1
        print(f"Mark anomaly from {shifted_begin}({ts_a}) to {shifted_end}({ts_b})")


def main():
    csv_filename = sys.argv[1]
    logs_filename = sys.argv[2]
    global timezone_shift
    timezone_shift = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    print(f"Apply timezone shift {timezone_shift}")

    df = pd.read_csv(csv_filename, header=5)
    with io.open(logs_filename, "r") as f:
        ls = f.readlines()

    marker = AnomalyMarker()
    anomaly_time_slices = marker.get_anomaly_time_slices(ls)
    mark_anomalies(df, anomaly_time_slices)

    new_filename = f"{csv_filename.rsplit('.csv',1)[0]}_label.csv"
    df.to_csv(new_filename, index=False)


if __name__ == "__main__":
    main()
