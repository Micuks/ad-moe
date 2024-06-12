import sys

incremental_metrics = [
    '"grow"',
    '"insert"',
    '"update"',
    '"delete"',
    '"comm"',
    '"roll"',
    '"heapr"',
    '"heaph"',
]


def get_ts(l):
    return int(l.split(",")[0])


def rm_ts(l):
    return l[l.find(",") :]


f = open(sys.argv[1], "r")
ls1 = f.readlines()
f.close()
f = open(sys.argv[2], "r")
ls2 = f.readlines()
f.close()

len_header = 5

res = ls1[: len_header - 1]
res.append(ls1[len_header - 1][:-1] + rm_ts(ls2[len_header - 1]))
res.append(ls1[len_header][:-1] + rm_ts(ls2[len_header]))
head = res[len_header].strip().split(",")[1:]
label = []
i = 0
j = 0
pre_t = get_ts(ls1[len_header + 1])
while i < len(ls1) - (len_header + 1) and j < len(ls2) - (len_header + 1):
    t1 = get_ts(ls1[i + len_header + 1])
    t2 = get_ts(ls2[j + len_header + 1])
    while t2 < pre_t:
        j += 1
        t2 = get_ts(ls2[j + len_header + 1])
    # print(f"ts1[{t1}], ts2[{t2}]")
    if abs(t1 - t2) < 5:
        if t1 - pre_t < 10:
            res.append(ls1[i + len_header + 1][:-1] + rm_ts(ls2[j + len_header + 1]))
            i += 1
            j += 1
            pre_t = t1
            label.append("0\n")
        else:
            last = ls1[i + len_header + 1 - 1][:-1] + rm_ts(ls2[j + len_header + 1 - 1])
            last = list(map(float, rm_ts(last).strip().split(",")[1:]))
            next = ls1[i + len_header + 1][:-1] + rm_ts(ls2[j + len_header + 1])
            next = list(map(float, rm_ts(next).strip().split(",")[1:]))
            tmp = []
            for k in range(len(last)):
                if head[k] in incremental_metrics:
                    tmp.append(next[k])
                else:
                    tmp.append((last[k] + next[k]) / 2)
            cur = ""
            for metric in tmp:
                cur += str(metric)
                cur += ","
            cur = cur[:-1]
            pre_t += 5
            res.append(str(pre_t) + "," + cur + "\n")
            label.append("b\n")
    elif t1 > t2:
        last = list(
            map(float, rm_ts(ls1[i + len_header + 1 - 1]).strip().split(",")[1:])
        )
        next = list(map(float, rm_ts(ls1[i + len_header + 1]).strip().split(",")[1:]))
        tmp = [(last[k] + next[k]) / 2 for k in range(len(last))]
        cur = ""
        for metric in tmp:
            cur += str(metric)
            cur += ","
        cur = cur[:-1]
        res.append(str(t2) + "," + cur + rm_ts(ls2[j + 6]))
        j += 1
        pre_t = t2
        label.append("x\n")
    else:
        last = list(
            map(float, rm_ts(ls2[j + len_header + 1 - 1]).strip().split(",")[1:])
        )
        next = list(map(float, rm_ts(ls2[j + len_header + 1]).strip().split(",")[1:]))
        tmp = []
        for k in range(len(last)):
            if head[k + 97] in incremental_metrics:
                tmp.append(next[k])
            else:
                tmp.append((last[k] + next[k]) / 2)
        cur = ""
        for metric in tmp:
            cur += str(metric)
            cur += ","
        cur = cur[:-1]
        res.append(ls1[i + len_header + 1][:-1] + "," + cur + "\n")
        i += 1
        pre_t = t1
        label.append("y\n")
f = open(sys.argv[1].replace("metricsx", "merge"), "w")
f.writelines(res)
f = open(sys.argv[1].replace("metricsx", "label"), "w")
f.writelines(label)
f.close()
