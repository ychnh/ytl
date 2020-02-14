def get_intervals(Subline):
    intervals = []

    Subline.sort()
    first, last = Subline[0], Subline[-1]

    i = []
    for s in Subline:
        if s == first:
            i.append(s)
        elif s == last:
        i.append(s)
            intervals.append(i)
        elif s - i[-1]!= 1:
            intervals.append(i)
            i = [s]
        else:
            i.append(s)
    return intervals
