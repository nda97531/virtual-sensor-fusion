def rreplace(s: str, old: str, new: str, occurrence: int = 1):
    """
    Replace last occurrence of a substring
    Args:
        s: the whole string
        old: old substring
        new: new substring
        occurrence: number of occurrences

    Returns:
        new string
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)


if __name__ == '__main__':
    print(rreplace(
        s='abcdefabcdefabcdef',
        old='a',
        new='A',
        occurrence=2
    ))
