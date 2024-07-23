import regex
import ast


def parse_wolfram_fmt_inputfile(filename: str):
    with open(filename, "r") as f:
        data = f.read()
        return _parse_wolfram_fmt_recursive(data)


def _parse_wolfram_fmt_recursive(data: str):
    res = {}
    reg_pattern = r"\"\w+\" ?-> \({(?R)}|\".*\"\)|.*;?(?=[ \"])"
    # reg_pattern = regex.compile(r"\(\"\w+\" ?-> [{\"]?\(?R\)?[}\"]?\|.*\)",
    #                             flags=regex.D)
    inner = regex.findall(reg_pattern, data)
    # we reached a "base"
    print(inner)
    if inner is None:
        return None
    return
    for name, val in inner:
        print(name)
        print(val)
        parsed_val = _parse_wolfram_fmt_recursive(val)
        print("parsed_val:\n", parsed_val)
        if parsed_val is None:
            parsed_val = ast.literal_eval(
                val.translate(
                    str.maketrans({"{": "[", "}": "]"})
                ))
            assert (type(parsed_val) is list)
        res[name] = parsed_val
    return res
