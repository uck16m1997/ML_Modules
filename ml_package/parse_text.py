from ml_package import *


def get_info(df, name_col, func="Title"):
    def get_gender(s):
        m = re.search("(Sir|Lady|Mrs?|Ms|Miss)\.", s)
        if m:
            if m.group(1) == "Mr" or m.group(1) == "Sir":
                return "Male"
            else:
                return "Female"
        else:
            # print(s)
            return np.nan

    def get_title(s):
        m = re.search("\s([A-Za-z]*)\.", s)
        return m.group(1)

    if func == "Name":
        f = get_gender
    elif func == "Title":
        f = get_title

    return df[name_col].apply(lambda s: f(s))
